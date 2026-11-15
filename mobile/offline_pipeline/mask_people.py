"""
Masks people out of an Atlas Mobile session's COLMAP model with SAM 3, run by
Spirula Studio (https://github.com/harry7557558/spirula-studio), and the
area outside each 360 lens with Spirula's own border mask.

Every image in <session>/colmap/images/ (iPhone photos and 360 views) is
searched for people. Their pixels turn black in the image's mask in
colmap/masks/ (white = keep, as COLMAP and Spirula read masks). A mask the
export already wrote is kept: the new mask removes both. iPhone photos are
searched for "person"; 360 images with the prompts of Spirula's 360-camera
dataset preset (whoever holds the camera, what they carry and their shadow;
pictures and statues of people are kept).

The lens images the app cuts out of each 360 photo (colmap/images/fisheye_*)
also lose what lies outside the lens circle: `spirula sam mask` fits the
circle over each folder's images and pulls it in by 1%, Spirula's "Remove
fixed areas of the frame" with its default settings.

An iPhone photo (colmap/images/face_iphone/scan_NNN.jpg) with a person in it
also loses its LiDAR: that scan's points leave colmap/sparse/0/points3D.bin,
and its depth files (colmap/depths/scan_NNN_*, colmap/depth_images/*/
scan_NNN.png, enhanced/depth_dense/scan_NNN.png) are moved out. LiDAR frames
saved between photos (raw/lidar_frames, app build 12m) have no photo to check
and stay.

Everything changed is first copied to <session>/people_mask_backup/. A second
run starts again from those originals, and --undo puts them back.

The SAM 3 checkpoint is the one Spirula Studio's GUI downloads (on Windows
into %LOCALAPPDATA%\\spirula-studio\\models\\). Loading it accepts Meta's SAM 3
licence, so this passes --accept-license sam3=yes.

Usage:
    python mask_people.py --session-dir A:\\d\\x\\session_2026-11-14_18-14-31
    python mask_people.py --session-dir A:\\d\\x\\session_2026-11-14_18-14-31 --undo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

BACKUP_DIR = "people_mask_backup"
WORK_DIR = "people_mask_work"
MANIFEST = "manifest.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IPHONE_DIR = "face_iphone"
# The app's folders of lens images cut out of the 360 photos.
FISHEYE_PREFIX = "fisheye_"
# Spirula Studio's "360-camera" dataset preset (src/app/gui/DatasetPreset.cpp):
# whoever holds a 360 camera is in every frame of it, and so are whatever they
# carry and their shadow; people that are pictures or carvings are kept.
PROMPT_360 = ("tourist; visitor; pedestrian; photographer; child; person; hand; "
              "backpack; selfie stick; shadow of person")
NEGATIVE_360 = "painting; statue; poster"
# Checkpoints Spirula Studio downloads, best first (q4_0 is a smaller file
# of the same model; both load as fp16).
MODEL_FILES = ("sam3-f16.ggml", "sam3-q4_0.ggml")

# A points3D.bin record with no track: id, xyz, rgb, error, track length.
POINT = np.dtype([
    ("id", "<u8"), ("xyz", "<f8", (3,)), ("rgb", "u1", (3,)),
    ("error", "<f8"), ("track_len", "<u8"),
])

# COLMAPExporter.unprojectDepth: LiDAR map size, sample step, depth range.
LIDAR_W, LIDAR_H, LIDAR_STEP = 256, 192, 4
MIN_DEPTH, MAX_DEPTH = np.float32(0.1), np.float32(5.0)

Log = Callable[[str], None]


class MaskingUnavailable(RuntimeError):
    """Spirula or its SAM 3 checkpoint wasn't found."""


class LidarLayoutError(RuntimeError):
    """points3D.bin can't be split into the photos' LiDAR points."""


# ---------------------------------------------------------------------------
# Finding Spirula and the model
# ---------------------------------------------------------------------------

def find_spirula(explicit: str | None = None) -> Path:
    """--spirula, then SPIRULA_EXE, then PATH, then the newest spirula.exe
    on the desktop or in Downloads (the Windows release is a zip you unzip
    and run, so it is wherever it was unzipped)."""
    for given, what in ((explicit, "--spirula"),
                        (os.environ.get("SPIRULA_EXE"), "SPIRULA_EXE")):
        if given:
            path = Path(given)
            if path.is_file():
                return path
            raise MaskingUnavailable(f"{what} points at {path}, which doesn't exist")
    on_path = shutil.which("spirula")
    if on_path:
        return Path(on_path)
    exe = "spirula.exe" if os.name == "nt" else "spirula"
    found: list[Path] = []
    for base in (Path.home() / "Desktop", Path.home() / "Downloads"):
        found += [p for p in [base / exe, *base.glob(f"spirula*/{exe}")] if p.is_file()]
    if not found:
        raise MaskingUnavailable(
            "spirula not found: pass --spirula or set SPIRULA_EXE")
    return max(found, key=lambda p: p.stat().st_mtime)


def find_sam_model(explicit: str | None = None) -> Path:
    """--sam-model, then SPIRULA_SAM_MODEL, then Spirula Studio's model cache
    (src/app/AppPaths.cpp: LOCALAPPDATA on Windows, XDG_CACHE_HOME or
    ~/.cache elsewhere)."""
    for given, what in ((explicit, "--sam-model"),
                        (os.environ.get("SPIRULA_SAM_MODEL"), "SPIRULA_SAM_MODEL")):
        if given:
            path = Path(given)
            if path.is_file():
                return path
            raise MaskingUnavailable(f"{what} points at {path}, which doesn't exist")
    if os.name == "nt":
        bases = [Path(os.environ["LOCALAPPDATA"])] if os.environ.get("LOCALAPPDATA") else []
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        bases = [Path(xdg) if xdg else Path.home() / ".cache"]
    for base in bases:
        # spirulae-splat is the app's name before its rename.
        for app in ("spirula-studio", "spirulae-splat"):
            for name in MODEL_FILES:
                path = base / app / "models" / name
                if path.is_file():
                    return path
    raise MaskingUnavailable(
        "no SAM 3 checkpoint: download it in Spirula Studio (masking) or pass --sam-model")


def supports_image_mode(exe: Path) -> bool:
    """`sam track --mask-mode image` (Spirula v2026.11.12 and newer) masks
    each image on its own in one run. Older builds only track across frames,
    which carries a person's mask into the next, unrelated image."""
    proc = subprocess.run([str(exe), "sam", "--help"], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)
    return "--mask-mode" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Running SAM 3
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _run_spirula(cmd: list[str], log: Log | None = None, total: int = 0) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        tail = (tail + [line])[-15:]
        # "frame 12: 00012.jpg -- kept 95.6%"
        if log and total and line.startswith("frame ") and " -- " in line:
            number = line.split(":", 1)[0].split()[-1]
            if number.isdigit():
                done = int(number) + 1
                if done % 10 == 0 or done == total:
                    log(f"  SAM 3: {done}/{total} images")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"spirula exited with code {code}:\n" + "\n".join(tail))


def find_people(exe: Path, model: Path, images: list[Path], work: Path,
                prompt: str, jobs: int, log: Log,
                negative: str | None = None) -> list[Path | None]:
    """Masks `prompt` in every image, keeping what `negative` matches.
    Returns each image's mask file (white = keep, black = matched), or None
    where spirula wrote none."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    base = [str(exe), "sam", "track", "--model", str(model), "--text", prompt,
            "--accept-license", "sam3=yes"]
    if negative:
        base += ["--neg-text", negative]

    if supports_image_mode(exe):
        frames, out = work / "in", work / "out"
        frames.mkdir()
        out.mkdir()
        # Numbered names: spirula numbers its masks in the sorted file order.
        for i, image in enumerate(images):
            _link_or_copy(image, frames / f"{i:05d}{image.suffix.lower()}")
        log(f"  SAM 3: {len(images)} images, one run")
        _run_spirula(base + ["--mask-mode", "image", "--frames", str(frames),
                             "--out", str(out)], log, len(images))
        masks = [out / f"frame_{i:05d}.png" for i in range(len(images))]
    else:
        log(f"  SAM 3: {len(images)} images, one run each, {jobs} at a time "
            "(this spirula build has no --mask-mode; v2026.11.12 or newer "
            "does all of them in one run, about 10x faster)")

        def one(i: int) -> Path:
            frames, out = work / f"in_{i:05d}", work / f"out_{i:05d}"
            frames.mkdir()
            out.mkdir()
            _link_or_copy(images[i], frames / f"00000{images[i].suffix.lower()}")
            _run_spirula(base + ["--frames", str(frames), "--out", str(out)])
            return out / "frame_00000.png"

        masks = []
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for done, mask in enumerate(pool.map(one, range(len(images))), 1):
                masks.append(mask)
                if done % 10 == 0 or done == len(images):
                    log(f"  SAM 3: {done}/{len(images)} images")

    result: list[Path | None] = []
    for image, mask in zip(images, masks):
        if mask.is_file():
            result.append(mask)
        else:
            log(f"  warning: spirula wrote no mask for {image.name} (unreadable image?)")
            result.append(None)
    return result


def find_lens_edges(exe: Path, images_dir: Path, work: Path,
                    log: Log) -> dict[str, Path]:
    """Spirula's border mask for each folder of 360 lens images
    (images/fisheye_*): `spirula sam mask` with its defaults fits the lens
    circle over the folder's images (24 at most) and keeps what is inside
    it, pulled in by 1%. Returns each image's mask file by its path under
    images/ (white = inside the lens)."""
    shutil.rmtree(work, ignore_errors=True)
    found: dict[str, Path] = {}
    folders = sorted(p for p in images_dir.iterdir()
                     if p.is_dir() and p.name.startswith(FISHEYE_PREFIX))
    for folder in folders:
        out = work / folder.name
        out.mkdir(parents=True)
        proc = subprocess.run([str(exe), "sam", "mask", str(folder), "--out", str(out),
                               "--replace"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
        if proc.returncode != 0:
            raise RuntimeError(f"spirula sam mask exited with code {proc.returncode} on "
                               f"{folder.name}:\n" + "\n".join(lines[-15:]))
        # "border: ellipse 0.4987,0.5004,0.5078,0.5078 -- keeps 80.6% of the frame"
        # or "no border found in ...; name one with --shape"
        said = next((ln for ln in lines if ln.startswith(("border:", "no border"))),
                    "spirula said nothing about a border")
        log(f"  lens edge, {folder.name}: {said}")
        for image in folder.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                mask = out / f"{image.stem}.png"
                if mask.is_file():
                    found[f"{folder.name}/{image.name}"] = mask
    return found


# ---------------------------------------------------------------------------
# Backup of everything a run changes
# ---------------------------------------------------------------------------

def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images_bin_sha1(session: Path) -> str | None:
    path = session / "colmap" / "sparse" / "0" / "images.bin"
    return _sha1(path) if path.is_file() else None


class Backup:
    """people_mask_backup/: the original of every file a run changes, and
    manifest.json listing them. Each change is listed before it is made, so
    a run that stops half way can still be undone."""

    def __init__(self, session: Path):
        self.session = session
        self.root = session / BACKUP_DIR
        self.manifest = {
            "version": 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images_bin_sha1": _images_bin_sha1(session),
            "masks_modified": [],
            "masks_created": [],
            "points3d": None,
            "moved": [],
            "people": {},
            "lidar_removed": {},
        }

    def rel(self, path: Path) -> str:
        return path.relative_to(self.session).as_posix()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        os.replace(tmp, self.root / MANIFEST)

    def keep_original(self, path: Path) -> None:
        dst = self.root / self.rel(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    def will_write_mask(self, path: Path, existed: bool) -> None:
        if existed:
            self.keep_original(path)
        key = "masks_modified" if existed else "masks_created"
        self.manifest[key].append(self.rel(path))
        self.save()

    def will_write_points3d(self, path: Path) -> None:
        self.keep_original(path)
        self.manifest["points3d"] = {"path": self.rel(path), "written_sha1": None}
        self.save()

    def wrote_points3d(self, path: Path) -> None:
        self.manifest["points3d"]["written_sha1"] = _sha1(path)
        self.save()

    def move_out(self, path: Path) -> None:
        rel = self.rel(path)
        self.manifest["moved"].append(rel)
        self.save()
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dst))

    def finish(self, people: dict, lidar_removed: dict) -> None:
        if self.root.exists():
            self.manifest["people"] = people
            self.manifest["lidar_removed"] = lidar_removed
            self.save()


def restore(session: Path, log: Log, undo: bool = False) -> None:
    """Puts back the originals of the last run's changes and removes the
    backup. A backup made for another export (images.bin or points3D.bin
    changed since) is not restored: it is renamed and left alone."""
    root = session / BACKUP_DIR
    if not root.exists():
        if undo:
            log("nothing to undo: no people_mask_backup folder")
        return
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None

    reason = None
    if manifest is None:
        reason = "its manifest.json is missing"
    elif _images_bin_sha1(session) != manifest.get("images_bin_sha1"):
        reason = "colmap/sparse/0/images.bin changed since (a new export?)"
    elif manifest.get("points3d") and manifest["points3d"].get("written_sha1"):
        p3d = session / manifest["points3d"]["path"]
        if not p3d.is_file() or _sha1(p3d) != manifest["points3d"]["written_sha1"]:
            reason = "points3D.bin changed since"
    if reason:
        if undo:
            raise RuntimeError(f"can't undo: {reason}. The backup is in {root}")
        stale = root.with_name(f"{BACKUP_DIR}_stale_{time.strftime('%Y%m%d-%H%M%S')}")
        root.rename(stale)
        log(f"  an earlier backup doesn't match this export ({reason}); "
            f"kept it as {stale.name} and started from the current files")
        return

    for rel in manifest["masks_modified"]:
        if (root / rel).is_file():
            shutil.copy2(root / rel, session / rel)
    for rel in manifest["masks_created"]:
        (session / rel).unlink(missing_ok=True)
    if manifest.get("points3d"):
        rel = manifest["points3d"]["path"]
        if (root / rel).is_file():
            shutil.copy2(root / rel, session / rel)
    for rel in manifest["moved"]:
        src, dst = root / rel, session / rel
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    shutil.rmtree(root)
    log("  restored the files the last masking run changed")


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def mask_path_for(colmap: Path, rel: str) -> tuple[Path, bool]:
    """The image's mask file and whether it exists. The app writes
    masks/<dir>/scan_NNN.png; COLMAP's own naming, scan_NNN.jpg.png, and
    scan_NNN_mask.png are found too (Spirula reads all three)."""
    image = Path(rel)
    folder = colmap / "masks" / image.parent
    for name in (f"{image.stem}.png", f"{image.name}.png", f"{image.stem}_mask.png"):
        if (folder / name).is_file():
            return folder / name, True
    return folder / f"{image.stem}.png", False


def write_mask(colmap: Path, rel: str, person: np.ndarray | None,
               inside: np.ndarray | None, backup: Backup) -> None:
    """Takes `person` (True where a person is) and everything outside
    `inside` (True inside the lens) out of the image's mask; either may be
    None. A mask the image already has is kept and narrowed."""
    path, existed = mask_path_for(colmap, rel)
    shape = (person if person is not None else inside).shape
    keep = (np.asarray(Image.open(path).convert("L")) >= 128 if existed
            else np.ones(shape, dtype=bool))

    def fit(part: np.ndarray) -> np.ndarray:
        if part.shape == keep.shape:
            return part
        resized = Image.fromarray(part.astype(np.uint8) * 255).resize(
            (keep.shape[1], keep.shape[0]), Image.NEAREST)
        return np.asarray(resized) >= 128

    if person is not None:
        keep &= ~fit(person)
    if inside is not None:
        keep &= fit(inside)
    backup.will_write_mask(path, existed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(np.where(keep, 255, 0).astype(np.uint8)).save(tmp, format="PNG")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# LiDAR of photos with people
# ---------------------------------------------------------------------------

def read_points3d(path: Path) -> np.ndarray | None:
    """All points, or None when some point has a track (then the file isn't
    the export's LiDAR points)."""
    data = path.read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    if len(data) != 8 + POINT.itemsize * count:
        return None
    points = np.frombuffer(data, dtype=POINT, count=count, offset=8)
    return None if (points["track_len"] != 0).any() else points


def write_points3d(path: Path, points: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        f.write(np.ascontiguousarray(points).tobytes())
    os.replace(tmp, path)


def read_image_names(path: Path) -> list[str]:
    """Image names of images.bin, in file order."""
    data = path.read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    names = []
    for _ in range(count):
        offset += 4 + 7 * 8 + 4  # image id, qvec + tvec, camera id
        end = data.index(b"\x00", offset)
        names.append(data[offset:end].decode("utf-8"))
        n_points2d = struct.unpack_from("<Q", data, end + 1)[0]
        offset = end + 1 + 8 + 24 * n_points2d
    return names


def app_export_slices(session: Path, sparse: Path,
                      points: np.ndarray) -> dict[str, tuple[int, int]]:
    """Where each iPhone photo's LiDAR points are in the app's points3D.bin.

    COLMAPExporter writes, for each iPhone image in images.bin order, the
    depth.bin samples on a 4-pixel grid between 0.1 and 5 m, then the LiDAR
    frames' points. Counting the samples splits the file; recomputing every
    point's position checks the split."""
    raw = session / "raw" / "iphone"
    if not raw.is_dir():
        raise LidarLayoutError("raw/iphone is missing, so the points can't be matched to photos")
    names = read_image_names(sparse / "images.bin")
    scans = [Path(n).stem for n in names if n.startswith(IPHONE_DIR + "/")]

    # The exporter scales the first scan's intrinsics for every scan.
    first = None
    for scan_dir in sorted(raw.glob("scan_*")):
        try:
            first = json.loads((scan_dir / "pose.json").read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if first is None:
        raise LidarLayoutError("no readable pose.json in raw/iphone")
    K = np.array(first["intrinsics"], dtype=np.float64)
    fx = K[0, 0] * LIDAR_W / first["image_width"]
    fy = K[1, 1] * LIDAR_H / first["image_height"]
    cx = K[0, 2] * LIDAR_W / first["image_width"]
    cy = K[1, 2] * LIDAR_H / first["image_height"]

    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for scan in scans:
        depth_path = raw / scan / "depth.bin"
        depth = np.fromfile(depth_path, dtype="<f4") if depth_path.is_file() else np.zeros(0, "<f4")
        if depth.size < LIDAR_W * LIDAR_H:
            slices[scan] = (start, 0)  # the exporter adds no points
            continue
        grid = depth[: LIDAR_W * LIDAR_H].reshape(LIDAR_H, LIDAR_W)[::LIDAR_STEP, ::LIDAR_STEP]
        valid = (grid > MIN_DEPTH) & (grid < MAX_DEPTH)
        count = int(valid.sum())
        if start + count > len(points):
            raise LidarLayoutError(f"{scan}: points3D.bin has fewer points than the photos' LiDAR")
        if count:
            rows, cols = np.nonzero(valid)
            z = grid[rows, cols].astype(np.float64)
            cam = np.stack([(cols * LIDAR_STEP - cx) / fx * z,
                            -(rows * LIDAR_STEP - cy) / fy * z, -z])
            pose = json.loads((raw / scan / "pose.json").read_text(encoding="utf-8"))
            T = np.array(pose["transform_matrix"], dtype=np.float64)
            world = T[:3, :3] @ cam + T[:3, 3:4]
            colmap_xyz = (world * np.array([[1.0], [-1.0], [-1.0]])).T
            err = np.linalg.norm(colmap_xyz - points["xyz"][start:start + count], axis=1)
            if np.median(err) > 1e-3 or np.percentile(err, 95) > 5e-3:
                raise LidarLayoutError(
                    f"{scan}: the points at its place in points3D.bin aren't its LiDAR "
                    f"(median {1000 * np.median(err):.1f} mm off)")
        slices[scan] = (start, count)
        start += count
    extra = len(points) - start
    if extra and not (session / "raw" / "lidar_frames").is_dir():
        raise LidarLayoutError(f"{extra} points in points3D.bin come from no photo")
    return slices


def lidar_slices(session: Path, sparse: Path,
                 points: np.ndarray) -> dict[str, tuple[int, int]]:
    # assemble_colmap.py (host processing) lists its slices next to the file.
    sources = sparse / "points3D_sources.json"
    if sources.is_file():
        listing = json.loads(sources.read_text(encoding="utf-8"))
        if listing.get("total") != len(points):
            raise LidarLayoutError("points3D_sources.json doesn't match points3D.bin")
        return {s["scan"]: (s["first"], s["count"]) for s in listing["scans"]}
    return app_export_slices(session, sparse, points)


def depth_files(session: Path, scan: str) -> list[Path]:
    """The scan's depth files: colmap/depths/scan_NNN_depth_smoothed.bin
    (app), colmap/depth_images/*/scan_NNN.png and enhanced/depth_dense/
    scan_NNN.png (host processing)."""
    found = []
    for root in (session / "colmap" / "depths", session / "colmap" / "depth_images",
                 session / "enhanced" / "depth_dense"):
        if root.is_dir():
            found += [f for f in root.rglob("*") if f.is_file()
                      and (f.stem == scan or f.name.startswith(scan + "_"))]
    return sorted(found)


def remove_lidar(session: Path, scans: list[str], backup: Backup,
                 log: Log) -> dict[str, int]:
    """Takes the scans' LiDAR points out of points3D.bin and moves their
    depth files to the backup. Returns the points removed per scan."""
    removed: dict[str, int] = {}
    sparse = session / "colmap" / "sparse" / "0"
    p3d = sparse / "points3D.bin"
    if p3d.is_file():
        points = read_points3d(p3d)
        if points is None:
            log("  points3D.bin has feature tracks, so it isn't the LiDAR export; left as is")
        else:
            try:
                slices = lidar_slices(session, sparse, points)
            except LidarLayoutError as e:
                log(f"  warning: LiDAR points left in points3D.bin: {e}")
                slices = {}
            keep = np.ones(len(points), dtype=bool)
            for scan in scans:
                if scan in slices:
                    first, count = slices[scan]
                    keep[first:first + count] = False
                    removed[scan] = count
            if not keep.all():
                backup.will_write_points3d(p3d)
                write_points3d(p3d, points[keep])
                backup.wrote_points3d(p3d)
                log(f"  points3D.bin: {len(points)} -> {int(keep.sum())} points")
    for scan in scans:
        for f in depth_files(session, scan):
            backup.move_out(f)
            log(f"  moved out {backup.rel(f)}")
    return removed


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

def mask_session(session_dir: Path | str, spirula: str | None = None,
                 sam_model: str | None = None, prompt: str = "person",
                 jobs: int = 3, log: Log = print,
                 prompt_360: str = PROMPT_360,
                 negative_360: str | None = NEGATIVE_360,
                 lens_edge: bool = True) -> dict:
    """Masks people in the session's COLMAP images (`prompt` in the iPhone
    photos, `prompt_360` but not `negative_360` in the 360 images), and the
    area outside each 360 lens, and removes the LiDAR of iPhone photos
    showing people. Raises MaskingUnavailable when Spirula or the SAM 3
    checkpoint isn't found."""
    session = Path(session_dir).resolve()
    colmap = session / "colmap"
    images_dir = colmap / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no COLMAP images in {images_dir}")
    exe = find_spirula(spirula)
    model = find_sam_model(sam_model)
    log(f"  spirula: {exe}")
    log(f"  model:   {model}")

    images = sorted(p for p in images_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log("  no images to mask")
        return {"images": 0, "people": {}, "lidar_removed": {}}

    work = session / WORK_DIR
    iphone_images = [p for p in images if p.parent.name == IPHONE_DIR]
    images_360 = [p for p in images if p.parent.name != IPHONE_DIR]
    edges_kept: dict[str, list[float]] = {}
    try:
        mask_files: dict[Path, Path | None] = {}
        if iphone_images:
            log(f'  iPhone photos: SAM 3 masks "{prompt}"')
            mask_files.update(zip(iphone_images, find_people(
                exe, model, iphone_images, work / "iphone", prompt, jobs, log)))
        if images_360:
            log(f'  360 images: SAM 3 masks "{prompt_360}"'
                + (f', keeping "{negative_360}"' if negative_360 else "")
                + " (Spirula's 360-camera preset)")
            mask_files.update(zip(images_360, find_people(
                exe, model, images_360, work / "360", prompt_360, jobs, log,
                negative=negative_360)))
        edges = find_lens_edges(exe, images_dir, work / "edges", log) if lens_edge else {}
        # Only now, with the new masks ready, go back to the original files.
        restore(session, log)
        backup = Backup(session)
        people: dict[str, float] = {}
        for image in images:
            rel = image.relative_to(images_dir).as_posix()
            person = None
            mask_file = mask_files.get(image)
            if mask_file is not None:
                found = np.asarray(Image.open(mask_file).convert("L")) < 128
                if found.any():
                    person = found
                    people[rel] = round(100.0 * float(found.mean()), 2)
            inside = None
            if rel in edges:
                lens = np.asarray(Image.open(edges[rel]).convert("L")) >= 128
                if not lens.all():
                    inside = lens
                    edges_kept.setdefault(image.parent.name, []).append(
                        100.0 * float(lens.mean()))
            if person is not None or inside is not None:
                write_mask(colmap, rel, person, inside, backup)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    iphone = sorted(Path(rel).stem for rel in people if rel.startswith(IPHONE_DIR + "/"))
    lidar_removed = remove_lidar(session, iphone, backup, log) if iphone else {}
    backup.finish(people, lidar_removed)

    n_iphone = sum(1 for p in images if p.parent.name == IPHONE_DIR)
    found_iphone = len(iphone)
    log(f"  people in {len(people)} of {len(images)} images: {found_iphone} of "
        f"{n_iphone} iPhone photos, {len(people) - found_iphone} of "
        f"{len(images) - n_iphone} 360 images")
    for rel, share in sorted(people.items()):
        log(f"    {rel}: {share:.2f}% masked")
    for folder, kept in sorted(edges_kept.items()):
        log(f"  lens edge masked in {len(kept)} {folder} images, "
            f"{kept[0]:.1f}% of each kept")
    if iphone:
        total = sum(lidar_removed.values())
        log(f"  LiDAR removed from {len(iphone)} iPhone photos ({total} points): "
            + ", ".join(iphone))
    if people or edges_kept:
        log(f"  originals backed up in {session / BACKUP_DIR} "
            "(--undo puts them back)")
    return {"images": len(images), "people": people, "lidar_removed": lidar_removed,
            "lens_edge": {f: len(k) for f, k in edges_kept.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mask people out of a session's COLMAP model with SAM 3 "
                    "(Spirula Studio), and drop the LiDAR of iPhone photos "
                    "that show people.")
    parser.add_argument("--session-dir", type=Path, required=True,
                        help="session folder with colmap/ (and raw/iphone/)")
    parser.add_argument("--spirula",
                        help="spirula executable (default: SPIRULA_EXE, PATH, or "
                             "the newest spirula.exe on the desktop or in Downloads)")
    parser.add_argument("--sam-model",
                        help="SAM 3 checkpoint (default: the one Spirula Studio "
                             "downloaded)")
    parser.add_argument("--prompt", default="person",
                        help='what to mask in the iPhone photos, ";" between several '
                             "(default: person)")
    parser.add_argument("--prompt-360", default=PROMPT_360,
                        help="what to mask in the 360 images (default: Spirula's "
                             f'360-camera preset, "{PROMPT_360}")')
    parser.add_argument("--neg-prompt-360", default=NEGATIVE_360,
                        help="what to keep in the 360 images even where --prompt-360 "
                             f'matches (default: "{NEGATIVE_360}")')
    parser.add_argument("--no-lens-edge", action="store_true",
                        help="don't mask the area outside the 360 lens circles")
    parser.add_argument("--jobs", type=int, default=3,
                        help="spirula processes at once with builds older than "
                             "v2026.11.12 (default: 3, about 2.5 GB of GPU memory each)")
    parser.add_argument("--undo", action="store_true",
                        help="put back the files the last run changed")
    args = parser.parse_args()

    session = args.session_dir.resolve()
    if not session.is_dir():
        sys.exit(f"error: no session folder {session}")
    if args.undo:
        try:
            restore(session, print, undo=True)
        except RuntimeError as e:
            sys.exit(f"error: {e}")
        return
    print(f"Masking people in {session}")
    try:
        mask_session(session, args.spirula, args.sam_model, args.prompt, args.jobs,
                     prompt_360=args.prompt_360,
                     negative_360=args.neg_prompt_360 or None,
                     lens_edge=not args.no_lens_edge)
    except (MaskingUnavailable, FileNotFoundError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
