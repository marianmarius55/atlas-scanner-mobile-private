"""
Offline enhancement pipeline for Atlas Mobile capture sessions.

Needs the app's COLMAP export in the session (export with "COLMAP Only" on
the phone first). Then:
1. Dense depth maps via PromptDA (LiDAR-anchored)
2. Surface normal maps via StableNormal
3. The COLMAP model: the app's export, kept as it is
4. People masked out of the COLMAP images with SAM 3 (Spirula Studio), the
   area outside each 360 lens with Spirula's border mask, and the LiDAR of
   iPhone photos with people removed (mask_people.py)

Stages 1 and 2 are optional: when one fails, the others still run, and
without PyTorch they are skipped.

Usage:
    uv run python enhance_session.py --session-dir /path/to/session_YYYY-MM-DD_HH-MM-SS
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import traceback
from pathlib import Path

import os

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"
# Model weights download to models/hf_cache (Hugging Face) and
# models/torch_cache (torch.hub: StableNormal's DINOv2) unless HF_HOME and
# TORCH_HOME say otherwise.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ARKit's LiDAR depth map (raw/iphone/scan_NNN/depth.bin): Float32 metres,
# 256 x 192, in the landscape sensor's orientation.
LIDAR_W, LIDAR_H = 256, 192


def _use_repo(name: str) -> None:
    """Puts models/<name> (a checkout of the model's repository) first on
    the import path. The repos pin old torch and numpy versions, so they are
    imported from the checkout rather than pip-installed."""
    repo = MODELS_DIR / name
    if not repo.is_dir():
        raise ModuleNotFoundError(f"models/{name} is missing", name=name)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _scans(session_dir: Path) -> list[Path]:
    """The raw/iphone/scan_NNN folders to process: the iPhone photos the
    phone's COLMAP export kept (colmap/images/face_iphone), else all."""
    raw_dir = session_dir / "raw" / "iphone"
    exported = session_dir / "colmap" / "images" / "face_iphone"
    if exported.is_dir():
        names = sorted(p.stem for p in exported.glob("*.jpg"))
        return [raw_dir / n for n in names if (raw_dir / n).is_dir()]
    return sorted(raw_dir.glob("scan_*"))


def run_prompt_da(session_dir: Path) -> None:
    """Stage 1: Depth completion with PromptDA (Prompt Depth Anything).

    Input:  raw/iphone/scan_NNN/depth.bin (the LiDAR prompt)
            raw/iphone/scan_NNN/rgb.jpg   (portrait: the landscape sensor
                                           image turned 90 degrees clockwise)
    Output: enhanced/depth_dense/scan_NNN.png (uint16, mm, at RGB resolution)

    The image goes in as the repo's load_image prepares it: RGB 0..1, the
    longer side at most 1008 px and both sides a multiple of 14. The prompt
    goes in at its own 256 x 192 (the model resizes it), turned like the
    photo.
    """
    # Imported here, so the masking stage runs without OpenCV and PyTorch.
    import cv2
    import torch
    _use_repo("PromptDA")
    from promptda.promptda import PromptDA

    output_dir = session_dir / "enhanced" / "depth_dense"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Photos without a depth map yet; with none the model isn't loaded.
    todo = [d for d in _scans(session_dir)
            if (d / "rgb.jpg").exists() and (d / "depth.bin").exists()
            and not (output_dir / f"{d.name}.png").exists()]
    if not todo:
        print(f"  PromptDA: every depth map is already done ({output_dir})")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  PromptDA device: {device}, {len(todo)} photos to do")
    model = PromptDA.from_pretrained("depth-anything/promptda_vitl").to(device).eval()

    max_size, multiple = 1008, 14
    done = 0
    for scan_dir in todo:
        rgb_path   = scan_dir / "rgb.jpg"
        depth_path = scan_dir / "depth.bin"
        scan_name   = scan_dir.name
        output_path = output_dir / f"{scan_name}.png"

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        rgb_h, rgb_w = rgb.shape[:2]
        scale = min(1.0, max_size / max(rgb_h, rgb_w))
        tar_h = int(rgb_h * scale // multiple * multiple)
        tar_w = int(rgb_w * scale // multiple * multiple)
        image = cv2.resize(rgb, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

        raw = np.frombuffer(depth_path.read_bytes(), dtype=np.float32)
        if raw.size < LIDAR_W * LIDAR_H:
            print(f"  PromptDA: {scan_name} has no full depth map, skipping")
            continue
        prompt = raw[: LIDAR_W * LIDAR_H].reshape(LIDAR_H, LIDAR_W)
        if rgb_h > rgb_w:
            prompt = np.rot90(prompt, k=-1)   # clockwise, like the photo
        prompt = np.nan_to_num(np.ascontiguousarray(prompt), nan=0.0)

        with torch.no_grad():
            depth = model.predict(
                torch.from_numpy(image).permute(2, 0, 1)[None].to(device),
                torch.from_numpy(prompt.copy())[None, None].to(device),
            )  # (1, 1, tar_h, tar_w), metres

        depth_m = depth.squeeze().float().cpu().numpy()
        depth_m = cv2.resize(depth_m, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_mm)
        done += 1
        if done % 10 == 0:
            print(f"  PromptDA: {done} depth maps")
    print(f"  PromptDA: {done} depth maps written to {output_dir}")


def _diffusers_compat() -> None:
    """The StableNormal models' own code (controlnetvae.py in both Hugging
    Face repos, loaded with trust_remote_code) imports ControlNetOutput from
    diffusers.models.controlnet, which newer diffusers moved to
    diffusers.models.controlnets.controlnet. The old name is pointed at the
    new module."""
    import importlib
    old, new = "diffusers.models.controlnet", "diffusers.models.controlnets.controlnet"
    try:
        importlib.import_module(old)
    except ModuleNotFoundError:
        sys.modules[old] = importlib.import_module(new)


def run_stable_normal(session_dir: Path) -> None:
    """Stage 2: Normal estimation with StableNormal.

    Input:  raw/iphone/scan_NNN/rgb.jpg
    Output: enhanced/normals/scan_NNN.png (camera-space normals as RGB,
            [-1, 1] mapped to 0..255, at RGB resolution)

    The model is the repo's own entry point (hubconf.StableNormal: the YOSO
    pass and the diffusion pass, fp16), run on the GPU at its 1024 px.
    """
    import torch
    _use_repo("StableNormal")
    import hubconf  # models/StableNormal/hubconf.py

    output_dir = session_dir / "enhanced" / "normals"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Photos without a normal map yet; with none nothing is downloaded or
    # loaded.
    todo = [d for d in _scans(session_dir)
            if (d / "rgb.jpg").exists()
            and not (output_dir / f"{d.name}.png").exists()]
    if not todo:
        print(f"  StableNormal: every normal map is already done ({output_dir})")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("StableNormal runs in fp16 and needs a CUDA GPU")
    print(f"  StableNormal device: cuda, {len(todo)} photos to do")
    print("  StableNormal: getting the model weights (7.4 GB the first time,"
          " plus 1.2 GB for DINOv2; a stopped download resumes)")
    # hubconf joins "Stable-X" and the model name with os.path.join, which
    # on Windows makes "Stable-X\\yoso-normal-v0-3", not a Hugging Face id.
    # So both models are downloaded into one folder and loaded from there.
    from huggingface_hub import snapshot_download
    local = MODELS_DIR / "stable_x"
    for repo in ("yoso-normal-v0-3", "stable-normal-v0-1"):
        # The fp16 weights only, which is what hubconf loads (variant
        # "fp16"): 7.4 GB instead of 15.5.
        snapshot_download(repo_id=f"Stable-X/{repo}", local_dir=local / repo,
                          ignore_patterns=["*.bin"])
    _diffusers_compat()
    predictor = hubconf.StableNormal(local_cache_dir=str(local), device="cuda:0")

    done = 0
    for scan_dir in todo:
        rgb_path = scan_dir / "rgb.jpg"
        output_path = output_dir / f"{scan_dir.name}.png"
        rgb = Image.open(rgb_path).convert("RGB")
        normal = predictor(rgb, resolution=1024, data_type="indoor")
        normal.save(output_path)
        done += 1
        if done % 10 == 0:
            print(f"  StableNormal: {done} normal maps")
    print(f"  StableNormal: {done} normal maps written to {output_dir}")


def has_app_export(session_dir: Path) -> bool:
    """True when the app's COLMAP export is in the session (colmap/sparse/0
    and its images). A model assemble_colmap.py built (it writes
    points3D_sources.json) doesn't count: for this app's sessions it has no
    360 photos, the iPhone cameras face backwards and their size is
    portrait/landscape swapped (checked on session 2026-11-14_20-55-31)."""
    colmap = session_dir / "colmap"
    sparse = colmap / "sparse" / "0"
    return (
        (sparse / "images.bin").is_file()
        and not (sparse / "points3D_sources.json").is_file()
        and any(p.is_file() for p in (colmap / "images").rglob("*.jpg"))
    )


def run_stage(name: str, fn, session_dir: Path,
              failed: list[str], skipped: list[str]) -> None:
    """Runs an enhancement stage; a failure is logged and the rest still runs."""
    try:
        fn(session_dir)
    except ModuleNotFoundError as e:
        missing = e.name or str(e)
        if "." in missing:
            # A module the installed package doesn't have: another version
            # than the code was written for.
            failed.append(name)
            print(f"  {name} failed: the installed {missing.split('.')[0]} has "
                  f"no module '{missing}' (the code expects another version)")
            traceback.print_exc()
        else:
            skipped.append(name)
            print(f"  {name} skipped: the Python package '{missing}' isn't installed")
    except Exception as e:  # noqa: BLE001 -- any failure of an optional stage
        failed.append(name)
        print(f"  {name} failed: {type(e).__name__}: {e}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Mobile offline enhancement")
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to captured session directory",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip PromptDA depth completion",
    )
    parser.add_argument(
        "--skip-normals",
        action="store_true",
        help="Skip StableNormal estimation",
    )
    parser.add_argument(
        "--skip-people-mask",
        action="store_true",
        help="Skip masking people with SAM 3 (Spirula Studio)",
    )
    parser.add_argument(
        "--spirula",
        help="spirula executable (default: SPIRULA_EXE, PATH, or the newest "
             "spirula.exe on the desktop or in Downloads)",
    )
    parser.add_argument(
        "--sam-model",
        help="SAM 3 checkpoint (default: the one Spirula Studio downloaded)",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    print(f"Enhancing session: {session_dir}")

    # The COLMAP model comes from the phone: its exporter is the only one
    # that handles the 360 photos (dual fisheye, rig, shot timing).
    if not has_app_export(session_dir):
        print("This upload has no COLMAP export from the phone "
              "(colmap/sparse/0 and colmap/images are empty).")
        print("The server can't build the model: only the app's exporter "
              "handles the 360 photos.")
        print("On the phone, export with \"COLMAP Only\" first, then upload "
              "with \"Host Processing\".")
        sys.exit(2)

    failed: list[str] = []
    skipped: list[str] = []

    if not args.skip_depth:
        print("Stage 1: PromptDA depth completion")
        run_stage("PromptDA", run_prompt_da, session_dir, failed, skipped)

    if not args.skip_normals:
        print("Stage 2: StableNormal estimation")
        run_stage("StableNormal", run_stable_normal, session_dir, failed, skipped)

    print("Stage 3: COLMAP model: using the app's export (colmap/sparse/0)")

    if not args.skip_people_mask:
        print("Stage 4: masking people and the 360 lens edges (Spirula Studio)")
        from mask_people import MaskingUnavailable, mask_session
        try:
            mask_session(session_dir, spirula=args.spirula, sam_model=args.sam_model)
        except MaskingUnavailable as e:
            print(f"  skipped: {e}")

    notes = []
    if skipped:
        notes.append("skipped " + ", ".join(skipped))
    if failed:
        notes.append("failed " + ", ".join(failed))
    print("Done" + (f" ({'; '.join(notes)})." if notes else "."))


if __name__ == "__main__":
    main()
