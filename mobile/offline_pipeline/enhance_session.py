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

import numpy as np
from PIL import Image


def run_prompt_da(session_dir: Path) -> None:
    """Stage 1: Depth completion with PromptDA.

    Input:  raw/iphone/scan_NNN/depth.bin (Float32, 256x192, metres)
            raw/iphone/scan_NNN/rgb.jpg   (12MP)
    Output: enhanced/depth_dense/scan_NNN.png (uint16, mm, at RGB resolution)
    """
    # Imported here, so the masking stage runs without OpenCV and PyTorch.
    import cv2
    import torch
    from promptda.promptda import PromptDA  # installed from models/PromptDA

    raw_dir    = session_dir / "raw" / "iphone"
    output_dir = session_dir / "enhanced" / "depth_dense"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  PromptDA device: {device}")
    model = PromptDA.from_pretrained("depth-anything/promptda_vitl").to(device).eval()

    scan_dirs = sorted(raw_dir.glob("scan_*"))
    for scan_dir in scan_dirs:
        rgb_path   = scan_dir / "rgb.jpg"
        depth_path = scan_dir / "depth.bin"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        scan_name   = scan_dir.name
        output_path = output_dir / f"{scan_name}.png"
        if output_path.exists():
            print(f"  PromptDA: {scan_name} already done, skipping")
            continue

        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_w, rgb_h = rgb.size

        # Load sparse LiDAR depth (Float32, 256x192, metres)
        raw = np.frombuffer(depth_path.read_bytes(), dtype=np.float32)
        sparse = raw[: 256 * 192].reshape(192, 256)
        # Resize sparse depth to RGB resolution for PromptDA prompt input
        sparse_resized = cv2.resize(sparse, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        with torch.no_grad():
            depth_pred = model.predict(
                image=rgb,
                prompt_depth=torch.from_numpy(sparse_resized).unsqueeze(0).unsqueeze(0).to(device),
            )  # returns (1, 1, H, W) tensor in metres

        depth_m = depth_pred.squeeze().cpu().numpy()  # (H, W) float32, metres
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_mm)
        print(f"  PromptDA: {scan_name} -> {output_path.name}")


def run_stable_normal(session_dir: Path) -> None:
    """Stage 2: Normal estimation with StableNormal.

    Input:  raw/iphone/scan_NNN/rgb.jpg
    Output: enhanced/normals/scan_NNN.png (RGB-encoded world-space normals)
    """
    import cv2
    import torch
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    from stablenormal.pipeline_stablenormal import StableNormalPipeline

    raw_dir    = session_dir / "raw" / "iphone"
    output_dir = session_dir / "enhanced" / "normals"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  StableNormal device: {device}")

    # Two-step pipeline: coarse YOSO pass → refined diffusion pass
    yoso = YOSONormalsPipeline.from_pretrained(
        "Stable-X/yoso-normal-v0-3", trust_remote_code=True
    ).to(device)
    pipe = StableNormalPipeline.from_pretrained(
        "Stable-X/stable-normal-v0-1", trust_remote_code=True,
        yoso_version=yoso,
    ).to(device)

    scan_dirs = sorted(raw_dir.glob("scan_*"))
    for scan_dir in scan_dirs:
        rgb_path = scan_dir / "rgb.jpg"
        if not rgb_path.exists():
            continue

        scan_name   = scan_dir.name
        output_path = output_dir / f"{scan_name}.png"
        if output_path.exists():
            print(f"  StableNormal: {scan_name} already done, skipping")
            continue

        rgb = Image.open(rgb_path).convert("RGB")
        with torch.no_grad():
            result = pipe(rgb, match_input_resolution=True)

        # result.prediction is (H, W, 3) float32 in [-1, 1]
        normal = result.prediction
        # Encode to uint8 RGB: map [-1,1] → [0,255]
        normal_rgb = ((normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
        print(f"  StableNormal: {scan_name} -> {output_path.name}")


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
        skipped.append(name)
        print(f"  {name} skipped: the Python package '{e.name}' isn't "
              "installed (setup.sh installs it)")
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
