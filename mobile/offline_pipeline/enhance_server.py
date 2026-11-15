"""
Atlas Mobile — Host Enhancement Server

Receives session zips from the iOS app, runs the enhancement pipeline
(enhance_session.py: PromptDA + StableNormal + COLMAP model + people masking
with SAM 3 via Spirula Studio), and serves the results back.

Usage:
    cd mobile/offline_pipeline
    python enhance_server.py [--host 0.0.0.0] [--port 8765] [--work-dir DIR]
                             [--spirula PATH] [--sam-model PATH]

On Windows run it with the Python that has the pipeline's packages, e.g.
    py -3.13 enhance_server.py --work-dir C:\\atlas_jobs
and allow it through Windows Firewall when asked (private networks).

The iOS app (HostUploader.swift) connects to this server when "Host
Processing" is selected. Set the host URL in the app to:
    http://<your-machine-ip>:8765

Find your machine IP (the server also prints its addresses):
    Windows: ipconfig        (IPv4 Address of the Wi-Fi/Ethernet adapter)
    macOS:   ipconfig getifaddr en0
    Linux:   hostname -I | awk '{print $1}'

Both devices must be on the same network.

People masking needs Spirula Studio and its SAM 3 checkpoint (see
mask_people.py). The pipeline finds both on its own; --spirula and
--sam-model choose others. Export COLMAP on the phone before uploading:
the app's export has the 360 photos, and the pipeline masks it as it is.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}  # job_id -> {status, log_lines, session_dir, result_zip}
_work_dir = Path(tempfile.gettempdir()) / "atlas_jobs"


def _job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def _zip_name(header: str | None) -> str:
    """The upload's file name from X-Session-Name, reduced to a plain name
    so a header can't write outside the job folder."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(header or "").name).strip("._")
    if not name:
        name = "session"
    return name if name.lower().endswith(".zip") else name + ".zip"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload")
def upload():
    """Receive a session zip, unpack it, start processing in background."""
    job_id = str(uuid.uuid4())[:8]

    job_dir = _work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / _zip_name(request.headers.get("X-Session-Name"))
    zip_path.write_bytes(request.data)

    # Unzip
    session_dir = job_dir / "session"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(session_dir)
        # Find the actual session directory inside the zip
        subdirs = [d for d in session_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            session_dir = subdirs[0]
    except Exception as e:
        return jsonify({"error": f"Unzip failed: {e}"}), 400

    _jobs[job_id] = {
        "status": "queued",
        "log_lines": [],
        "session_dir": str(session_dir),
        "result_zip": None,
    }

    # Run pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline, args=(job_id, session_dir), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id}), 200


@app.get("/status/<job_id>")
def status(job_id: str):
    """Return current job status and accumulated log lines."""
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "log_lines": job["log_lines"],
    })


@app.get("/download/<job_id>")
def download(job_id: str):
    """Return the enhanced results zip once processing is complete."""
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    if job["status"] != "complete":
        return jsonify({"error": f"Job not complete (status: {job['status']})"}), 409
    result_zip = job.get("result_zip")
    if not result_zip or not Path(result_zip).exists():
        return jsonify({"error": "Result zip not found"}), 500
    return send_file(result_zip, mimetype="application/zip",
                     as_attachment=True,
                     download_name=Path(result_zip).name)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, session_dir: Path) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def log(line: str) -> None:
        print(line)
        job["log_lines"].append(line)

    try:
        script = Path(__file__).parent / "enhance_session.py"
        # This server's own Python: "python3" is a Microsoft Store stub on
        # Windows, which exits with code 9009.
        cmd = [sys.executable, str(script), "--session-dir", str(session_dir)]
        # UTF-8 and unbuffered, so the pipeline's lines (some with ✓ and →)
        # neither fail on Windows' default code page nor arrive in chunks.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

        log(f"Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            log(line.rstrip())
        proc.wait()

        if proc.returncode != 0:
            job["status"] = "error"
            log(f"Pipeline exited with code {proc.returncode}")
            return

        # Zip the enhanced/ and colmap/ directories for download
        result_zip = session_dir.parent / f"{session_dir.name}_enhanced.zip"
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in ("enhanced", "colmap"):
                src = session_dir / folder
                if src.exists():
                    for f in sorted(src.rglob("*")):
                        if f.is_file():
                            zf.write(f, f.relative_to(session_dir))

        job["result_zip"] = str(result_zip)
        job["status"] = "complete"
        log(f"✓ Complete — {result_zip.stat().st_size // 1024} KB ready for download")

    except Exception as e:
        job["status"] = "error"
        job["log_lines"].append(f"Error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _local_ipv4_addresses() -> list[str]:
    """This machine's IPv4 addresses, for the URL to set in the app."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    found = []
    for info in infos:
        ip = info[4][0]
        if not ip.startswith("127.") and ip not in found:
            found.append(ip)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Mobile host enhancement server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 — all interfaces)")
    parser.add_argument("--port", type=int, default=8765)
    default_work_dir = Path(tempfile.gettempdir()) / "atlas_jobs"
    parser.add_argument("--work-dir", type=Path, default=default_work_dir,
                        help=f"Directory for job working files (default: {default_work_dir})")
    parser.add_argument("--spirula",
                        help="spirula executable for people masking (sets SPIRULA_EXE)")
    parser.add_argument("--sam-model",
                        help="SAM 3 checkpoint for people masking (sets SPIRULA_SAM_MODEL)")
    args = parser.parse_args()

    global _work_dir
    _work_dir = args.work_dir
    _work_dir.mkdir(parents=True, exist_ok=True)
    # The pipeline runs as a child process and reads these.
    if args.spirula:
        os.environ["SPIRULA_EXE"] = str(Path(args.spirula).resolve())
    if args.sam_model:
        os.environ["SPIRULA_SAM_MODEL"] = str(Path(args.sam_model).resolve())

    addresses = _local_ipv4_addresses() or ["<this-machine-ip>"]
    print("Atlas Mobile Enhancement Server")
    print(f"  Listening on http://{args.host}:{args.port}")
    for ip in addresses:
        print(f"  Set in the app: http://{ip}:{args.port}")
    print(f"  Work dir: {_work_dir}")
    print(f"  Health check: http://{addresses[0]}:{args.port}/health")
    print()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
