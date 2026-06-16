from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> dict[str, object]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"available": False, "path": None, "returncode": None, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {
            "available": True,
            "path": exe,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"available": True, "path": exe, "error": repr(exc)}


def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_cuda_env_check.json"))
    args = parser.parse_args()

    try:
        import triton  # type: ignore

        triton_info = {"available": True, "version": getattr(triton, "__version__", None)}
    except Exception as exc:
        triton_info = {"available": False, "error": repr(exc)}

    report = {
        "torch": {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "triton": triton_info,
        "nvcc": run_cmd(["nvcc", "--version"]),
        "msvc_cl": run_cmd(["cl"]),
        "notes": [
            "PyTorch CUDA extensions on Windows usually need both NVCC and MSVC cl.exe.",
            "torch.compile/Inductor GPU fusion needs a working Triton install.",
            "A CUDA toolkit newer than torch.version.cuda may still compile extensions, but ABI/toolchain compatibility must be tested.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
