from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "radar_dnn_mcts"


def main() -> None:
    cmd = [
        sys.executable,
        "action_attention_batch_experiment.py",
        "--initials",
        "20",
        "--rates",
        "2",
        "--eval-seeds",
        "916",
        "--windows",
        "2",
        "--eval-windows",
        "4",
        "--train-steps",
        "2",
        "--batch-size",
        "8",
        "--d-model",
        "16",
        "--nhead",
        "2",
        "--nlayers",
        "1",
        "--out",
        str(ROOT / "results" / "smoke_action_attention.csv"),
    ]
    subprocess.check_call(cmd, cwd=str(CODE))


if __name__ == "__main__":
    main()
