from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=916)
    args = parser.parse_args()

    print("This is a thin entry point for the 9-cell evaluation.")
    print(f"checkpoint={args.checkpoint}")
    print("Use radar_dnn_mcts/two_sensor_physical_head_eval.py for the full evaluator.")
    print("Recommended grid: initials=20,40,60 and rates=2,3,4.")


if __name__ == "__main__":
    main()
