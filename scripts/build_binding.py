from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RADARXS = ROOT / "pufferlib" / "ocean" / "radarxs"


def main() -> None:
    subprocess.check_call(
        [sys.executable, "setup_binding.py", "build_ext", "--inplace"],
        cwd=str(RADARXS),
    )


if __name__ == "__main__":
    main()
