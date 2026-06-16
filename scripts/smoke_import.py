from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def main() -> None:
    import pufferlib.ocean.radarxs.engine as engine
    from pufferlib.ocean.radarxs.models.edf import EDFPlanner
    from pufferlib.ocean.radarxs.models.est import ESTPlanner
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    _ = engine.GRID_SIZE
    _ = EDFPlanner
    _ = ESTPlanner
    model = ActionAttentionFactorizedNet(d_model=16, nhead=2, nlayers=1)
    print("smoke_import ok")
    print(f"model={model.__class__.__name__}")


if __name__ == "__main__":
    main()
