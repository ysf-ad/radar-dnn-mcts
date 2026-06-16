from __future__ import annotations


def test_core_imports():
    from pufferlib.ocean.radarxs.models.edf import EDFPlanner
    from pufferlib.ocean.radarxs.models.est import ESTPlanner
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    assert EDFPlanner is not None
    assert ESTPlanner is not None
    assert ActionAttentionFactorizedNet is not None
