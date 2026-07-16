from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def radar_obs():
    return {
        "grid": np.linspace(-500.0, 3000.0, 300, dtype=np.float32),
        "t_desired": np.asarray([-20.0, 150.0, 400.0, -1.0, -1.0], dtype=np.float32),
        "t_deadline": np.asarray([500.0, 900.0, 1200.0, -1.0, -1.0], dtype=np.float32),
        "t_dwell": np.asarray([30.0, 40.0, 50.0, 10.0, 10.0], dtype=np.float32),
        "priority": np.zeros(5, dtype=np.float32),
        "active_mask": np.asarray([True, True, True, False, False]),
        "tracked_mask": np.asarray([True, True, True, False, False]),
        "az_bin": np.linspace(0.0, 1.0, 5, dtype=np.float32),
        "el_bin": np.linspace(0.0, 1.0, 5, dtype=np.float32),
        "target_range": np.asarray([20e6, 30e6, 40e6, 0.0, 0.0], dtype=np.float32),
        "search_debt_ms": 40.0,
        "arrival_rate": 3.0,
        "sensor_id": 0,
    }
