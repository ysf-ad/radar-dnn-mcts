import numpy as np

from scripts.self_play import GROUPED_KEYS, merge_trajectories


def grouped_arrays(windows: int, steps: int) -> dict[str, np.ndarray]:
    """Create a minimal grouped trajectory fixture."""
    return {
        "tokens": np.zeros((windows, steps, 3, 2), dtype=np.float32),
        "context": np.zeros((windows, steps, 2), dtype=np.float32),
        "policy": np.full((windows, steps, 3), 1 / 3, dtype=np.float32),
        "actions": np.zeros((windows, steps), dtype=np.int64),
        "rewards": np.zeros((windows, steps), dtype=np.float32),
        "returns": np.zeros((windows, steps), dtype=np.float32),
        "durations_ms": np.ones((windows, steps), dtype=np.float32),
        "action_mask": np.ones((windows, steps), dtype=bool),
        "return_scale": np.asarray(32.0),
    }


def test_merge_trajectories_pads_steps_and_preserves_masks(tmp_path):
    """Merging pads short windows without marking padding valid."""
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged.npz"
    np.savez_compressed(first, **grouped_arrays(2, 3))
    np.savez_compressed(second, **grouped_arrays(1, 5))

    merge_trajectories([first, second], output)

    with np.load(output) as merged:
        assert set(GROUPED_KEYS) <= set(merged.files)
        assert merged["tokens"].shape == (3, 5, 3, 2)
        assert merged["action_mask"].shape == (3, 5)
        assert not merged["action_mask"][:2, 3:].any()
        assert float(merged["return_scale"]) == 32.0
