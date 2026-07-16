import numpy as np

from radar_dnn_mcts.env.actions import valid_action_mask
from radar_dnn_mcts.env.features import CONTEXT_DIM, TOKEN_DIM, FeatureBuilder


def test_feature_shapes_and_selected_mask(radar_obs):
    builder = FeatureBuilder(max_targets=5)
    tokens = builder.tokens(radar_obs, selected={2}, search_count=3)
    context = builder.context(radar_obs, elapsed_ms=70.0, search_count=3, track_count=2, last_row=2)
    assert tokens.shape == (6, TOKEN_DIM)
    assert context.shape == (CONTEXT_DIM,)
    assert tokens[2, 8] == 1.0
    assert np.isfinite(tokens).all()


def test_invalid_and_dropped_targets_are_masked(radar_obs):
    mask = valid_action_mask(radar_obs["active_mask"], radar_obs["t_deadline"], np.array([False, True, False, False, False]))
    assert mask.tolist() == [True, True, False, True, False, False]
