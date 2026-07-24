import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models import AutoregressiveDecoder, BatchDecoder, LatentDynamics, RadarSchedulerModel
from radar_dnn_mcts.models.boundary import BoundaryPredictor


def _inputs(obs):
    features = FeatureBuilder(max_targets=5)
    tokens = torch.from_numpy(features.tokens(obs)).unsqueeze(0)
    context = torch.from_numpy(features.context(obs, 0.0, 0, 0, -1)).unsqueeze(0)
    return tokens, context


def test_policy_q_shapes_and_masks(radar_obs):
    """The shared model scores every row and rejects inactive targets."""
    tokens, context = _inputs(radar_obs)
    model = RadarSchedulerModel(d_model=32, nhead=4, encoder_layers=1, action_layers=1).eval()
    output = model(tokens, context)
    assert output.policy_logits.shape == (1, 6)
    assert output.q_values.shape == (1, 6)
    assert output.policy_logits[0, 4] < -1e8


def test_recurrent_ar_batch_and_boundary_contracts(radar_obs):
    """Decoder variants preserve their public tensor contracts."""
    tokens, context = _inputs(radar_obs)
    core = RadarSchedulerModel(d_model=32, nhead=4, encoder_layers=1, action_layers=1).eval()
    state = core.encode(tokens, context)
    dynamics = LatentDynamics(max_rows=6, d_model=32).eval()
    next_state, reward = dynamics(state, torch.tensor([1]))
    assert next_state.target_states.shape == state.target_states.shape
    assert reward.shape == (1,)

    ar = AutoregressiveDecoder(core, max_rows=6).eval()
    ar_state, prefix = ar.initial(tokens, context)
    output = ar.step(ar_state, prefix, context, torch.zeros(1, 6, dtype=torch.bool))
    assert output.policy_logits.shape == (1, 6)
    assert prefix.shape == (1, 0)

    batch = BatchDecoder(max_steps=4, d_model=32, nhead=4, encoder_layers=1, decoder_layers=1).eval()
    sequence = batch(tokens, context)
    assert sequence.policy_logits.shape == (1, 4, 6)

    boundary = BoundaryPredictor(max_rows=6, max_suffix=4, d_model=32, nhead=4, layers=1).eval()
    predicted = boundary(state, torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool), torch.tensor([100.0]))
    assert torch.allclose(predicted.global_state, state.global_state)
    assert torch.allclose(predicted.target_states, state.target_states)
