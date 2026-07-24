import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models import AutoregressiveDecoder, BatchDecoder, LatentDynamics, RadarSchedulerModel
from radar_dnn_mcts.schedulers import (
    AutoregressiveScheduler,
    BatchScheduler,
    FullReencodeScheduler,
    FullWindowPUCTScheduler,
    MuZeroScheduler,
)
from radar_dnn_mcts.search import PUCTConfig


def test_all_scheduler_modes_return_legal_nonempty_plans(radar_obs):
    torch.manual_seed(916)
    features = FeatureBuilder(max_targets=5)
    core = RadarSchedulerModel(d_model=32, nhead=4, encoder_layers=1, action_layers=1).eval()
    planners = [
        FullReencodeScheduler(core, features=features, max_steps=8),
        FullWindowPUCTScheduler(
            core,
            config=PUCTConfig(rollouts=2, dirichlet_fraction=0.0),
            features=features,
        ),
        MuZeroScheduler(core, LatentDynamics(max_rows=6, d_model=32), features=features, max_steps=8),
        AutoregressiveScheduler(AutoregressiveDecoder(core, max_rows=6), features=features, max_steps=8),
        BatchScheduler(BatchDecoder(max_steps=8, d_model=32, nhead=4, encoder_layers=1, decoder_layers=1), features=features),
    ]
    for planner in planners:
        plan = planner.plan(radar_obs, 200.0)
        assert plan
        assert all(0 <= row <= 3 for row in plan)
        tracks = [row for row in plan if row > 0]
        assert len(tracks) == len(set(tracks))
