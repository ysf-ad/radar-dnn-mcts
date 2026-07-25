from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models import LatentDynamics, RadarSchedulerModel
from reference_implementations.muzero_general_stepwise import (
    MuZeroGeneralStepwiseScheduler,
)
from reference_implementations.muzero_general_windowed import (
    MuZeroGeneralWindowedScheduler,
)


def test_reference_searches_are_runnable(radar_obs):
    features = FeatureBuilder(max_targets=5)
    model = RadarSchedulerModel(
        d_model=32,
        nhead=4,
        encoder_layers=1,
        action_layers=1,
    )
    dynamics = LatentDynamics(max_rows=6, d_model=32)
    planners = [
        MuZeroGeneralStepwiseScheduler(
            model, dynamics, features, simulations=1, max_steps=4
        ),
        MuZeroGeneralWindowedScheduler(
            model, dynamics, features, simulations=1, max_steps=4
        ),
    ]
    for planner in planners:
        plan = planner.plan(radar_obs)
        assert plan
        assert all(0 <= action <= 3 for action in plan)
        assert planner.last_g_calls > 0
        assert planner.last_observations == 1
