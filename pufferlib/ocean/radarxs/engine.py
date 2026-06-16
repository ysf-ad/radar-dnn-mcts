"""
Radar Scheduling Engine.
Connects planners (EST, MCTS, Transformer) to the C binding environment.

Observation Features (per tracker):
  - t_desired: Time until next update is needed.
  - t_deadline: Time until track is lost (deadline).
  - t_dwell: Estimated execution time for this action.
  - priority: Urgency score. -1 (NO_TARGET) indicates an inactive/empty slot.
  - az_bin: Target azimuth bin index normalized to [0, 1].
  - el_bin: Target elevation bin index normalized to [0, 1].
"""
import numpy as np
import warnings
from . import binding


# Environment constants (from radarxs.h)
GRID_SIZE = 300  # 30 az * 10 el slices (MAX_AZ_SLICES * MAX_EL_SLICES)
MAX_TRACKERS = 500
FEATURES_PER_TRACKER = 6  # t_desired, t_deadline, t_dwell, priority, az_bin, el_bin
NO_TARGET = -1
 

def get_obs_from_buf(obs_buf, max_trackers=MAX_TRACKERS):
    """
    Convert flat observation buffer to planner-compatible format.
    
    Args:
        obs_buf: Flat observation buffer from binding, shape (1, obs_size).
                 Layout: [Grid(300)] + [Tracker0(4), Tracker1(4), ...] + [sensor_id]
        max_trackers: Maximum number of trackers.
    
    Returns:
        dict: Observation with keys:
            - 'grid': (300,) array of sector freshness values
            - 't_desired': (max_trackers,) time until desired update
            - 't_deadline': (max_trackers,) time until deadline
            - 't_dwell': (max_trackers,) dwell time estimate
            - 'priority': (max_trackers,) priority (-1 = inactive)
            - 'active_mask': (max_trackers,) boolean mask of active targets
            - 'sensor_id': Current sensor (0=S-band, 1=X-band)
    """
    if obs_buf.ndim == 2:
        obs_flat = obs_buf[0]
    else:
        obs_flat = obs_buf
    
    # Extract grid (sector freshness)
    grid = obs_flat[:GRID_SIZE]
    
    # Infer per-tracker feature count from buffer shape to tolerate stale bindings.
    # Expected: 6, legacy fallback: 4.
    inferred = int((len(obs_flat) - GRID_SIZE - 1) / max_trackers)
    features_per_tracker = inferred if inferred in (4, 6) else FEATURES_PER_TRACKER

    # Extract tracker data.
    base_idx = GRID_SIZE
    end_idx = base_idx + max_trackers * features_per_tracker
    flat_trackers = obs_flat[base_idx:end_idx].reshape(max_trackers, features_per_tracker)
    
    t_desired = flat_trackers[:, 0]
    t_deadline = flat_trackers[:, 1]
    t_dwell = flat_trackers[:, 2]
    priority = flat_trackers[:, 3]
    az_bin = flat_trackers[:, 4] if features_per_tracker >= 6 else np.zeros(max_trackers, dtype=np.float32)
    el_bin = flat_trackers[:, 5] if features_per_tracker >= 6 else np.zeros(max_trackers, dtype=np.float32)

    # Sanitize uninitialized values coming from C side.
    # NaN/Inf should be treated as inactive slots.
    t_desired = np.where(np.isfinite(t_desired), t_desired, NO_TARGET)
    t_deadline = np.where(np.isfinite(t_deadline), t_deadline, NO_TARGET)
    t_dwell = np.where(np.isfinite(t_dwell), t_dwell, 10.0)
    # Dwell must be positive for valid window-budget accounting.
    t_dwell = np.clip(t_dwell, 1.0, 2000.0)
    priority = np.where(np.isfinite(priority), priority, 0.0)
    az_bin = np.where(np.isfinite(az_bin), az_bin, 0.0)
    el_bin = np.where(np.isfinite(el_bin), el_bin, 0.0)
    az_bin = np.clip(az_bin, 0.0, 1.0)
    el_bin = np.clip(el_bin, 0.0, 1.0)
    
    # Active mask: target is active if t_desired != NO_TARGET (-1)
    active_mask = np.isfinite(t_desired) & (t_desired != NO_TARGET)
    
    # Sensor ID (last element) - handle NaN gracefully
    sensor_id = 0
    if end_idx < len(obs_flat):
        val = obs_flat[end_idx]
        if not np.isnan(val) and np.isfinite(val):
            sensor_id = int(val)
    
    return {
        'grid': grid,
        't_desired': t_desired,
        't_deadline': t_deadline,
        't_dwell': t_dwell,
        'priority': priority,
        'az_bin': az_bin,
        'el_bin': el_bin,
        'active_mask': active_mask,
        'sensor_id': sensor_id,
    }


class RadarEngine:
    """
    Variable-length Window Execution Engine for Radar Task Scheduling.
    
    Connects any planner (EST, MCTS, Transformer) to the C environment binding.
    Each call to step_window() generates an action plan (length determined by planner) and executes it.
    """
    
    def __init__(
        self,
        planner,
        initial_targets=50,
        max_trackers=MAX_TRACKERS,
        seed=1,
        window_ms=200,
        enable_global_delay=False,
        enable_local_delay=True,
        enable_x_band=False,
        enable_search_refresh_tracked=True,
        search_refresh_gain=1.0,
        enable_priority=True,
        enable_poisson_arrivals=False,
        activate_all_targets_without_poisson=True,
        poisson_rate_per_second=5.0,
        search_action_reward=0.1,
        track_update_reward=0.1,
        track_loss_penalty=1.0,
        track_urgency_bonus_weight=0.0,
        target_service_weight=0.0,
        target_service_horizon_ms=1000.0,
        sector_staleness_weight=0.0,
        searched_sector_reward_weight=0.0,
        search_frame_overdue_weight=0.0,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=0.0,
        search_task_cost_mode=0,
        revisit_time_scale=1.0,
        dwell_time_scale=1.0,
        penalize_hidden_targets=False,
        enable_track_beam_scan=False,
        episode_time_limit_ms=60000,
        search_delay_mode=0,
        search_debt_penalty_weight=0.0001,
        search_debt_tau_ms=10.0,
        search_delay_penalty_cap=-1.0,
    ):
        """
        Args:
            planner: Planner instance with a plan() method (EST, MCTS, or Transformer).
            initial_targets: Number of targets to initialize.
            max_trackers: Maximum tracker capacity.
            seed: Random seed.
            window_ms: Execution window duration in milliseconds (default 200).
        """
        self.window_ms = window_ms
        self.planner = planner
        self.initial_targets = initial_targets
        self.max_trackers = max_trackers
        self.seed = seed
        
        # Environment buffers
        obs_size = GRID_SIZE + max_trackers * FEATURES_PER_TRACKER + 1
        self.num_envs = 1
        self.obs_buf = np.zeros((self.num_envs, obs_size), dtype=np.float32)
        self.act_buf = np.zeros((self.num_envs,), dtype=np.int32)
        self.rew_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self.term_buf = np.zeros((self.num_envs,), dtype=np.uint8)
        self.trunc_buf = np.zeros((self.num_envs,), dtype=np.uint8)
        
        # Initialize environment
        self.env = binding.vec_init(
            self.obs_buf, self.act_buf, self.rew_buf, 
            self.term_buf, self.trunc_buf, self.num_envs, seed,
            initial_targets=initial_targets,
            max_trackers=max_trackers,
            enable_global_delay=int(enable_global_delay),
            enable_local_delay=int(enable_local_delay),
            enable_x_band=int(enable_x_band),
            enable_search_refresh_tracked=int(enable_search_refresh_tracked),
            search_refresh_gain=float(search_refresh_gain),
            enable_priority=int(enable_priority),
            enable_poisson_arrivals=int(enable_poisson_arrivals),
            activate_all_targets_without_poisson=int(activate_all_targets_without_poisson),
            poisson_rate_per_second=float(poisson_rate_per_second),
            search_action_reward=float(search_action_reward),
            track_update_reward=float(track_update_reward),
            track_loss_penalty=float(track_loss_penalty),
            track_urgency_bonus_weight=float(track_urgency_bonus_weight),
            target_service_weight=float(target_service_weight),
            target_service_horizon_ms=float(target_service_horizon_ms),
            sector_staleness_weight=float(sector_staleness_weight),
            searched_sector_reward_weight=float(searched_sector_reward_weight),
            search_frame_overdue_weight=float(search_frame_overdue_weight),
            search_frame_desired_ms=float(search_frame_desired_ms),
            search_frame_deadline_ms=float(search_frame_deadline_ms),
            search_frame_drop_penalty=float(search_frame_drop_penalty),
            search_task_cost_mode=int(search_task_cost_mode),
            revisit_time_scale=float(revisit_time_scale),
            dwell_time_scale=float(dwell_time_scale),
            penalize_hidden_targets=int(penalize_hidden_targets),
            enable_track_beam_scan=int(enable_track_beam_scan),
            episode_time_limit_ms=int(episode_time_limit_ms),
            search_delay_mode=int(search_delay_mode),
            search_debt_penalty_weight=float(search_debt_penalty_weight),
            search_debt_tau_ms=float(search_debt_tau_ms),
            search_delay_penalty_cap=float(search_delay_penalty_cap),
        )
        
        # Statistics
        self.total_reward = 0.0
        self.total_steps = 0
        self.windows_completed = 0
        self._warned_obs_layout = False

        # Ensure the observation buffer is populated before first use.
        self.reset(seed)
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is None:
            seed = self.seed
        binding.vec_reset(self.env, seed)
        self.total_reward = 0.0
        self.total_steps = 0
        self.windows_completed = 0

        # One-time sanity warning for common binding/layout mismatch issues.
        if not self._warned_obs_layout:
            nan_ratio = float(np.mean(~np.isfinite(self.obs_buf[0])))
            if nan_ratio > 0.01:
                warnings.warn(
                    f"RadarEngine observation buffer has {nan_ratio*100:.1f}% non-finite values after reset. "
                    "This usually indicates binding/source mismatch (e.g., stale compiled .pyd vs current header). "
                    "Metrics may be invalid until binding is rebuilt.",
                    RuntimeWarning,
                )
            self._warned_obs_layout = True
    
    def step_window(self):
        """
        Execute a single planning window.
        
        1. Convert observation to MCTS format.
        2. Call planner.plan() to get action sequence.
        3. Execute each action via binding.vec_step().
        
        Returns:
            float: Total reward accumulated in this window.
        """
        # Get observation in planner-compatible format
        planner_obs = get_obs_from_buf(self.obs_buf, self.max_trackers)
        
        # Generate plan (passing window budget to hint candidate count)
        plan = self.planner.plan(planner_obs, budget_ms=self.window_ms)
        
        # Execute plan
        window_reward = 0.0
        cumulative_time = 0.0
        t_dwell = planner_obs['t_dwell']
        
        for action in plan:
            # Estimate Dwell Time (for budget)
            if action == 0: # SEARCH
                est_dt = 10.0
            else: # TRACK
                est_dt = t_dwell[action-1]

            # Execute action
            self.act_buf[0] = int(action)
            binding.vec_step(self.env)
            window_reward += self.rew_buf[0]
            self.total_steps += 1
            
            cumulative_time += est_dt
            
            # Check for episode termination
            if self.term_buf[0]:
                break
            
            # Check for Window Time Budget
            if cumulative_time >= self.window_ms:
                break
        
        self.total_reward += window_reward
        self.windows_completed += 1
        
        return window_reward
    
    def run_episode(self, num_windows=50):
        """
        Run a full episode of specified windows.
        
        Args:
            num_windows: Number of planning windows to execute.
        
        Returns:
            dict: Episode statistics.
        """
        self.reset()
        
        for w in range(num_windows):
            window_reward = self.step_window()
            
            if self.term_buf[0]:
                break
        
        return {
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "windows_completed": self.windows_completed,
            "avg_reward_per_window": self.total_reward / max(1, self.windows_completed),
            "avg_reward_per_step": self.total_reward / max(1, self.total_steps),
        }
    
    def get_active_count(self):
        """Get the number of currently active (tracked) targets."""
        end_idx = GRID_SIZE + self.max_trackers * FEATURES_PER_TRACKER
        flat_trackers = self.obs_buf[0, GRID_SIZE:end_idx].reshape(-1, FEATURES_PER_TRACKER)
        # FEATURES_PER_TRACKER=3 => [t_desired, t_deadline, t_dwell].
        # Slot is active if t_desired is not NO_TARGET.
        return int(np.sum(flat_trackers[:, 0] != NO_TARGET))
    
    def close(self):
        """Close the environment."""
        if not getattr(self, '_closed', False):
            binding.vec_close(self.env)
            self._closed = True


def benchmark_planner(planner, target_counts=[50, 100, 200, 500], num_windows=50, seed=1):
    """
    Benchmark a planner across different target loads. Testing utility
    
    Args:
        planner: Planner instance with a plan() method.
        target_counts: List of target counts to test.
        num_windows: Windows per test.
        seed: Random seed.
    
    Returns:
        dict: Results mapping target_count -> avg_reward.
    """
    results = {}
    
    for n_targets in target_counts:
        engine = RadarEngine(planner, initial_targets=n_targets, seed=seed)
        stats = engine.run_episode(num_windows)
        results[n_targets] = stats["avg_reward_per_step"]
        engine.close()
        
        print(f"  {n_targets} targets: {stats['avg_reward_per_step']:.4f} reward/step")
    
    return results
