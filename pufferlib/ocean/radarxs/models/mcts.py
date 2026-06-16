"""
Pure MCTS Planner for Radar Task Scheduling.
Uses heuristic-based tree search without neural network guidance.
"""
import numpy as np


from .planner import Planner


SEARCH_DELAY_MODE = 0  # 0=linear, 1=exponential
SEARCH_PENALTY = 0.1 / 1000.0
SEARCH_DEBT_PENALTY_WEIGHT = SEARCH_PENALTY
SEARCH_DEBT_TAU_MS = 10
SEARCH_DELAY_PENALTY_CAP = -1.0
SECTOR_STALENESS_MODE = 0  # 0=linear, 1=exponential
SECTOR_STALENESS_TAU_MS = 500


def _initial_staggered_grid():
    grid = np.zeros((300,), dtype=np.float32)
    macro_rows = 10 // 2
    macro_cols = 30 // 2
    macro_count = macro_rows * macro_cols
    macro_step_ms = 3000.0 / float(max(1, macro_count))
    for macro_r in range(macro_rows):
        for macro_c in range(macro_cols):
            macro_idx = macro_r * macro_cols + macro_c
            freshness = max(0.0, 3000.0 - macro_step_ms * float(macro_idx))
            base_r = macro_r * 2
            base_c = macro_c * 2
            sectors = [
                base_c + base_r * 30,
                base_c + 1 + base_r * 30,
                base_c + (base_r + 1) * 30,
                base_c + 1 + (base_r + 1) * 30,
            ]
            grid[np.asarray(sectors, dtype=np.int32)] = freshness
    return grid

class Node:
    """MCTS Tree Node."""
    
    def __init__(
        self,
        t_desired,
        t_deadline,
        t_dwell,
        priority,
        active_mask,
        grid=None,
        az_bin=None,
        el_bin=None,
        tracked_mask=None,
        refresh_t_desired=None,
        refresh_t_deadline=None,
        search_debt_ms=0.0,
        parent=None,
        action=None,
        prior_prob=0.0,
        edge_reward=0.0,
    ):
        self.t_desired = t_desired.copy()
        self.t_deadline = t_deadline.copy()
        self.t_dwell = t_dwell.copy()
        self.priority = priority.copy()
        self.active_mask = active_mask.copy()
        self.grid = grid.copy() if grid is not None else _initial_staggered_grid()
        self.az_bin = az_bin.copy() if az_bin is not None else np.zeros_like(t_desired, dtype=np.float32)
        self.el_bin = el_bin.copy() if el_bin is not None else np.zeros_like(t_desired, dtype=np.float32)
        self.tracked_mask = (
            tracked_mask.copy()
            if tracked_mask is not None
            else (self.active_mask.copy() & (self.t_deadline > 0))
        )
        if refresh_t_desired is None or refresh_t_deadline is None:
            inferred_des, inferred_dead = self._infer_refresh_timers(
                self.t_desired, self.t_deadline, self.priority
            )
            self.refresh_t_desired = inferred_des
            self.refresh_t_deadline = inferred_dead
        else:
            self.refresh_t_desired = refresh_t_desired.copy()
            self.refresh_t_deadline = refresh_t_deadline.copy()
        self.search_debt_ms = float(search_debt_ms)
        
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.edge_reward = float(edge_reward)
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.expanded = False
        self.nn_priors = None

    @staticmethod
    def _infer_refresh_timers(t_desired, t_deadline, priority, revisit_time_scale=1.0):
        denom = np.maximum(1e-3, 1.5 + 0.75 * priority.astype(np.float32))
        base_desired = (t_deadline - t_desired) / denom
        base_desired = base_desired * float(revisit_time_scale)
        base_desired = np.clip(base_desired, 100.0, 30000.0).astype(np.float32)
        deadline_mult = 2.5 - 0.75 * priority.astype(np.float32)
        deadline_mult = np.maximum(1.0, deadline_mult)
        base_deadline = np.clip(base_desired * deadline_mult, 100.0, 30000.0)
        return base_desired.astype(np.float32), base_deadline.astype(np.float32)
    
    def is_terminal(self):
        return not np.any(self.active_mask)
    
    def get_valid_actions(self):
        """Return list of valid actions (0=Search, 1..N=Track).
        Only tracked (discovered) targets are valid track actions."""
        valid = [0]  # Search always valid
        scheduled = getattr(self, "scheduled_mask", None)
        if scheduled is None:
            scheduled = np.zeros_like(self.active_mask, dtype=bool)
        valid.extend((np.where(self.active_mask & self.tracked_mask & ~scheduled)[0] + 1).tolist())
        return valid
    
    def calc_urgency_cost(self):
        """Lower is better - penalize overdue and near-deadline targets."""
        if not np.any(self.active_mask):
            return 0.0
        overdue = np.maximum(0, -self.t_desired) * self.active_mask
        deadline_urgency = np.maximum(0, 100 - self.t_deadline) * self.active_mask
        return np.sum(overdue) + 0.1 * np.sum(deadline_urgency)


class MCTSPlanner(Planner):
    """Pure MCTS with EST heuristic rollout."""
    
    def __init__(
        self,
        max_trackers=500,
        num_rollouts=100,
        exploration_constant=2.0,
        tardiness_mode="local",
        local_tardiness_weight=1.0,
        global_tardiness_weight=1.0,
        normalize_delay_penalty=True,
        global_aggregation="sum",
        tardiness_accounting="legacy",
        settle_rollout_debt=True,
        enable_search_refresh_tracked=True,
        search_refresh_gain=1.0,
        search_action_reward=0.1,
        track_update_reward=0.1,
        track_loss_penalty=1.0,
        track_urgency_bonus_weight=0.0,
        track_uncertainty_bonus_weight=0.0,
        target_service_weight=0.0,
        target_service_horizon_ms=1000.0,
        sector_staleness_weight=0.0,
        sector_target_cycle_ms=-1.0,
        searched_sector_reward_weight=0.0,
        search_frame_overdue_weight=0.0,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=0.0,
        enable_track_beam_scan=False,
        revisit_time_scale=1.0,
        search_delay_mode=SEARCH_DELAY_MODE,
        search_debt_penalty_weight=SEARCH_DEBT_PENALTY_WEIGHT,
        search_debt_tau_ms=SEARCH_DEBT_TAU_MS,
        search_delay_penalty_cap=SEARCH_DELAY_PENALTY_CAP,
        search_delay_overdue_gate_threshold=-1.0,
        search_delay_overdue_gate_min_scale=0.0,
        search_delay_gate_metric="overdue_frac",
        search_delay_gate_delay_ms=1000.0,
        search_delay_gate_rescue_debt_ms=-1.0,
        search_delay_gate_rescue_min_scale=0.0,
        penalize_hidden_targets=False,
        rollout_candidate_cap=96,
        simulation_window_ms=200.0,
    ):
        super().__init__(max_trackers)
        self.num_rollouts = num_rollouts
        self.c = exploration_constant
        self.SEARCH_ACTION = 0
        # tardiness_mode: local | global | hybrid | global_integral
        # global_integral = single global objective updated every simulated step.
        self.tardiness_mode = tardiness_mode
        self.local_tardiness_weight = float(local_tardiness_weight)
        self.global_tardiness_weight = float(global_tardiness_weight)
        self.normalize_delay_penalty = bool(normalize_delay_penalty)
        self.global_aggregation = str(global_aggregation).lower()
        if self.global_aggregation not in ("sum", "mean"):
            raise ValueError(f"Unsupported global_aggregation: {global_aggregation}")
        self.tardiness_accounting = str(tardiness_accounting).lower()
        if self.tardiness_accounting not in ("legacy", "deferred"):
            raise ValueError(f"Unsupported tardiness_accounting: {tardiness_accounting}")
        self.settle_rollout_debt = bool(settle_rollout_debt)
        self.enable_search_refresh_tracked = bool(enable_search_refresh_tracked)
        self.search_refresh_gain = float(search_refresh_gain)
        self.search_action_reward = float(search_action_reward)
        self.track_update_reward = float(track_update_reward)
        self.track_loss_penalty = float(track_loss_penalty)
        self.track_urgency_bonus_weight = float(track_urgency_bonus_weight)
        self.track_uncertainty_bonus_weight = float(track_uncertainty_bonus_weight)
        self.target_service_weight = float(target_service_weight)
        self.target_service_horizon_ms = float(target_service_horizon_ms)
        self.sector_staleness_weight = float(sector_staleness_weight)
        self.sector_target_cycle_ms = float(sector_target_cycle_ms)
        self.searched_sector_reward_weight = float(searched_sector_reward_weight)
        self.search_frame_overdue_weight = float(search_frame_overdue_weight)
        self.search_frame_desired_ms = float(search_frame_desired_ms)
        self.search_frame_deadline_ms = float(search_frame_deadline_ms)
        self.search_frame_drop_penalty = float(search_frame_drop_penalty)
        self.enable_track_beam_scan = bool(enable_track_beam_scan)
        self.revisit_time_scale = float(revisit_time_scale)
        self.search_delay_mode = int(search_delay_mode)
        self.search_debt_penalty_weight = float(search_debt_penalty_weight)
        self.search_debt_tau_ms = float(search_debt_tau_ms)
        self.search_delay_penalty_cap = float(search_delay_penalty_cap)
        self.search_delay_overdue_gate_threshold = float(search_delay_overdue_gate_threshold)
        self.search_delay_overdue_gate_min_scale = float(search_delay_overdue_gate_min_scale)
        self.search_delay_gate_metric = str(search_delay_gate_metric).lower()
        self.search_delay_gate_delay_ms = float(search_delay_gate_delay_ms)
        self.search_delay_gate_rescue_debt_ms = float(search_delay_gate_rescue_debt_ms)
        self.search_delay_gate_rescue_min_scale = float(search_delay_gate_rescue_min_scale)
        self.penalize_hidden_targets = bool(penalize_hidden_targets)
        self.rollout_candidate_cap = int(max(4, rollout_candidate_cap))
        self.simulation_window_ms = float(max(10.0, simulation_window_ms))
        self._search_delay_gate_disabled = (
            (
                self.search_delay_gate_metric == "overdue_frac"
                and self.search_delay_overdue_gate_threshold <= 0.0
            )
            or (
                self.search_delay_gate_metric in ("mean_delay", "max_delay")
                and self.search_delay_gate_delay_ms <= 0.0
            )
        )
        self._needs_local_term = (
            self.tardiness_accounting == "deferred"
            and self.tardiness_mode in ("local", "hybrid")
        ) or (
            self.tardiness_accounting != "deferred"
            and (
                self.tardiness_mode == "local"
                or (
                    self.tardiness_mode == "hybrid"
                    and self.local_tardiness_weight > 0.0
                )
            )
        )
        self._needs_global_term = (
            self.tardiness_accounting == "deferred"
            and self.tardiness_mode in ("global", "hybrid", "global_integral")
        ) or (
            self.tardiness_accounting != "deferred"
            and self.tardiness_mode in ("global", "hybrid", "global_integral")
            and (
                self.tardiness_mode == "global_integral"
                or self.global_tardiness_weight > 0.0
            )
        )

    def _search_delay_penalty(self, search_debt_ms):
        if self.search_debt_penalty_weight <= 0.0 or search_debt_ms <= 0.0:
            return 0.0
        if self.search_delay_mode == 0:
            penalty = float(self.search_debt_penalty_weight * search_debt_ms)
        else:
            arg = min(float(search_debt_ms) / max(1e-3, self.search_debt_tau_ms), 20.0)
            penalty = float(
                self.search_debt_penalty_weight
                * (np.exp(arg) - 1.0)
            )
        if self.search_delay_penalty_cap >= 0.0:
            penalty = min(penalty, self.search_delay_penalty_cap)
        return penalty

    def _search_delay_gate_scale(self, t_desired, active_mask, tracked_mask, search_debt_ms=0.0):
        if self._search_delay_gate_disabled:
            return 1.0
        tracked_active = np.logical_and(active_mask, tracked_mask)
        denom = int(np.sum(tracked_active))
        if denom <= 0:
            return 1.0

        metric = self.search_delay_gate_metric
        if metric == "overdue_frac":
            threshold = self.search_delay_overdue_gate_threshold
            if threshold <= 0.0:
                return 1.0
            severity = float(np.mean(t_desired[tracked_active] < 0.0)) / threshold
        else:
            delay_ms = self.search_delay_gate_delay_ms
            if delay_ms <= 0.0:
                return 1.0
            tracked_delays = np.maximum(0.0, -t_desired[tracked_active])
            if metric == "mean_delay":
                severity = float(np.mean(tracked_delays)) / delay_ms
            elif metric == "max_delay":
                severity = float(np.max(tracked_delays)) / delay_ms
            else:
                raise ValueError(f"Unsupported search_delay_gate_metric: {self.search_delay_gate_metric}")

        scale = 1.0 - severity
        rescue_debt_ms = self.search_delay_gate_rescue_debt_ms
        if rescue_debt_ms > 0.0 and search_debt_ms >= rescue_debt_ms:
            scale = max(scale, self.search_delay_gate_rescue_min_scale)
        scale = max(float(self.search_delay_overdue_gate_min_scale), scale)
        return float(min(1.0, scale))

    def _track_uncertainty_proxy(self, t_desired, refresh_t_desired, t_dwell, action_idx):
        """Approximate motion-model uncertainty growth from revisit/dwell observables."""
        base_refresh = max(100.0, float(refresh_t_desired[action_idx]))
        dwell_ms = max(1.0, float(t_dwell[action_idx]))
        dwell_factor = pow(dwell_ms / 10.0, 0.25)
        maneuver_factor = pow(1000.0 / base_refresh, 2.5)
        lateness = max(0.0, -float(t_desired[action_idx]))
        lateness_factor = 1.0 + lateness / base_refresh
        raw = dwell_factor * maneuver_factor * lateness_factor
        return float(min(10.0, np.log1p(max(0.0, raw))))

    @staticmethod
    def _sector_staleness_penalty(grid_before, grid_after, weight):
        if weight <= 0.0 or grid_before is None or grid_after is None:
            return 0.0
        stale_before = float(np.mean(np.maximum(0.0, -grid_before)))
        stale_after = float(np.mean(np.maximum(0.0, -grid_after)))
        if SECTOR_STALENESS_MODE == 0:
            # Potential-based surveillance reward. Callers subtract this:
            # positive delta penalizes staleness growth; negative delta rewards
            # search actions that reduce stale sector debt.
            return float(weight * (stale_after - stale_before))
        before_arg = min(stale_before / SECTOR_STALENESS_TAU_MS, 20.0)
        after_arg = min(stale_after / SECTOR_STALENESS_TAU_MS, 20.0)
        return float(
            weight
            * (
                (np.exp(after_arg) - 1.0)
                - (np.exp(before_arg) - 1.0)
            )
        )

    @staticmethod
    def _sector_target_cycle_penalty(grid_before, grid_after, weight, target_cycle_ms):
        if weight <= 0.0 or target_cycle_ms <= 0.0 or grid_before is None or grid_after is None:
            return 0.0
        excess_before = float(np.mean(np.maximum(0.0, -grid_before - target_cycle_ms))) / max(1e-3, target_cycle_ms)
        excess_after = float(np.mean(np.maximum(0.0, -grid_after - target_cycle_ms))) / max(1e-3, target_cycle_ms)
        if excess_after <= excess_before:
            return 0.0
        return float(weight * (excess_after - excess_before))

    def _searched_sector_reward(self, grid_before, refreshed_sectors):
        if (
            self.searched_sector_reward_weight <= 0.0
            or self.search_frame_desired_ms <= 0.0
            or grid_before is None
            or refreshed_sectors is None
            or len(refreshed_sectors) == 0
        ):
            return 0.0
        vals = grid_before[np.asarray(refreshed_sectors, dtype=np.int32)]
        age = np.maximum(0.0, 3000.0 - vals)
        debt = np.clip(age / self.search_frame_desired_ms, 0.0, 1.0)
        return float(self.searched_sector_reward_weight * np.sum(debt))

    def _search_frame_overdue_penalty(self, grid_after):
        if (
            self.search_frame_overdue_weight <= 0.0
            or self.search_frame_desired_ms <= 0.0
            or grid_after is None
        ):
            return 0.0
        age = 3000.0 - grid_after
        overdue = np.maximum(0.0, age - self.search_frame_desired_ms)
        norm = overdue / max(1e-6, self.search_frame_desired_ms)
        frame_cost = float(np.mean(norm * norm))
        if self.search_frame_drop_penalty > 0.0 and self.search_frame_deadline_ms > 0.0:
            frame_cost += float(self.search_frame_drop_penalty * np.mean(age > self.search_frame_deadline_ms))
        return float(self.search_frame_overdue_weight * frame_cost)

    def _target_service_cost(self, t_desired, active_mask, tracked_mask, priority):
        if self.target_service_weight <= 0.0 or self.target_service_horizon_ms <= 0.0:
            return 0.0
        mask = np.asarray(active_mask, dtype=bool) & np.asarray(tracked_mask, dtype=bool)
        if not np.any(mask):
            return 0.0
        horizon = max(1e-6, float(self.target_service_horizon_ms))
        pressure = np.maximum(0.0, (horizon - np.asarray(t_desired, dtype=np.float32)[mask]) / horizon)
        scale = 1.0 + 2.0 * np.asarray(priority, dtype=np.float32)[mask]
        return float(self.target_service_weight * np.sum(scale * pressure * pressure))
    
    def plan(self, obs, budget_ms=200):
        """
        Generate action plan using MCTS.
        
        Args:
            obs: Dict with 't_desired', 't_deadline', 'priority', 'active_mask'
            budget_ms: Time budget (used to estimate max steps, approx 10ms per step)
        
        Returns:
            List[int]: Actions
        """
        # Approx conversion: Average step is ~10ms. 
        # But soft windows allow packing many short tasks (e.g. 2ms track).
        # Increasing max_steps limit to 50 to allow flexible packing.
        max_steps = 50 

        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask'],
            grid=obs.get('grid', None),
            az_bin=obs.get('az_bin', None),
            el_bin=obs.get('el_bin', None),
            tracked_mask=(obs['active_mask'] & (obs['t_deadline'] > 0)),
            search_debt_ms=float(obs.get('search_debt_ms', 0.0)),
        )
        root.refresh_t_desired, root.refresh_t_deadline = Node._infer_refresh_timers(
            root.t_desired, root.t_deadline, root.priority, self.revisit_time_scale
        )
        
        # Determine steps
        if max_steps is None:
            max_steps = int(np.sum(obs['active_mask'])) + 1  # All active + 1 search
        
        # Run MCTS
        for _ in range(self.num_rollouts):
            leaf = self._select(root)
            if not leaf.is_terminal() and not leaf.expanded:
                self._expand(leaf)
            reward = self._simulate(leaf)
            self._backprop(leaf, reward)
        
        # Extract plan
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                if not node.is_terminal():
                    self._expand(node)
                if not node.children:
                    break

            if getattr(self, "action_selection", "visits") == "q":
                best_child = self._best_child_by_q(node.children)
            else:
                # Use visit counts for action extraction. This is more stable
                # for policy distillation, but Q extraction is often better for
                # direct online scheduling with dense rewards.
                best_child = self._best_child_by_visits(node.children)
            plan.append(best_child.action)
            node = best_child
            
            if node.is_terminal():
                break
        
        return plan

    def plan_with_policy(self, obs, budget_ms=200):
        """
        Generate action plan and normalized root visit distribution.

        This exposes a strong pure-MCTS reference planner for policy distillation.
        """
        max_steps = 50

        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask'],
            grid=obs.get('grid', None),
            az_bin=obs.get('az_bin', None),
            el_bin=obs.get('el_bin', None),
            tracked_mask=(obs['active_mask'] & (obs['t_deadline'] > 0)),
            search_debt_ms=float(obs.get('search_debt_ms', 0.0)),
        )
        root.refresh_t_desired, root.refresh_t_deadline = Node._infer_refresh_timers(
            root.t_desired, root.t_deadline, root.priority, self.revisit_time_scale
        )

        for _ in range(self.num_rollouts):
            leaf = self._select(root)
            if not leaf.is_terminal() and not leaf.expanded:
                self._expand(leaf)
            reward = self._simulate(leaf)
            self._backprop(leaf, reward)

        policy = np.zeros((len(obs['t_desired']) + 1,), dtype=np.float32)
        total_visits = sum(child.visits for child in root.children)
        if total_visits > 0:
            for child in root.children:
                policy[child.action] = child.visits / total_visits

        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                if not node.is_terminal():
                    self._expand(node)
                if not node.children:
                    break

            if getattr(self, "action_selection", "visits") == "q":
                best_child = self._best_child_by_q(node.children)
            else:
                best_child = self._best_child_by_visits(node.children)
            plan.append(best_child.action)
            node = best_child

            if node.is_terminal():
                break

        return plan, policy
    
    def _select(self, node):
        while node.expanded and node.children and not node.is_terminal():
            node = self._ucb_select(node)
        return node

    @staticmethod
    def _best_child_by_visits(children):
        max_visits = max(c.visits for c in children)
        tied = [c for c in children if c.visits == max_visits]
        best_q = max((c.edge_reward + c.total_reward / max(1, c.visits)) for c in tied)
        top = [c for c in tied if (c.edge_reward + c.total_reward / max(1, c.visits)) == best_q]
        return top[np.random.randint(len(top))]

    @staticmethod
    def _best_child_by_q(children):
        visited = [c for c in children if c.visits > 0]
        pool = visited if visited else children
        best_q = max((c.edge_reward + c.total_reward / max(1, c.visits)) for c in pool)
        top = [c for c in pool if (c.edge_reward + c.total_reward / max(1, c.visits)) == best_q]
        return top[np.random.randint(len(top))]
    
    def _ucb_select(self, node):
        if not node.children:
            return None

        # In radar scheduling the root action space can be much larger than the
        # rollout budget.  Randomly sampling unvisited children makes low-rollout
        # MCTS mostly noise.  Order first visits by true one-step reward plus
        # prior mass; deeper rollout value then refines these candidates.
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            best = max(float(child.edge_reward) + self.c * float(child.prior_prob) for child in unvisited)
            top = [
                child for child in unvisited
                if float(child.edge_reward) + self.c * float(child.prior_prob) == best
            ]
            return top[np.random.randint(len(top))]

        best_score, best_child = -np.inf, node.children[0]
        
        for child in node.children:
            exploit = child.edge_reward + child.total_reward / max(1, child.visits)
            # PUCT formula: Q + C * P * sqrt(parent_N) / (1 + child_N)
            explore = self.c * child.prior_prob * np.sqrt(node.visits + 1) / (1 + child.visits)
            score = exploit + explore
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def _action_edge_reward(self, node, action, dwell_time, refreshed_sectors=None):
        """Immediate reward for taking action from node before rollout continues."""
        step_reward = 0.0
        local_term = 0.0
        global_term = 0.0
        TRACK_DELAY_PENALTY = 0.001
        if action == 0:
            step_reward += self.search_action_reward
            next_search_debt_ms = float(dwell_time)
            if refreshed_sectors is not None and len(refreshed_sectors) > 0 and node.grid is not None:
                refreshed_vals = node.grid[np.asarray(refreshed_sectors, dtype=np.int32)]
                if self.searched_sector_reward_weight <= 0.0 and self.search_frame_overdue_weight <= 0.0:
                    step_reward += float(np.sum(refreshed_vals[refreshed_vals < 0.0])) * SEARCH_PENALTY
                step_reward += self._searched_sector_reward(node.grid, refreshed_sectors)
        else:
            action_idx = action - 1
            next_search_debt_ms = float(node.search_debt_ms + dwell_time)
            if (not node.tracked_mask[action_idx]) or node.t_deadline[action_idx] <= 0:
                step_reward -= self.track_loss_penalty * (1.0 + 2.0 * float(node.priority[action_idx]))
            else:
                step_reward += self.track_update_reward
                priority_scale = 1.0 + 2.0 * float(node.priority[action_idx])
                tardiness = max(0.0, -float(node.t_desired[action_idx]))
                deadline_pressure = 0.0
                if self.track_urgency_bonus_weight > 0.0 or self.track_uncertainty_bonus_weight > 0.0:
                    deadline_pressure = max(0.0, 100.0 - float(node.t_deadline[action_idx]))
                if self.track_urgency_bonus_weight > 0.0:
                    step_reward += self.track_urgency_bonus_weight * (
                        tardiness * priority_scale * TRACK_DELAY_PENALTY
                        + 0.25 * deadline_pressure * priority_scale * TRACK_DELAY_PENALTY
                    )
                if self.track_uncertainty_bonus_weight > 0.0:
                    uncertainty_proxy = self._track_uncertainty_proxy(
                        node.t_desired, node.refresh_t_desired, node.t_dwell, action_idx
                    )
                    step_reward += self.track_uncertainty_bonus_weight * uncertainty_proxy * (
                        tardiness * priority_scale * TRACK_DELAY_PENALTY
                        + 0.25 * deadline_pressure * priority_scale * TRACK_DELAY_PENALTY
                    )
                if self._needs_local_term and tardiness > 0:
                    local_term = tardiness * priority_scale

        if self.sector_staleness_weight > 0.0 and node.grid is not None:
            grid_after = node.grid.copy()
            if refreshed_sectors is not None and len(refreshed_sectors) > 0:
                # Match env semantics: freshness is measured from the end of the
                # 10ms search dwell, so a searched sector should remain at 3000ms
                # after the step, not immediately decay to 2990ms.
                grid_after[np.asarray(refreshed_sectors, dtype=np.int32)] = 3010.0
            grid_after -= dwell_time
            step_reward -= self._sector_staleness_penalty(
                node.grid, grid_after, self.sector_staleness_weight
            )
            step_reward -= self._sector_target_cycle_penalty(
                node.grid, grid_after, self.sector_staleness_weight, self.sector_target_cycle_ms
            )

        if self.search_frame_overdue_weight > 0.0 and node.grid is not None:
            grid_after = node.grid.copy()
            if refreshed_sectors is not None and len(refreshed_sectors) > 0:
                grid_after[np.asarray(refreshed_sectors, dtype=np.int32)] = 3010.0
            grid_after -= dwell_time
            step_reward -= self._search_frame_overdue_penalty(grid_after)

        if self.target_service_weight > 0.0:
            t_des_after = node.t_desired.copy()
            t_dead_after = node.t_deadline.copy()
            tracked_after = node.tracked_mask.copy()
            active_after = node.active_mask.copy()
            if action != 0:
                action_idx = int(action) - 1
                if (
                    0 <= action_idx < len(t_des_after)
                    and tracked_after[action_idx]
                    and t_dead_after[action_idx] > 0
                ):
                    t_des_after[action_idx] = node.refresh_t_desired[action_idx]
                    t_dead_after[action_idx] = node.refresh_t_deadline[action_idx]
            timer_mask = active_after & (tracked_after | self.penalize_hidden_targets)
            t_des_after[timer_mask] -= dwell_time
            t_dead_after[timer_mask] -= dwell_time
            expired = timer_mask & (t_dead_after <= 0)
            tracked_after[expired] = False
            if not self.penalize_hidden_targets:
                active_after[expired] = False
            before_cost = self._target_service_cost(
                node.t_desired, node.active_mask, node.tracked_mask, node.priority
            )
            after_cost = self._target_service_cost(
                t_des_after, active_after, tracked_after, node.priority
            )
            step_reward += before_cost - after_cost

        active_idx = None
        if self._needs_global_term:
            active_idx = np.where(node.active_mask & (node.tracked_mask | self.penalize_hidden_targets))[0]
        if active_idx is not None and len(active_idx) > 0:
            overdue_before = np.maximum(0.0, -node.t_desired[active_idx])
            overdue_after = np.maximum(0.0, -(node.t_desired[active_idx] - dwell_time))
            overdue_inc = overdue_after - overdue_before
            if np.any(overdue_inc > 0):
                overdue_norm = overdue_inc * (1.0 + 2.0 * node.priority[active_idx])
                if self.global_aggregation == "sum":
                    global_term = float(np.sum(overdue_norm))
                else:
                    global_term = float(np.mean(overdue_norm))

        if self.tardiness_accounting == "deferred":
            if self.tardiness_mode in ("global", "hybrid", "global_integral") and global_term > 0:
                step_reward -= self.global_tardiness_weight * TRACK_DELAY_PENALTY * global_term * dwell_time
        elif self.tardiness_mode == "global_integral":
            if global_term > 0:
                step_reward -= self.global_tardiness_weight * TRACK_DELAY_PENALTY * global_term * dwell_time
        elif self.normalize_delay_penalty:
            if self.tardiness_mode == "local":
                lw, gw = 1.0, 0.0
            elif self.tardiness_mode == "global":
                lw, gw = 0.0, 1.0
            else:
                lw, gw = self.local_tardiness_weight, self.global_tardiness_weight
            z = max(1e-8, lw + gw)
            mix_term = (lw / z) * local_term + (gw / z) * global_term
            step_reward -= TRACK_DELAY_PENALTY * mix_term
        else:
            if local_term > 0 and self.tardiness_mode in ("local", "hybrid"):
                step_reward -= self.local_tardiness_weight * TRACK_DELAY_PENALTY * local_term
            if global_term > 0 and self.tardiness_mode in ("global", "hybrid"):
                legacy_scale = max(1, len(active_idx)) if self.global_aggregation == "mean" else 1.0
                step_reward -= self.global_tardiness_weight * TRACK_DELAY_PENALTY * global_term * legacy_scale

        search_penalty = self._search_delay_penalty(next_search_debt_ms)
        if search_penalty > 0.0 and not self._search_delay_gate_disabled:
            search_penalty *= self._search_delay_gate_scale(
                node.t_desired, node.active_mask, node.tracked_mask, next_search_debt_ms
            )
        step_reward -= search_penalty

        return float(step_reward)

    def _preview_search_refresh(self, grid):
        if grid is None:
            return np.array([], dtype=np.int32)
        grid_copy = grid.copy()
        return self._refresh_stalest_sectors(grid_copy, n_sectors=4)
    
    def _expand(self, node, force_engagement=False, priors=None, top_k=None):
        valid_actions = node.get_valid_actions()
        if force_engagement and len(valid_actions) > 1:
            valid_actions = [a for a in valid_actions if a != 0]
            
        # Expand top_k by prior, but always force-include:
        #   - search (action 0) so new arrivals are discoverable
        #   - the most overdue tracked target so the model always sees the hardest case
        if priors is not None and top_k is not None and len(valid_actions) > top_k:
            valid_priors = [(a, priors[a]) for a in valid_actions]
            valid_priors.sort(key=lambda x: x[1], reverse=True)
            valid_actions = [x[0] for x in valid_priors[:top_k]]

            # Force search into tree if not already present
            if 0 not in valid_actions and any(a == 0 for a in [x[0] for x in valid_priors]):
                valid_actions[-1] = 0

            # Force the most overdue active tracked target into the tree.
            # This guarantees the hardest scheduling case is always reachable,
            # regardless of what the policy prior says.
            track_actions = [a for a in [x[0] for x in valid_priors] if a != 0]
            if track_actions:
                t_desired = node.t_desired
                t_deadline = node.t_deadline
                t_dwell = node.t_dwell
                tracked   = node.tracked_mask
                active    = node.active_mask
                # Most overdue = most negative t_desired among active tracked targets
                overdue_scores = []
                deadline_scores = []
                priority_scores = []
                for a in track_actions:
                    idx = a - 1
                    if 0 <= idx < len(t_desired) and active[idx] and tracked[idx]:
                        overdue_scores.append((a, t_desired[idx]))
                        slack = float(t_deadline[idx]) - max(1.0, float(t_dwell[idx]))
                        deadline_scores.append((a, slack))
                        priority_scores.append((a, -float(node.priority[idx])))
                if overdue_scores:
                    forced = [
                        min(overdue_scores, key=lambda x: x[1])[0],
                        min(deadline_scores, key=lambda x: x[1])[0],
                        min(priority_scores, key=lambda x: x[1])[0],
                    ]
                    for forced_action in forced:
                        if forced_action not in valid_actions:
                            protected = set([0] + forced)
                            replace_candidates = [
                                i for i, a in enumerate(valid_actions) if a not in protected
                            ]
                            replace_idx = replace_candidates[-1] if replace_candidates else len(valid_actions) - 1
                            valid_actions[replace_idx] = forced_action
        
        for action in valid_actions:
            # Copy state for child
            child_t_desired = node.t_desired.copy()
            child_t_deadline = node.t_deadline.copy()
            child_active = node.active_mask.copy()
            child_grid = node.grid.copy()
            child_tracked = node.tracked_mask.copy()
            
            # Simulate action effect
            edge_reward = 0.0
            if action == 0:  # SEARCH
                dwell_time = 10.0  # Search takes 10ms
                refreshed = self._refresh_stalest_sectors(child_grid, n_sectors=4)
                edge_reward = self._action_edge_reward(node, action, dwell_time, refreshed)
                child_search_debt_ms = dwell_time
                self._apply_search_refresh(
                    child_t_desired,
                    child_t_deadline,
                    child_tracked,
                    child_active,
                    node.az_bin,
                    node.el_bin,
                    refreshed,
                    node.refresh_t_desired,
                    node.refresh_t_deadline,
                    refresh_tracked=self.enable_search_refresh_tracked,
                    refresh_gain=self.search_refresh_gain,
                )
            else:  # TRACK
                action_idx = action - 1
                dwell_time = float(node.t_dwell[action_idx]) if action_idx < len(node.t_dwell) else 10.0
                refreshed = None
                if self.enable_track_beam_scan and 0 <= action_idx < len(child_tracked):
                    az = int(np.clip(round(float(node.az_bin[action_idx]) * 29.0), 0, 29))
                    el = int(np.clip(round(float(node.el_bin[action_idx]) * 9.0), 0, 9))
                    refreshed = np.array([el * 30 + az], dtype=np.int32)
                edge_reward = self._action_edge_reward(node, action, dwell_time, refreshed)
                child_search_debt_ms = float(node.search_debt_ms + dwell_time)
                if 0 <= action_idx < len(child_tracked):
                    if child_tracked[action_idx] and child_t_deadline[action_idx] > 0:
                        child_t_desired[action_idx] = node.refresh_t_desired[action_idx]
                        child_t_deadline[action_idx] = node.refresh_t_deadline[action_idx]
                        if self.enable_track_beam_scan:
                            self._refresh_tracker_sector(child_grid, node.az_bin[action_idx], node.el_bin[action_idx])
                            self._apply_discovery_only_refresh(
                                child_t_desired,
                                child_t_deadline,
                                child_tracked,
                                child_active,
                                node.az_bin,
                                node.el_bin,
                                refreshed,
                                node.refresh_t_desired,
                                node.refresh_t_deadline,
                            )
            
            # Time passes. In the C environment, hidden active targets keep
            # latent timers when penalize_hidden_targets is enabled; the old
            # tree model only advanced tracked timers, which made search look
            # nearly worthless after a target was lost.
            timer_mask = child_active & (child_tracked | self.penalize_hidden_targets)
            timer_idx = np.where(timer_mask)[0]
            child_t_desired[timer_idx] = child_t_desired[timer_idx] - dwell_time
            child_t_deadline[timer_idx] = child_t_deadline[timer_idx] - dwell_time
            child_grid = child_grid - dwell_time
            expired = np.where(timer_mask & (child_t_deadline <= 0))[0]
            if len(expired) > 0:
                edge_reward -= float(np.sum(self.track_loss_penalty * (1.0 + 2.0 * node.priority[expired])))
                child_tracked[expired] = False
                if self.penalize_hidden_targets:
                    child_t_desired[expired] = node.refresh_t_desired[expired]
                    child_t_deadline[expired] = node.refresh_t_deadline[expired]
                else:
                    child_active[expired] = False
                    child_t_desired[expired] = -1.0
                    child_t_deadline[expired] = -1.0
            
            prior = 1.0 / len(valid_actions) if priors is None else priors[action]
            
            child = Node(
                t_desired=child_t_desired,
                t_deadline=child_t_deadline,
                t_dwell=node.t_dwell,
                priority=node.priority,
                grid=child_grid,
                az_bin=node.az_bin,
                el_bin=node.el_bin,
                tracked_mask=child_tracked,
                refresh_t_desired=node.refresh_t_desired,
                refresh_t_deadline=node.refresh_t_deadline,
                search_debt_ms=child_search_debt_ms,
                action=action,
                active_mask=child_active,
                parent=node,
                prior_prob=prior,
                edge_reward=edge_reward,
            )
            parent_scheduled = getattr(node, "scheduled_mask", None)
            if parent_scheduled is None:
                child.scheduled_mask = np.zeros_like(child_active, dtype=bool)
            else:
                child.scheduled_mask = parent_scheduled.copy()
            if action != 0:
                action_idx = action - 1
                if 0 <= action_idx < len(child.scheduled_mask):
                    child.scheduled_mask[action_idx] = True
            node.children.append(child)
        node.expanded = True
    
    def _simulate(self, node):
        """Simulate rollout using the configured reward, not a stale hand-tuned heuristic."""
        total_reward = 0.0
        
        # Copy state for simulation
        active = node.active_mask.copy()
        t_desired = node.t_desired.copy()
        t_deadline = node.t_deadline.copy()
        priority = node.priority
        grid = node.grid.copy()
        az_bin = node.az_bin.copy()
        el_bin = node.el_bin.copy()
        tracked = node.tracked_mask.copy()
        refresh_t_desired = node.refresh_t_desired.copy()
        refresh_t_deadline = node.refresh_t_deadline.copy()
        
        # Constants from radarxs.h
        SEARCH_REWARD = self.search_action_reward
        TRACK_UPDATE_REWARD = self.track_update_reward
        TRACK_DELAY_PENALTY = 0.001
        TRACK_LOSS_PENALTY = self.track_loss_penalty
        
        simulated_time = 0.0
        WINDOW_MS = self.simulation_window_ms
        # For "deferred" accounting:
        # debt[i] accumulates overdue penalty mass over time for target i and is
        # charged when the target is serviced (local branch).
        delay_debt = np.zeros_like(t_desired, dtype=np.float32)
        search_debt_ms = float(node.search_debt_ms)
        # Simulate until window full (time-based, not step-based)
        while simulated_time < WINDOW_MS:
            # 1. Select action using the same configured reward terms as the tree edges.
            rollout_node = Node(
                t_desired=t_desired,
                t_deadline=t_deadline,
                t_dwell=node.t_dwell,
                priority=priority,
                active_mask=active,
                grid=grid,
                az_bin=az_bin,
                el_bin=el_bin,
                tracked_mask=tracked,
                refresh_t_desired=refresh_t_desired,
                refresh_t_deadline=refresh_t_deadline,
                search_debt_ms=search_debt_ms,
            )
            valid_actions = rollout_node.get_valid_actions()
            if len(valid_actions) > self.rollout_candidate_cap:
                tracked_active = active & tracked
                masked_deadline = np.where(tracked_active, t_deadline, np.inf)
                take = max(1, self.rollout_candidate_cap - 1)
                finite_count = int(np.sum(np.isfinite(masked_deadline)))
                if finite_count > take:
                    top_track_idx = np.argpartition(masked_deadline, take - 1)[:take]
                    top_track_idx = top_track_idx[np.argsort(masked_deadline[top_track_idx])]
                else:
                    top_track_idx = np.where(np.isfinite(masked_deadline))[0]
                valid_actions = [0] + [int(i) + 1 for i in top_track_idx if np.isfinite(masked_deadline[i])]

            rollout_policy = getattr(self, "rollout_policy", "greedy")
            if rollout_policy == "edf":
                tracked_active = active & tracked & (t_deadline > 0.0)
                search_period_ms = float(getattr(self, "rollout_search_period_ms", 120.0))
                if (0 in valid_actions) and (search_debt_ms >= search_period_ms or not np.any(tracked_active)):
                    best_action = 0
                else:
                    candidates = []
                    for action in valid_actions:
                        if action == 0:
                            continue
                        idx = action - 1
                        if 0 <= idx < len(t_deadline) and tracked_active[idx]:
                            candidates.append((float(t_deadline[idx]), action))
                    best_action = min(candidates, key=lambda x: x[0])[1] if candidates else 0
            else:
                best_action = 0
                best_action_score = -np.inf
                for action in valid_actions:
                    if action == 0:
                        dwell_time = 10.0
                        refreshed = self._preview_search_refresh(grid)
                    else:
                        action_idx = action - 1
                        dwell_time = float(node.t_dwell[action_idx]) if action_idx < len(node.t_dwell) else 10.0
                        refreshed = None
                        if self.enable_track_beam_scan and 0 <= action_idx < len(tracked):
                            az = int(np.clip(round(float(az_bin[action_idx]) * 29.0), 0, 29))
                            el = int(np.clip(round(float(el_bin[action_idx]) * 9.0), 0, 9))
                            refreshed = np.array([el * 30 + az], dtype=np.int32)
                    score = self._action_edge_reward(rollout_node, action, dwell_time, refreshed)
                    if score > best_action_score:
                        best_action_score = score
                        best_action = action

            action_idx = best_action - 1 if best_action != 0 else -1
            
            # 2. Determine Dwell Time
            dwell_time = 10.0 # Default/Search
            if action_idx != -1 and hasattr(node, 't_dwell'):
                # Actual dwell time from sensor estimate
                dwell_time = float(node.t_dwell[action_idx])
            
            # 3. Calculate Reward
            step_reward = 0.0
            
            local_term = 0.0
            global_term = 0.0

            if action_idx == -1:
                # Search Action
                step_reward += SEARCH_REWARD
                search_debt_ms = dwell_time
                grid_before_search = grid.copy()
                refreshed = self._refresh_stalest_sectors(grid, n_sectors=4)
                step_reward += self._searched_sector_reward(grid_before_search, refreshed)
                if (
                    len(refreshed) > 0
                    and self.searched_sector_reward_weight <= 0.0
                    and self.search_frame_overdue_weight <= 0.0
                ):
                    refreshed_vals = grid_before_search[np.asarray(refreshed, dtype=np.int32)]
                    step_reward += float(np.sum(refreshed_vals[refreshed_vals < 0.0])) * SEARCH_PENALTY
                self._apply_search_refresh(
                    t_desired,
                    t_deadline,
                    tracked,
                    active,
                    az_bin,
                    el_bin,
                    refreshed,
                    refresh_t_desired,
                    refresh_t_deadline,
                    refresh_tracked=self.enable_search_refresh_tracked,
                    refresh_gain=self.search_refresh_gain,
                )
            else:
                # Track Action
                search_debt_ms += dwell_time
                if (not tracked[action_idx]) or t_deadline[action_idx] <= 0:
                    step_reward -= TRACK_LOSS_PENALTY * (1.0 + 2.0 * float(priority[action_idx]))
                else:
                    step_reward += TRACK_UPDATE_REWARD
                    priority_scale = 1.0 + 2.0 * float(priority[action_idx])
                    tardiness = max(0, -t_desired[action_idx])
                    deadline_pressure = max(0.0, 100.0 - t_deadline[action_idx])
                    step_reward += self.track_urgency_bonus_weight * (
                        tardiness * priority_scale * TRACK_DELAY_PENALTY
                        + 0.25 * deadline_pressure * priority_scale * TRACK_DELAY_PENALTY
                    )
                    if self.track_uncertainty_bonus_weight > 0.0:
                        uncertainty_proxy = self._track_uncertainty_proxy(
                            t_desired, refresh_t_desired, node.t_dwell, action_idx
                        )
                        step_reward += self.track_uncertainty_bonus_weight * uncertainty_proxy * (
                            tardiness * priority_scale * TRACK_DELAY_PENALTY
                            + 0.25 * deadline_pressure * priority_scale * TRACK_DELAY_PENALTY
                        )
                    if tardiness > 0:
                        local_term = tardiness * priority_scale
                    t_desired[action_idx] = refresh_t_desired[action_idx]
                    t_deadline[action_idx] = refresh_t_deadline[action_idx]
                    if self.enable_track_beam_scan:
                        self._refresh_tracker_sector(grid, az_bin[action_idx], el_bin[action_idx])
                        az = int(np.clip(round(float(az_bin[action_idx]) * 29.0), 0, 29))
                        el = int(np.clip(round(float(el_bin[action_idx]) * 9.0), 0, 9))
                        refreshed = np.array([el * 30 + az], dtype=np.int32)
                        self._apply_discovery_only_refresh(
                            t_desired,
                            t_deadline,
                            tracked,
                            active,
                            az_bin,
                            el_bin,
                            refreshed,
                            refresh_t_desired,
                            refresh_t_deadline,
                        )

            # Global tardiness term: penalize *overdue increment* over this dwell.
            # This matches service-time local delay in mass, but distributes it in time.
            timer_mask = active & (tracked | self.penalize_hidden_targets)
            active_idx = np.where(timer_mask)[0]
            if len(active_idx) > 0:
                overdue_before = np.maximum(0.0, -t_desired[active_idx])
                overdue_after = np.maximum(0.0, -(t_desired[active_idx] - dwell_time))
                overdue_inc = overdue_after - overdue_before
                if np.any(overdue_inc > 0):
                    overdue_norm = overdue_inc * (1.0 + 2.0 * priority[active_idx])
                    if self.global_aggregation == "sum":
                        global_term = float(np.sum(overdue_norm))
                    else:
                        global_term = float(np.mean(overdue_norm))

            if self.tardiness_accounting == "deferred":
                # Same base quantity for local/global: overdue_norm * dwell_time.
                # Local = deferred charge at service time.
                # Global = immediate charge each step.
                if len(active_idx) > 0:
                    overdue = np.maximum(0.0, -t_desired[active_idx])
                    if np.any(overdue > 0):
                        overdue_norm = overdue * (1.0 + 2.0 * priority[active_idx])
                        debt_inc = overdue_norm * dwell_time
                        if self.tardiness_mode in ("local", "hybrid"):
                            delay_debt[active_idx] += debt_inc
                        if self.tardiness_mode in ("global", "hybrid", "global_integral"):
                            if self.global_aggregation == "sum":
                                global_flow = float(np.sum(debt_inc))
                            else:
                                global_flow = float(np.mean(debt_inc))
                            step_reward -= (
                                self.global_tardiness_weight * TRACK_DELAY_PENALTY * global_flow
                            )

                # On service, realize accumulated local debt for that target.
                if action_idx != -1 and self.tardiness_mode in ("local", "hybrid"):
                    if t_deadline[action_idx] > 0:
                        if delay_debt[action_idx] > 0:
                            step_reward -= (
                                self.local_tardiness_weight
                                * TRACK_DELAY_PENALTY
                                * float(delay_debt[action_idx])
                            )
                        delay_debt[action_idx] = 0.0
                # If targets expire, realize any deferred local debt before deactivation.
                if self.tardiness_mode in ("local", "hybrid"):
                    for i in range(len(active)):
                        if active[i] and t_deadline[i] <= dwell_time and delay_debt[i] > 0:
                            step_reward -= (
                                self.local_tardiness_weight
                                * TRACK_DELAY_PENALTY
                                * float(delay_debt[i])
                            )
                            delay_debt[i] = 0.0
            elif self.tardiness_mode == "global_integral":
                # Pure global flow penalty:
                # penalize aggregate overdue every simulated step (scaled by dwell).
                # This avoids local/global unit-mismatch in mixed objectives.
                if global_term > 0:
                    step_reward -= (
                        self.global_tardiness_weight
                        * TRACK_DELAY_PENALTY
                        * global_term
                        * dwell_time
                    )
            elif self.normalize_delay_penalty:
                # Fair-mode mixing: weights are treated as ratios and normalized to a fixed scale.
                if self.tardiness_mode == "local":
                    lw, gw = 1.0, 0.0
                elif self.tardiness_mode == "global":
                    lw, gw = 0.0, 1.0
                else:
                    lw, gw = self.local_tardiness_weight, self.global_tardiness_weight

                z = max(1e-8, lw + gw)
                mix_term = (lw / z) * local_term + (gw / z) * global_term
                step_reward -= TRACK_DELAY_PENALTY * mix_term
            else:
                # Legacy behavior kept for backwards reproducibility.
                if local_term > 0 and self.tardiness_mode in ("local", "hybrid"):
                    step_reward -= self.local_tardiness_weight * TRACK_DELAY_PENALTY * local_term
                if global_term > 0 and self.tardiness_mode in ("global", "hybrid"):
                    legacy_scale = max(1, len(active_idx)) if self.global_aggregation == "mean" else 1.0
                    step_reward -= self.global_tardiness_weight * TRACK_DELAY_PENALTY * global_term * legacy_scale

            search_penalty = self._search_delay_penalty(search_debt_ms)
            if search_penalty > 0.0:
                search_penalty *= self._search_delay_gate_scale(
                    t_desired, active, tracked, search_debt_ms
                )
            step_reward -= search_penalty
                    
            total_reward += step_reward
            
            # 4. Advance Time
            simulated_time += dwell_time
            grid_before_decay = grid.copy()
            timer_mask = active & (tracked | self.penalize_hidden_targets)
            timer_idx = np.where(timer_mask)[0]
            t_desired[timer_idx] -= dwell_time
            t_deadline[timer_idx] -= dwell_time
            grid -= dwell_time
            if self.sector_staleness_weight > 0.0:
                total_reward -= self._sector_staleness_penalty(
                    grid_before_decay, grid, self.sector_staleness_weight
                )
                total_reward -= self._sector_target_cycle_penalty(
                    grid_before_decay, grid, self.sector_staleness_weight, self.sector_target_cycle_ms
                )
            if self.search_frame_overdue_weight > 0.0:
                total_reward -= self._search_frame_overdue_penalty(grid)
            
            # Check for Timeouts
            for i in range(len(active)):
                if active[i] and (tracked[i] or self.penalize_hidden_targets) and t_deadline[i] <= 0:
                    total_reward -= TRACK_LOSS_PENALTY * (1.0 + 2.0 * float(priority[i]))
                    tracked[i] = False
                    if self.penalize_hidden_targets:
                        t_desired[i] = refresh_t_desired[i]
                        t_deadline[i] = refresh_t_deadline[i]
                    else:
                        active[i] = False
                        t_desired[i] = -1.0
                        t_deadline[i] = -1.0

            if not np.any(active & tracked) and action_idx == -1:
                # If no targets active and we just searched, maybe fast forward?
                # For now just continue searching
                pass

        # Deferred-local fairness: settle any remaining accumulated debt
        # at rollout boundary so local/global compare the same total mass.
        if (
            self.settle_rollout_debt
            and self.tardiness_accounting == "deferred"
            and self.tardiness_mode in ("local", "hybrid")
        ):
            remaining = float(np.sum(delay_debt))
            if remaining > 0:
                total_reward -= (
                    self.local_tardiness_weight
                    * TRACK_DELAY_PENALTY
                    * remaining
                )
        
        return total_reward

    @staticmethod
    def _refresh_stalest_sectors(grid, n_sectors=4):
        if grid is None or len(grid) < 300:
            return np.array([], dtype=np.int32)

        grid2d = grid.reshape(10, 30)
        best_r = 0
        best_c = 0
        best_sum = np.inf
        for r in range(0, 9, 2):
            for c in range(0, 29, 2):
                s = (
                    grid2d[r, c]
                    + grid2d[r, c + 1]
                    + grid2d[r + 1, c]
                    + grid2d[r + 1, c + 1]
                )
                if s < best_sum:
                    best_sum = s
                    best_r = r
                    best_c = c

        idx = np.array(
            [
                best_r * 30 + best_c,
                best_r * 30 + (best_c + 1),
                (best_r + 1) * 30 + best_c,
                (best_r + 1) * 30 + (best_c + 1),
            ],
            dtype=np.int32,
        )
        # Match env semantics: freshness is measured from the end of the search
        # action, so searched sectors remain at 3000ms after the 10ms step.
        grid[idx] = 3010.0
        return idx

    @staticmethod
    def _refresh_tracker_sector(grid, az_bin, el_bin):
        if grid is None or len(grid) < 300:
            return
        az = int(np.clip(round(float(az_bin) * 29.0), 0, 29))
        el = int(np.clip(round(float(el_bin) * 9.0), 0, 9))
        sector = el * 30 + az
        if 0 <= sector < len(grid):
            grid[sector] = 3000.0

    @staticmethod
    def _apply_discovery_only_refresh(
        t_desired,
        t_deadline,
        tracked_mask,
        active_mask,
        az_bin,
        el_bin,
        refreshed_sectors,
        refresh_t_desired,
        refresh_t_deadline,
    ):
        if refreshed_sectors is None or len(refreshed_sectors) == 0:
            return
        az = np.clip(np.round(az_bin.astype(np.float32) * 29.0).astype(np.int32), 0, 29)
        el = np.clip(np.round(el_bin.astype(np.float32) * 9.0).astype(np.int32), 0, 9)
        sectors = el * 30 + az
        refreshed_mask = np.isin(sectors, refreshed_sectors)
        untracked_hit = np.where(active_mask & (~tracked_mask) & refreshed_mask)[0]
        if len(untracked_hit) > 0:
            t_desired[untracked_hit] = refresh_t_desired[untracked_hit]
            t_deadline[untracked_hit] = refresh_t_deadline[untracked_hit]
            tracked_mask[untracked_hit] = True

    @staticmethod
    def _apply_search_refresh(
        t_desired,
        t_deadline,
        tracked_mask,
        active_mask,
        az_bin,
        el_bin,
        refreshed_sectors,
        refresh_t_desired,
        refresh_t_deadline,
        refresh_tracked=False,
        refresh_gain=1.0,
    ):
        if refreshed_sectors is None or len(refreshed_sectors) == 0:
            return
        az = np.clip(np.round(az_bin.astype(np.float32) * 29.0).astype(np.int32), 0, 29)
        el = np.clip(np.round(el_bin.astype(np.float32) * 9.0).astype(np.int32), 0, 9)
        sectors = el * 30 + az
        refreshed_mask = np.isin(sectors, refreshed_sectors)
        untracked_hit = np.where(active_mask & (~tracked_mask) & refreshed_mask)[0]
        if len(untracked_hit) > 0:
            t_desired[untracked_hit] = refresh_t_desired[untracked_hit]
            t_deadline[untracked_hit] = refresh_t_deadline[untracked_hit]
            tracked_mask[untracked_hit] = True

        if refresh_tracked and refresh_gain > 0.0:
            tracked_hit = np.where(active_mask & tracked_mask & refreshed_mask)[0]
            if len(tracked_hit) > 0:
                gain = np.clip(float(refresh_gain), 0.0, 1.0)
                t_desired[tracked_hit] = t_desired[tracked_hit] + gain * (
                    refresh_t_desired[tracked_hit] - t_desired[tracked_hit]
                )
                t_deadline[tracked_hit] = t_deadline[tracked_hit] + gain * (
                    refresh_t_deadline[tracked_hit] - t_deadline[tracked_hit]
                )
    
    def _backprop(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward += node.edge_reward
            node = node.parent
    
    def _est_action(self, node):
        tracked_active = node.active_mask & node.tracked_mask
        if not np.any(tracked_active):
            return self.SEARCH_ACTION
        t_desired_masked = np.where(tracked_active, node.t_desired, np.inf)
        return int(np.argmin(t_desired_masked)) + 1

