"""
Policy-Only Transformer MCTS Planner.

Uses the multiplicative PUCT formula:
    U(s,a) ∝ Q(s,a) * P(s,a) / (1 + N(s,a))

Where:
- Q(s,a) = Value from full simulation rollouts (not value head)
- P(s,a) = Prior probability from policy head
- N(s,a) = Visit count

This is the policy-only variant where value estimation comes from
full simulation rollouts, not a neural network value head.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .planner import Planner
from .mcts import Node, MCTSPlanner


class MinMaxStats:
    """Track running value bounds within a search tree."""

    def __init__(self):
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, value: float):
        self.minimum = min(self.minimum, float(value))
        self.maximum = max(self.maximum, float(value))

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (float(value) - self.minimum) / (self.maximum - self.minimum)
        return float(value)


class _TaskOnlyLinearHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, task_outputs, cls_output, pooled_tasks):
        return self.proj(task_outputs).squeeze(-1)


class _TaskOnlyMLPHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, task_outputs, cls_output, pooled_tasks):
        return self.net(task_outputs).squeeze(-1)


class _TaskCLSAddMLPHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, task_outputs, cls_output, pooled_tasks):
        cls = cls_output.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        return self.net(task_outputs + cls).squeeze(-1)


class _TaskCLSConcatMLPHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, task_outputs, cls_output, pooled_tasks):
        cls = cls_output.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        x = torch.cat([task_outputs, cls], dim=-1)
        return self.net(x).squeeze(-1)


class _TaskGlobalConcatMLPHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, task_outputs, cls_output, pooled_tasks):
        cls = cls_output.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        pooled = pooled_tasks.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        x = torch.cat([task_outputs, cls, pooled], dim=-1)
        return self.net(x).squeeze(-1)


def build_policy_head(policy_head_type: str, d_model: int):
    head_type = str(policy_head_type).lower()
    if head_type == "linear":
        return _TaskOnlyLinearHead(d_model)
    if head_type == "task_mlp":
        return _TaskOnlyMLPHead(d_model)
    if head_type == "cls_add_mlp":
        return _TaskCLSAddMLPHead(d_model)
    if head_type == "cls_concat_mlp":
        return _TaskCLSConcatMLPHead(d_model)
    if head_type == "global_concat_mlp":
        return _TaskGlobalConcatMLPHead(d_model)
    raise ValueError(f"Unknown policy_head_type: {policy_head_type}")


class PolicyOnlyTransformer(nn.Module):
    """Transformer model that only outputs policy priors."""
    
    def __init__(self, num_tasks=501, num_features=8, d_model=128, nhead=4, nlayers=2, policy_head_type="linear"):
        super().__init__()
        self.num_tasks = num_tasks
        self.policy_head_type = str(policy_head_type).lower()
        
        # Feature embedding
        self.input_proj = nn.Linear(num_features, d_model)
        
        # CLS token for global context
        self.cls_token = nn.Parameter(torch.randn(d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.policy_head = build_policy_head(self.policy_head_type, d_model)
    
    def forward(self, x):
        """Forward pass returning only policy logits."""
        batch_size = x.shape[0]
        # Token 0 is SEARCH and is always valid. Track tokens are padded/inactive
        # when their active-flag feature is 0.
        token_is_active = x[:, :, 4] > 0.5
        token_is_active[:, 0] = True
        
        # Project features
        embeddings = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        src_key_padding_mask = ~torch.cat([cls_valid, token_is_active], dim=1)
        
        # Transformer
        output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        task_outputs = output[:, 1:, :]
        cls_output = output[:, 0, :]
        active_f = token_is_active.unsqueeze(-1).float()
        pooled_tasks = (task_outputs * active_f).sum(dim=1) / active_f.sum(dim=1).clamp_min(1.0)
        logits = self.policy_head(task_outputs, cls_output, pooled_tasks)
        # Never allocate probability mass to inactive/padded track tokens.
        logits = logits.masked_fill(~token_is_active, -1e9)
        
        return logits
    
    def predict(self, x):
        """Get policy priors from observation (for MCTS)."""
        x = np.array(x)
        if len(x.shape) == 2:
            x = torch.from_numpy(x).float().unsqueeze(0)
        elif len(x.shape) == 3:
            x = torch.from_numpy(x).float()
            
        if TORCH_AVAILABLE:
            device = next(self.parameters()).device
            x = x.to(device)
            
            with torch.inference_mode():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
        return None


class PolicyValueTransformer(nn.Module):
    """Transformer model with policy and scalar value heads."""

    def __init__(
        self,
        num_tasks=501,
        num_features=8,
        d_model=128,
        nhead=4,
        nlayers=2,
        policy_head_type="linear",
        value_head_use_tanh=True,
        q_head_use_tanh=True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.policy_head_type = str(policy_head_type).lower()
        self.value_head_use_tanh = bool(value_head_use_tanh)
        self.q_head_use_tanh = bool(q_head_use_tanh)

        self.input_proj = nn.Linear(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.policy_head = build_policy_head(self.policy_head_type, d_model)
        # Value quality was the weak point in earlier runs. Pooling the active
        # task embeddings gives the value head direct access to scene-level load
        # and urgency, rather than relying on CLS alone.
        self.value_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.q_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        token_is_active = x[:, :, 4] > 0.5
        token_is_active[:, 0] = True

        embeddings = self.input_proj(x)
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        src_key_padding_mask = ~torch.cat([cls_valid, token_is_active], dim=1)

        output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)

        task_outputs = output[:, 1:, :]
        cls_output = output[:, 0, :]
        active_f = token_is_active.unsqueeze(-1).float()
        pooled_tasks = (task_outputs * active_f).sum(dim=1) / active_f.sum(dim=1).clamp_min(1.0)
        logits = self.policy_head(task_outputs, cls_output, pooled_tasks)
        logits = logits.masked_fill(~token_is_active, -1e9)
        value_input = torch.cat([cls_output, pooled_tasks], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        if self.value_head_use_tanh:
            value = torch.tanh(value)
        cls = cls_output.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        pooled = pooled_tasks.unsqueeze(1).expand(-1, task_outputs.shape[1], -1)
        q_input = torch.cat([task_outputs, cls, pooled], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        if self.q_head_use_tanh:
            q_values = torch.tanh(q_values)
        q_values = q_values.masked_fill(~token_is_active, 0.0)
        return logits, value, q_values

    def predict(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = torch.from_numpy(x).float().unsqueeze(0)
        elif len(x.shape) == 3:
            x = torch.from_numpy(x).float()

        if TORCH_AVAILABLE:
            device = next(self.parameters()).device
            x = x.to(device)

            with torch.inference_mode():
                logits, value, q_values = self.forward(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs, value.cpu().numpy(), q_values.cpu().numpy()
        return None, None, None


class PolicyOnlyMCTSPlanner(Planner):
    """
    MCTS Planner with multiplicative PUCT utility function.
    
    Utility: U(s,a) = Q(s,a) * P(s,a) / (1 + N(s,a))
    
    - Policy priors P(s,a) come from the neural network
    - Value Q(s,a) comes from full simulation rollouts
    """
    
    def __init__(
        self,
        checkpoint_path=None,
        model=None,
        max_trackers=500,
        num_rollouts=50,
        exploration_constant=2.0,
        device='cuda',
        ucb_mode="additive",
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
        sector_staleness_weight=0.0,
        sector_target_cycle_ms=-1.0,
        enable_track_beam_scan=False,
        revisit_time_scale=1.0,
        search_delay_mode=0,
        search_debt_penalty_weight=0.0001,
        search_debt_tau_ms=10.0,
        search_delay_penalty_cap=-1.0,
        search_delay_overdue_gate_threshold=-1.0,
        search_delay_overdue_gate_min_scale=0.0,
        search_delay_gate_metric="overdue_frac",
        search_delay_gate_delay_ms=1000.0,
        search_delay_gate_rescue_debt_ms=-1.0,
        search_delay_gate_rescue_min_scale=0.0,
        search_prior_overdue_gate_threshold=-1.0,
        search_prior_overdue_gate_min_scale=1.0,
        force_search_debt_ms=-1.0,
        use_value_head=False,
        value_scale=20.0,
        leaf_value_mix=1.0,
        use_q_head=False,
        q_scale=20.0,
        q_utility_weight=0.0,
        training_mode=False,
        root_dirichlet_alpha=0.30,
        root_dirichlet_eps=0.25,
        expand_top_k=32,
        action_select_mode="value",
        search_prior_scale=1.0,
        prior_mix=1.0,
        label_mode="visits",
        label_temperature=2.0,
        rollout_candidate_cap=96,
        root_search_strategy="puct",
        gumbel_considered_actions=8,
        selection_q_mode="raw",
        value_utility_weight=0.0,
        completed_q_weight=1.0,
        completed_q_transform="minmax",
        completed_q_expand_all_root=False,
        search_macro_len=1,
        search_macro_min_margin=0.0,
        value_head_use_tanh=True,
        q_head_use_tanh=True,
        policy_head_type="linear",
        pb_c_base=19652.0,
        pb_c_init=1.25,
        puct_parent_visits_power=0.5,
        penalize_hidden_targets=False,
        simulation_window_ms=200.0,
    ):
        super().__init__(max_trackers)
        self.max_trackers = max_trackers
        self.num_tasks = max_trackers + 1
        self.SEARCH_ACTION = 0
        self.num_rollouts = num_rollouts
        self.ucb_mode = ucb_mode
        self.training_mode = bool(training_mode)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_dirichlet_eps = float(root_dirichlet_eps)
        self.expand_top_k = int(expand_top_k)
        self.action_select_mode = str(action_select_mode).lower()
        self.search_prior_scale = float(search_prior_scale)
        self.prior_mix = float(prior_mix)
        self.label_mode = str(label_mode).lower()
        self.root_search_strategy = str(root_search_strategy).lower()
        self.gumbel_considered_actions = int(max(2, gumbel_considered_actions))
        self.selection_q_mode = str(selection_q_mode).lower()
        self.value_utility_weight = float(value_utility_weight)
        self.completed_q_weight = float(completed_q_weight)
        self.completed_q_transform = str(completed_q_transform).lower()
        self.completed_q_expand_all_root = bool(completed_q_expand_all_root)
        self.search_macro_len = int(max(1, search_macro_len))
        self.search_macro_min_margin = float(search_macro_min_margin)
        self.value_head_use_tanh = bool(value_head_use_tanh)
        self.q_head_use_tanh = bool(q_head_use_tanh)
        self.policy_head_type = str(policy_head_type).lower()
        self.pb_c_base = float(pb_c_base)
        self.pb_c_init = float(pb_c_init)
        self.puct_parent_visits_power = float(puct_parent_visits_power)
        self.search_prior_overdue_gate_threshold = float(search_prior_overdue_gate_threshold)
        self.search_prior_overdue_gate_min_scale = float(search_prior_overdue_gate_min_scale)
        self.force_search_debt_ms = float(force_search_debt_ms)
        self.use_value_head = bool(use_value_head)
        self.value_scale = float(max(1e-6, value_scale))
        self.leaf_value_mix = float(np.clip(leaf_value_mix, 0.0, 1.0))
        self.use_q_head = bool(use_q_head)
        self.q_scale = float(max(1e-6, q_scale))
        self.q_utility_weight = float(q_utility_weight)
        self.label_temperature = float(label_temperature)
        
        # Fallback pure MCTS (for simulation)
        self.pure_mcts = MCTSPlanner(
            max_trackers=max_trackers,
            num_rollouts=num_rollouts,
            exploration_constant=exploration_constant,
            tardiness_mode=tardiness_mode,
            local_tardiness_weight=local_tardiness_weight,
            global_tardiness_weight=global_tardiness_weight,
            normalize_delay_penalty=normalize_delay_penalty,
            global_aggregation=global_aggregation,
            tardiness_accounting=tardiness_accounting,
            settle_rollout_debt=settle_rollout_debt,
            enable_search_refresh_tracked=enable_search_refresh_tracked,
            search_refresh_gain=search_refresh_gain,
            search_action_reward=search_action_reward,
            track_update_reward=track_update_reward,
            track_loss_penalty=track_loss_penalty,
            track_urgency_bonus_weight=track_urgency_bonus_weight,
            track_uncertainty_bonus_weight=track_uncertainty_bonus_weight,
            sector_staleness_weight=sector_staleness_weight,
            sector_target_cycle_ms=sector_target_cycle_ms,
            enable_track_beam_scan=enable_track_beam_scan,
            revisit_time_scale=revisit_time_scale,
            search_delay_mode=search_delay_mode,
            search_debt_penalty_weight=search_debt_penalty_weight,
            search_debt_tau_ms=search_debt_tau_ms,
            search_delay_penalty_cap=search_delay_penalty_cap,
            search_delay_overdue_gate_threshold=search_delay_overdue_gate_threshold,
            search_delay_overdue_gate_min_scale=search_delay_overdue_gate_min_scale,
            search_delay_gate_metric=search_delay_gate_metric,
            search_delay_gate_delay_ms=search_delay_gate_delay_ms,
            search_delay_gate_rescue_debt_ms=search_delay_gate_rescue_debt_ms,
            search_delay_gate_rescue_min_scale=search_delay_gate_rescue_min_scale,
            penalize_hidden_targets=penalize_hidden_targets,
            rollout_candidate_cap=rollout_candidate_cap,
            simulation_window_ms=simulation_window_ms,
        )
        
        # Load/Create model
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if self.model is None and TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            model_cls = PolicyValueTransformer if self.use_value_head else PolicyOnlyTransformer
            model_kwargs = dict(num_tasks=self.num_tasks, policy_head_type=self.policy_head_type)
            if model_cls is PolicyValueTransformer:
                model_kwargs.update(
                    value_head_use_tanh=self.value_head_use_tanh,
                    q_head_use_tanh=self.q_head_use_tanh,
                )
            self.model = model_cls(**model_kwargs).to(self.device)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Filter compatible weights
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            self.model.load_state_dict(model_state)
            self.model.eval()
            print(f"Loaded {'PolicyValueTransformer' if self.use_value_head else 'PolicyOnlyTransformer'} from {checkpoint_path}")

    @staticmethod
    def _edge_q(child):
        return float(child.edge_reward + child.total_reward / max(1, child.visits))

    def _new_search_minmax(self):
        self._search_minmax = MinMaxStats()

    def _update_search_minmax(self, node):
        cursor = node
        while cursor is not None and cursor.parent is not None:
            self._search_minmax.update(self._edge_q(cursor))
            cursor = cursor.parent

    def _model_predict(self, adapted):
        predicted = self.model.predict(adapted)
        if isinstance(predicted, tuple):
            probs = predicted[0]
            value = predicted[1]
            value = float(np.asarray(value).reshape(-1)[0]) * self.value_scale
            q_values = np.zeros((self.num_tasks,), dtype=np.float32)
            if len(predicted) >= 3 and predicted[2] is not None:
                q_values = np.asarray(predicted[2], dtype=np.float32).reshape(-1, self.num_tasks)[0] * self.q_scale
            return probs[0], value, q_values
        return predicted[0], 0.0, np.zeros((self.num_tasks,), dtype=np.float32)

    def _model_predict_values_batch(self, adapted_batch):
        predicted = self.model.predict(adapted_batch)
        if isinstance(predicted, tuple):
            value = predicted[1]
            return np.asarray(value, dtype=np.float32).reshape(-1) * self.value_scale
        return np.zeros((len(adapted_batch),), dtype=np.float32)

    def _ensure_child_values(self, node):
        if not self.use_value_head or self.value_utility_weight == 0.0 or not node.children:
            return
        missing = [child for child in node.children if getattr(child, "nn_value", None) is None]
        if not missing:
            return
        batch = []
        for child in missing:
            batch.append(self.adapt_obs({
                't_desired': child.t_desired,
                't_deadline': child.t_deadline,
                't_dwell': child.t_dwell,
                'priority': child.priority,
                'active_mask': child.active_mask,
                'tracked_mask': child.tracked_mask,
                'grid': child.grid,
                'az_bin': child.az_bin,
                'el_bin': child.el_bin,
                'search_debt_ms': child.search_debt_ms,
            })[0])
        vals = self._model_predict_values_batch(np.asarray(batch, dtype=np.float32))
        for child, val in zip(missing, vals):
            child.nn_value = float(val)

    def _pick_exec_child(self, children):
        """Choose execution action from expanded children."""
        if not children:
            return None
        if self.action_select_mode == "visits":
            return max(children, key=lambda c: c.visits)
        if self.action_select_mode == "completed_policy":
            probs = self._completed_policy_probs(children[0].parent, restrict_to_children=True)
            if probs is not None:
                visited = [c for c in children if c.visits > 0]
                pool = visited if visited else children
                return max(
                    pool,
                    key=lambda c: (
                        float(probs[c.action]),
                        c.visits,
                        c.edge_reward + c.total_reward / max(1, c.visits),
                    ),
                )
        if self.action_select_mode == "q":
            return max(
                children,
                key=lambda c: (
                    float(getattr(c.parent, "nn_qvalues", np.zeros((self.num_tasks,), dtype=np.float32))[c.action]),
                    c.visits,
                ),
            )
        if self.action_select_mode == "posterior":
            parent = children[0].parent
            if parent is not None:
                priors = np.asarray(getattr(parent, "nn_priors", np.zeros((self.num_tasks,), dtype=np.float32)), dtype=np.float32)
                total_visits = max(1.0, float(sum(c.visits for c in children)))
                lam = self.pure_mcts.c / np.sqrt(total_visits)
                best_child = children[0]
                best_score = -np.inf
                for child in children:
                    q = float(child.edge_reward + child.total_reward / max(1, child.visits))
                    p = float(max(1e-12, priors[child.action]))
                    score = np.log(p) + q / max(1e-6, lam)
                    if score > best_score:
                        best_score = score
                        best_child = child
                return best_child
        # Default: choose by estimated value, tie-break by visits.
        # CRITICAL: never let unvisited children (Q = edge_reward only) beat
        # visited children whose Q may be negative due to global tardiness.
        # Restrict to visited children when any exist.
        visited = [c for c in children if c.visits > 0]
        pool = visited if visited else children
        return max(
            pool,
            key=lambda c: (
                c.edge_reward + c.total_reward / max(1, c.visits),
                c.visits,
            ),
        )

    def _should_expand_search_macro(self, node, best_child, children):
        if self.search_macro_len <= 1 or best_child is None or best_child.action != self.SEARCH_ACTION:
            return False
        if self.action_select_mode not in ("posterior", "completed_policy", "visits", "value"):
            return False
        search_score = None
        best_track_score = -np.inf
        if self.action_select_mode == "posterior":
            priors = np.asarray(getattr(node, "nn_priors", np.zeros((self.num_tasks,), dtype=np.float32)), dtype=np.float32)
            total_visits = max(1.0, float(sum(c.visits for c in children)))
            lam = self.pure_mcts.c / np.sqrt(total_visits)
            for child in children:
                q = float(child.edge_reward + child.total_reward / max(1, child.visits))
                p = float(max(1e-12, priors[child.action]))
                score = np.log(p) + q / max(1e-6, lam)
                if child.action == self.SEARCH_ACTION:
                    search_score = score
                else:
                    best_track_score = max(best_track_score, score)
        elif self.action_select_mode == "completed_policy":
            probs = self._completed_policy_probs(node, restrict_to_children=True)
            if probs is None:
                return False
            for child in children:
                score = float(probs[child.action])
                if child.action == self.SEARCH_ACTION:
                    search_score = score
                else:
                    best_track_score = max(best_track_score, score)
        elif self.action_select_mode == "visits":
            for child in children:
                score = float(child.visits)
                if child.action == self.SEARCH_ACTION:
                    search_score = score
                else:
                    best_track_score = max(best_track_score, score)
        else:
            for child in children:
                score = float(child.edge_reward + child.total_reward / max(1, child.visits))
                if child.action == self.SEARCH_ACTION:
                    search_score = score
                else:
                    best_track_score = max(best_track_score, score)
        if search_score is None:
            return False
        margin = search_score - best_track_score if np.isfinite(best_track_score) else search_score
        return margin >= self.search_macro_min_margin
    
    def adapt_obs(self, obs):
        """Standardize observation to 8-feature radar format used in training."""
        adapted = np.zeros((1, self.num_tasks, 8), dtype=np.float32)

        grid = obs.get('grid', np.zeros((300,), dtype=np.float32))
        grid_min = float(np.min(grid))
        grid_neg_frac = float(np.mean(grid < 0.0))
        search_debt_ms = float(obs.get('search_debt_ms', 0.0))
        search_debt_norm = float(np.clip(search_debt_ms / 1000.0, 0.0, 10.0))

        active = obs['active_mask'].astype(bool)
        t_deadline = obs['t_deadline']
        tracked = obs.get('tracked_mask', active & (t_deadline > 0))
        tracked_active = active & tracked
        tracked_n = int(np.sum(tracked_active))
        tracked_count_norm = float(tracked_n / max(1, self.max_trackers))
        tracked_delays = np.maximum(0.0, -obs['t_desired'][tracked_active]) if tracked_n > 0 else np.zeros(0, dtype=np.float32)
        mean_tracked_delay_norm = float(np.clip(np.mean(tracked_delays) / 2000.0, 0.0, 10.0)) if tracked_n > 0 else 0.0
        overdue_frac = float(np.mean(obs['t_desired'][tracked_active] < 0.0)) if tracked_n > 0 else 0.0
        # Expose the same global control pressures that the rollout reward uses.
        global_tardiness_norm = (
            float(np.clip(np.sum(tracked_delays) / 20000.0, 0.0, 10.0))
            if tracked_n > 0 else 0.0
        )
        tracked_deadline_pressure = (
            np.maximum(0.0, 100.0 - obs['t_deadline'][tracked_active])
            if tracked_n > 0 else np.zeros(0, dtype=np.float32)
        )
        global_deadline_pressure_norm = (
            float(np.clip(np.sum(tracked_deadline_pressure) / 2000.0, 0.0, 10.0))
            if tracked_n > 0 else 0.0
        )
        search_penalty_norm = float(
            np.clip(self.pure_mcts._search_delay_penalty(search_debt_ms), 0.0, 10.0)
        )
        global_penalty_norm = float(
            np.clip(
                0.001 * (
                    self.pure_mcts.global_tardiness_weight * global_tardiness_norm
                    + self.pure_mcts.local_tardiness_weight * mean_tracked_delay_norm
                ),
                0.0,
                10.0,
            )
        )

        # SEARCH token gets coverage stats plus explicit global penalty context.
        adapted[0, 0, :] = [
            tracked_count_norm,
            grid_min,
            global_tardiness_norm,
            mean_tracked_delay_norm,
            overdue_frac,
            global_deadline_pressure_norm,
            search_penalty_norm,
            global_penalty_norm,
        ]

        az_bin = obs.get('az_bin', np.zeros_like(obs['t_desired'], dtype=np.float32))
        el_bin = obs.get('el_bin', np.zeros_like(obs['t_desired'], dtype=np.float32))
        az_idx = np.clip(np.round(az_bin * 29.0).astype(np.int32), 0, 29)
        el_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32), 0, 9)
        sector_idx = np.clip(el_idx * 30 + az_idx, 0, max(0, len(grid) - 1))
        sector_urgency = grid[sector_idx].astype(np.float32) if len(grid) > 0 else np.zeros_like(obs['t_desired'], dtype=np.float32)
        priority = obs.get('priority', np.zeros_like(obs['t_desired']))
        target_tardiness = np.maximum(0.0, -obs['t_desired']).astype(np.float32)
        local_penalty_norm = np.clip(
            0.001 * target_tardiness * (1.0 + 2.0 * priority) * self.pure_mcts.local_tardiness_weight,
            0.0,
            10.0,
        ).astype(np.float32)

        # Rows 1..N: TRACK actions with explicit broadcast global penalty context.
        n = len(obs['t_desired'])
        adapted[0, 1:n+1, 0] = obs['t_desired']
        adapted[0, 1:n+1, 1] = obs['t_deadline']
        adapted[0, 1:n+1, 2] = obs['t_dwell']
        adapted[0, 1:n+1, 3] = priority
        adapted[0, 1:n+1, 4] = (active & tracked).astype(np.float32)
        adapted[0, 1:n+1, 5] = sector_urgency
        adapted[0, 1:n+1, 6] = local_penalty_norm
        adapted[0, 1:n+1, 7] = global_penalty_norm + search_penalty_norm

        return adapted

    def _get_node_priors(self, node, add_root_noise=False):
        """Cache calibrated priors per node to avoid repeated model calls."""
        if getattr(node, "nn_priors", None) is None:
            adapted = self.adapt_obs({
                't_desired': node.t_desired,
                't_deadline': node.t_deadline,
                't_dwell': node.t_dwell,
                'priority': node.priority,
                'active_mask': node.active_mask,
                'tracked_mask': node.tracked_mask,
                'grid': node.grid,
                'az_bin': node.az_bin,
                'el_bin': node.el_bin,
                'search_debt_ms': node.search_debt_ms,
            })
            priors, value, q_values = self._model_predict(adapted)
            priors = self._calibrate_priors(priors, node=node)
            if self.prior_mix < 1.0:
                valid = node.get_valid_actions()
                if valid:
                    uniform = np.zeros_like(priors, dtype=np.float32)
                    uniform[np.asarray(valid, dtype=np.int32)] = 1.0 / float(len(valid))
                    priors = self.prior_mix * priors + (1.0 - self.prior_mix) * uniform
                    total = float(np.sum(priors))
                    if total > 0.0:
                        priors = priors / total
            # If all tracker slots are occupied, search cannot discover new targets.
            # Zero out the search prior so MCTS never explores it at full capacity,
            # forcing training labels to show 0% search → model learns this constraint.
            if int(np.sum(node.active_mask)) >= self.max_trackers:
                priors[0] = 0.0
                total = float(priors.sum())
                if total > 0.0:
                    priors = priors / total
            if add_root_noise:
                priors = self._apply_root_noise(priors, node)
            node.nn_priors = priors
            node.nn_value = value
            node.nn_qvalues = q_values
        return node.nn_priors

    def _build_root(self, obs):
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask'],
            grid=obs.get('grid', None),
            az_bin=obs.get('az_bin', None),
            el_bin=obs.get('el_bin', None),
            tracked_mask=obs.get('tracked_mask', obs['active_mask'] & (obs['t_deadline'] > 0)),
            search_debt_ms=obs.get('search_debt_ms', 0.0),
        )
        root.refresh_t_desired, root.refresh_t_deadline = Node._infer_refresh_timers(
            root.t_desired, root.t_deadline, root.priority, self.pure_mcts.revisit_time_scale
        )
        return root

    def _search_root(self, root, add_root_noise=True):
        """Run policy-guided MCTS from an already constructed root node."""
        root_priors = self._get_node_priors(root, add_root_noise=add_root_noise)
        if self.root_search_strategy == "gumbel":
            return self._search_root_gumbel(root, root_priors)
        root_top_k = None if self.completed_q_expand_all_root else self.expand_top_k

        for _ in range(self.num_rollouts):
            node = root
            priors = root_priors

            while node.expanded and node.children and not node.is_terminal():
                node = self._select_child(node, priors)
                priors = self._get_node_priors(node) if node.children else None

            reward = 0.0
            if not node.is_terminal():
                if not node.expanded:
                    priors = self._get_node_priors(node)
                    top_k = root_top_k if node is root else self.expand_top_k
                    self.pure_mcts._expand(node, priors=priors, top_k=top_k)
                if self.use_value_head:
                    value_reward = float(getattr(node, "nn_value", 0.0))
                    mix = self.leaf_value_mix
                    if mix >= 1.0 - 1e-6:
                        reward = value_reward
                    elif mix <= 1e-6:
                        reward = self.pure_mcts._simulate(node)
                    else:
                        rollout_reward = self.pure_mcts._simulate(node)
                        reward = (1.0 - mix) * rollout_reward + mix * value_reward
                else:
                    reward = self.pure_mcts._simulate(node)

            self.pure_mcts._backprop(node, reward)
            if self.selection_q_mode == "tree_minmax":
                self._update_search_minmax(node)
        return root

    def _run_simulation_from_child(self, child):
        node = child
        priors = self._get_node_priors(node) if node.children else None
        while node.expanded and node.children and not node.is_terminal():
            node = self._select_child(node, priors)
            priors = self._get_node_priors(node) if node.children else None

        reward = 0.0
        if not node.is_terminal():
            if not node.expanded:
                priors = self._get_node_priors(node)
                self.pure_mcts._expand(node, priors=priors, top_k=self.expand_top_k)
            if self.use_value_head:
                value_reward = float(getattr(node, "nn_value", 0.0))
                mix = self.leaf_value_mix
                if mix >= 1.0 - 1e-6:
                    reward = value_reward
                elif mix <= 1e-6:
                    reward = self.pure_mcts._simulate(node)
                else:
                    rollout_reward = self.pure_mcts._simulate(node)
                    reward = (1.0 - mix) * rollout_reward + mix * value_reward
            else:
                reward = self.pure_mcts._simulate(node)

        self.pure_mcts._backprop(node, reward)
        if self.selection_q_mode == "tree_minmax":
            self._update_search_minmax(node)

    @staticmethod
    def _sample_gumbel(size):
        u = np.random.uniform(low=1e-9, high=1.0 - 1e-9, size=size)
        return -np.log(-np.log(u))

    def _search_root_gumbel(self, root, root_priors):
        if not root.expanded:
            root_top_k = None if self.completed_q_expand_all_root else self.expand_top_k
            self.pure_mcts._expand(root, priors=root_priors, top_k=root_top_k)
        if not root.children:
            return root

        valid_children = list(root.children)
        candidate_count = min(
            len(valid_children),
            max(2, min(self.gumbel_considered_actions, self.num_rollouts)),
        )
        child_priors = np.asarray(
            [max(1e-12, float(root_priors[child.action])) for child in valid_children],
            dtype=np.float32,
        )
        gumbels = self._sample_gumbel(len(valid_children))
        gumbel_logits = np.log(child_priors) + gumbels
        order = np.argsort(gumbel_logits)[::-1]
        candidates = [valid_children[i] for i in order[:candidate_count]]
        gumbel_score = {valid_children[i].action: float(gumbel_logits[i]) for i in order[:candidate_count]}

        budget_remaining = int(self.num_rollouts)
        while budget_remaining > 0 and candidates:
            for child in list(candidates):
                if budget_remaining <= 0:
                    break
                self._run_simulation_from_child(child)
                budget_remaining -= 1

            if len(candidates) <= 1 or budget_remaining <= 0:
                break

            q_values = np.asarray(
                [
                    child.edge_reward + child.total_reward / max(1, child.visits)
                    for child in candidates
                ],
                dtype=np.float32,
            )
            q_min = float(np.min(q_values))
            q_span = float(np.max(q_values) - q_min)
            if q_span > 1e-6:
                q_norm = (q_values - q_min) / q_span
            else:
                q_norm = np.zeros_like(q_values)
            rank_scores = np.asarray(
                [gumbel_score[child.action] for child in candidates],
                dtype=np.float32,
            ) + q_norm
            keep = max(1, int(np.ceil(len(candidates) / 2.0)))
            keep_idx = np.argsort(rank_scores)[::-1][:keep]
            candidates = [candidates[i] for i in keep_idx]

        return root

    def _extract_plan(self, root, max_steps):
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                probs = self._get_node_priors(node)
                if self.action_select_mode == "q":
                    valid_actions = node.get_valid_actions()
                    q_values = np.asarray(
                        getattr(node, "nn_qvalues", np.zeros((self.num_tasks,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    if valid_actions:
                        action = max(valid_actions, key=lambda a: float(q_values[a]))
                    else:
                        action = self.SEARCH_ACTION
                    # Q selection needs the selected child state to continue the
                    # schedule. Expand all valid children here; otherwise the
                    # expander's prior top-k can omit the Q-best action.
                    self.pure_mcts._expand(node, priors=probs, top_k=None)
                    child = next((c for c in node.children if c.action == action), None)
                    if child is None:
                        break
                    plan.append(action)
                    node = child
                    if node.is_terminal():
                        break
                    continue
                for i in range(self.max_trackers):
                    if i >= len(node.active_mask) or not node.active_mask[i]:
                        probs[i+1] = 0
                action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                if action not in node.get_valid_actions():
                    action = self.SEARCH_ACTION
                # Do not use top_k=1 here. The low-level expander force-includes
                # SEARCH and the most-overdue track, so a one-child expansion can
                # replace the policy-selected action and truncate the extracted
                # concrete schedule to a single step. Three children preserve the
                # selected action plus both safety inclusions without exploding
                # extraction latency.
                self.pure_mcts._expand(node, priors=probs, top_k=3)
                child = next((c for c in node.children if c.action == action), None)
                if child is None:
                    break
                plan.append(action)
                node = child
                if node.is_terminal():
                    break
                continue

            best_child = self._pick_exec_child(node.children)
            if best_child is None:
                break
            if self._should_expand_search_macro(node, best_child, node.children):
                remaining = max_steps - len(plan)
                macro_len = min(self.search_macro_len, remaining)
                if macro_len > 1:
                    plan.extend([self.SEARCH_ACTION] * macro_len)
                    break
            plan.append(best_child.action)
            node = best_child
            if node.is_terminal():
                break
        return plan

    def _node_training_input(self, node):
        return self.adapt_obs({
            't_desired': node.t_desired,
            't_deadline': node.t_deadline,
            't_dwell': node.t_dwell,
            'priority': node.priority,
            'active_mask': node.active_mask,
            'tracked_mask': node.tracked_mask,
            'grid': node.grid,
            'az_bin': node.az_bin,
            'el_bin': node.el_bin,
            'search_debt_ms': node.search_debt_ms,
        })[0]

    def _extract_root_policy(self, root):
        policy = np.zeros(self.num_tasks, dtype=np.float32)
        if self.label_mode == "completed_q":
            probs = self._completed_policy_probs(root, restrict_to_children=False)
            if probs is not None:
                return probs
        total_visits = sum(c.visits for c in root.children)
        if total_visits > 0:
            for child in root.children:
                policy[child.action] = child.visits / total_visits
        return policy

    def _extract_root_q_targets(self, root):
        q_targets = np.zeros((self.num_tasks,), dtype=np.float32)
        q_mask = np.zeros((self.num_tasks,), dtype=np.float32)
        if root is None:
            return q_targets, q_mask
        for child in root.children:
            q_targets[child.action] = float(child.edge_reward + child.total_reward / max(1, child.visits))
            q_mask[child.action] = 1.0
        return q_targets, q_mask

    def _collect_training_examples(self, root, include_internal=True, visit_threshold=5):
        """
        Build training targets from the searched tree.

        Modes:
        - visits: softened visit-distribution labels
        - best_path: one-hot best-child labels along the executed path
        """
        examples = []
        if root is None:
            return examples

        best_path_nodes = []
        best_path_ids = set()
        cursor = root
        while cursor is not None:
            best_path_nodes.append(cursor)
            best_path_ids.add(id(cursor))
            if not cursor.children:
                break
            cursor = self._pick_exec_child(cursor.children)

        if self.label_mode == "best_path":
            for node in best_path_nodes:
                if not node.children:
                    continue
                best_child = self._pick_exec_child(node.children)
                if best_child is None:
                    continue
                label = np.zeros((self.num_tasks,), dtype=np.float32)
                label[best_child.action] = 1.0
                examples.append((self._node_training_input(node), label))
            return examples

        stack = [root]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.children))

            if not node.children:
                continue

            include = node is root
            if include_internal and not include:
                include = id(node) in best_path_ids or node.visits >= int(visit_threshold)
            if not include:
                continue

            # Apply temperature to visit counts before normalizing.
            # T > 1 softens the distribution (reduces overfitting to noisy
            # single-simulation Q estimates in early training).
            label = np.zeros((self.num_tasks,), dtype=np.float32)
            if self.label_mode == "completed_q":
                probs = self._completed_policy_probs(node, restrict_to_children=False)
                if probs is None:
                    continue
                label[:] = probs
            else:
                visit_counts = np.array([c.visits for c in node.children], dtype=np.float32)
                if self.label_temperature != 1.0:
                    visit_counts = visit_counts ** (1.0 / self.label_temperature)
                total_visits = float(np.sum(visit_counts))
                if total_visits == 0:
                    continue
                for i, child in enumerate(node.children):
                    label[child.action] = visit_counts[i] / total_visits
            examples.append((self._node_training_input(node), label))

        return examples

    def _apply_root_noise(self, priors, root):
        if not self.training_mode or self.root_dirichlet_eps <= 0.0:
            return priors
        valid = root.get_valid_actions()
        if len(valid) <= 1:
            return priors
        alpha = max(1e-4, self.root_dirichlet_alpha)
        noise = np.random.dirichlet(np.full(len(valid), alpha, dtype=np.float32))
        mixed = priors.copy()
        for i, action in enumerate(valid):
            mixed[action] = (1.0 - self.root_dirichlet_eps) * priors[action] + self.root_dirichlet_eps * noise[i]
        mixed = mixed / (np.sum(mixed) + 1e-12)
        return mixed

    def _search_prior_gate_scale(self, node):
        threshold = self.search_prior_overdue_gate_threshold
        if threshold <= 0.0:
            return 1.0
        scale = self.pure_mcts._search_delay_gate_scale(
            node.t_desired,
            node.active_mask,
            node.tracked_mask,
        )
        return float(max(self.search_prior_overdue_gate_min_scale, scale))

    def _calibrate_priors(self, priors, node=None):
        if priors is None:
            return priors
        search_scale = float(self.search_prior_scale)
        if node is not None:
            search_scale *= self._search_prior_gate_scale(node)
        if search_scale == 1.0:
            return priors
        scaled = np.array(priors, copy=True, dtype=np.float32)
        scaled[0] *= search_scale
        denom = float(np.sum(scaled))
        if denom <= 1e-12:
            return priors
        return scaled / denom
    
    def _select_child(self, node, priors):
        """
        Child selection for policy-only MCTS.

        Modes:
        - additive (default): normalized-Q + PUCT exploration
        - multiplicative: legacy Q * P / (1 + N)
        """
        if not node.children:
            return None
        if self.selection_q_mode in ("q_plus_value", "q_plus_value_norm"):
            self._ensure_child_values(node)

        # Ensure every expanded child is visited at least once before the
        # policy prior starts dominating selection. Without this, the first
        # appended child can monopolize early rollouts under tie conditions.
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            if priors is not None:
                max_prior = max(float(priors[child.action]) for child in unvisited)
                top = [child for child in unvisited if float(priors[child.action]) == max_prior]
            else:
                max_prior = max(float(child.prior_prob) for child in unvisited)
                top = [child for child in unvisited if float(child.prior_prob) == max_prior]
            return top[np.random.randint(len(top))]

        best_score = -np.inf
        best_child = node.children[0]
        raw_q_values = [self._edge_q(child) for child in node.children]
        value_candidates = None
        completed_policy_probs = None
        if self.selection_q_mode in ("local_minmax", "muzero_local"):
            q_min = float(min(raw_q_values))
            q_max = float(max(raw_q_values))
            q_span = q_max - q_min
            q_values = [((q - q_min) / q_span) if q_span > 1e-6 else q for q in raw_q_values]
        elif self.selection_q_mode in ("tree_minmax", "muzero_tree"):
            q_values = [self._search_minmax.normalize(q) for q in raw_q_values]
        elif self.selection_q_mode in ("q_plus_value", "q_plus_value_norm"):
            q_values = raw_q_values
            value_candidates = [float(getattr(child, "nn_value", 0.0)) for child in node.children]
            if self.selection_q_mode == "q_plus_value_norm":
                vmin = float(min(value_candidates))
                vmax = float(max(value_candidates))
                vspan = vmax - vmin
                value_candidates = [((v - vmin) / vspan) if vspan > 1e-6 else v for v in value_candidates]
        elif self.selection_q_mode == "completed_q":
            q_values = raw_q_values
            completed_policy_probs = self._completed_policy_probs(node, priors=priors, restrict_to_children=True)
        else:
            q_values = raw_q_values

        total_child_visits = float(sum(child.visits for child in node.children))
        for idx, child in enumerate(node.children):
            q_value = q_values[idx]
            p_value = float(priors[child.action]) if priors is not None else float(child.prior_prob)
            n_value = child.visits

            if self.ucb_mode == "multiplicative":
                score = (q_value + 1e-6) * p_value / (1 + n_value)
            elif self.selection_q_mode in ("muzero_local", "muzero_tree"):
                pb_c = np.log((node.visits + self.pb_c_base + 1.0) / self.pb_c_base) + self.pb_c_init
                pb_c *= np.sqrt(max(1, node.visits)) / (n_value + 1.0)
                prior_score = pb_c * p_value
                value_score = q_value if child.visits > 0 else 0.0
                score = value_score + prior_score
            elif self.selection_q_mode in ("q_plus_value", "q_plus_value_norm"):
                parent_visits = float(node.visits + 1)
                parent_scale = parent_visits ** self.puct_parent_visits_power
                explore = self.pure_mcts.c * p_value * parent_scale / (1 + n_value)
                score = q_value + self.value_utility_weight * value_candidates[idx] + explore
            elif self.selection_q_mode == "completed_q":
                if completed_policy_probs is None:
                    score = q_value
                else:
                    score = float(completed_policy_probs[child.action]) - float(n_value) / (1.0 + total_child_visits)
            else:
                # PUCT formula aligned with MCTSPlanner._ucb_select:
                # Q + C * P * sqrt(parent_N + 1) / (1 + child_N)
                explore = self.pure_mcts.c * p_value * np.sqrt(node.visits + 1) / (1 + n_value)
                score = q_value + explore
                if self.use_q_head and self.q_utility_weight != 0.0:
                    score += self.q_utility_weight * float(getattr(node, "nn_qvalues", np.zeros((self.num_tasks,), dtype=np.float32))[child.action])

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _completed_q_transform_values(self, values):
        vals = np.asarray(values, dtype=np.float32)
        mode = self.completed_q_transform
        if mode == "raw":
            return vals
        if mode == "zscore":
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            return (vals - mean) / max(std, 1e-6)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        span = vmax - vmin
        if span <= 1e-6:
            return np.zeros_like(vals)
        return (vals - vmin) / span

    def _completed_policy_probs(self, node, priors=None, restrict_to_children=False):
        if node is None:
            return None
        if priors is None:
            priors = getattr(node, "nn_priors", None)
            if priors is None:
                priors = self._get_node_priors(node)
        priors = np.asarray(priors, dtype=np.float32)
        if restrict_to_children:
            actions = [child.action for child in node.children]
            if not actions:
                return None
        else:
            actions = list(node.get_valid_actions())
            if not actions:
                return None
        child_map = {child.action: child for child in node.children}
        raw_value = float(getattr(node, "nn_value", 0.0)) if self.use_value_head else 0.0
        visit_counts = []
        qvalues = []
        for action in actions:
            child = child_map.get(action)
            if child is not None and child.visits > 0:
                visit_counts.append(float(child.visits))
                qvalues.append(self._edge_q(child))
            else:
                visit_counts.append(0.0)
                qvalues.append(0.0)
        visit_counts = np.asarray(visit_counts, dtype=np.float32)
        qvalues = np.asarray(qvalues, dtype=np.float32)
        prior_subset = np.clip(priors[np.asarray(actions, dtype=np.int32)], 1e-12, 1.0).astype(np.float32)
        visited_mask = visit_counts > 0
        if np.any(visited_mask):
            prior_mass = float(np.sum(prior_subset[visited_mask]))
            if prior_mass > 1e-12:
                weighted_q = float(np.sum((prior_subset[visited_mask] / prior_mass) * qvalues[visited_mask]))
            else:
                weighted_q = float(np.mean(qvalues[visited_mask]))
            mixed_value = float((raw_value + float(np.sum(visit_counts)) * weighted_q) / (1.0 + float(np.sum(visit_counts))))
        else:
            mixed_value = raw_value
        completed_q = np.where(visited_mask, qvalues, mixed_value).astype(np.float32)
        transformed_q = self._completed_q_transform_values(completed_q)
        max_visit = float(np.max(visit_counts)) if len(visit_counts) else 0.0
        q_term = self.completed_q_weight * (50.0 + max_visit) * transformed_q
        logits = np.log(prior_subset) + q_term
        logits = logits - np.max(logits)
        probs_local = np.exp(logits)
        probs_local = probs_local / np.clip(np.sum(probs_local), 1e-12, None)
        probs = np.zeros((self.num_tasks,), dtype=np.float32)
        probs[np.asarray(actions, dtype=np.int32)] = probs_local.astype(np.float32)
        return probs

    def _force_search_policy(self):
        policy = np.zeros(self.num_tasks, dtype=np.float32)
        policy[self.SEARCH_ACTION] = 1.0
        return policy

    def _should_force_search(self, obs):
        if self.force_search_debt_ms <= 0.0:
            return False
        return float(obs.get('search_debt_ms', 0.0)) >= self.force_search_debt_ms
    
    def plan(self, obs, budget_ms=200):
        """Generate action plan using policy-guided MCTS with rollout values."""
        if self._should_force_search(obs):
            return [self.SEARCH_ACTION]
        if self.model is None:
            return self.pure_mcts.plan(obs, budget_ms=budget_ms)
        
        max_steps = int(budget_ms / 10.0) + 2
        root = self._build_root(obs)
        self._new_search_minmax()
        self._search_root(root, add_root_noise=True)
        return self._extract_plan(root, max_steps)
    
    def plan_with_policy(self, obs, budget_ms=200):
        """
        Generate action plan AND policy target (visit counts) for training.
        
        Returns:
            plan: List[int] - action sequence
            policy: np.ndarray (num_tasks,) - normalized visit counts at root
        """
        if self._should_force_search(obs):
            return [self.SEARCH_ACTION], self._force_search_policy()
        if self.model is None:
            plan = self.pure_mcts.plan(obs, budget_ms=budget_ms)
            return plan, None
        
        max_steps = int(budget_ms / 10.0) + 2
        root = self._build_root(obs)
        self._new_search_minmax()
        self._search_root(root, add_root_noise=True)
        policy = self._extract_root_policy(root)
        plan = self._extract_plan(root, max_steps)
        return plan, policy

    def plan_with_targets(self, obs, budget_ms=200):
        if self._should_force_search(obs):
            zeros = np.zeros((self.num_tasks,), dtype=np.float32)
            return [self.SEARCH_ACTION], self._force_search_policy(), zeros, zeros
        if self.model is None:
            plan = self.pure_mcts.plan(obs, budget_ms=budget_ms)
            zeros = np.zeros((self.num_tasks,), dtype=np.float32)
            return plan, None, zeros, zeros

        max_steps = int(budget_ms / 10.0) + 2
        root = self._build_root(obs)
        self._new_search_minmax()
        self._search_root(root, add_root_noise=True)
        policy = self._extract_root_policy(root)
        q_targets, q_mask = self._extract_root_q_targets(root)
        plan = self._extract_plan(root, max_steps)
        return plan, policy, q_targets, q_mask

    def plan_with_training_examples(self, obs, budget_ms=200, include_internal=True, visit_threshold=5):
        """
        Run policy-guided MCTS and return executable plan plus one-hot
        training examples derived from the searched tree.
        """
        if self._should_force_search(obs):
            label = self._force_search_policy()
            return [self.SEARCH_ACTION], [(self._node_training_input(self._build_root(obs)), label)]
        if self.model is None:
            plan = self.pure_mcts.plan(obs, budget_ms=budget_ms)
            return plan, []

        max_steps = int(budget_ms / 10.0) + 2
        root = self._build_root(obs)
        self._new_search_minmax()
        self._search_root(root, add_root_noise=True)
        plan = self._extract_plan(root, max_steps)
        examples = self._collect_training_examples(
            root,
            include_internal=include_internal,
            visit_threshold=visit_threshold,
        )
        return plan, examples
