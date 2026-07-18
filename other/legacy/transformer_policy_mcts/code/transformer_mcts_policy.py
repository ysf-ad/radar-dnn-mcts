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


class PolicyOnlyTransformer(nn.Module):
    """
    Transformer model that only outputs policy priors.
    Value estimation is done via full simulation rollouts.
    """
    
    def __init__(self, num_tasks=501, num_features=8, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.num_tasks = num_tasks
        
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
        
        # Policy head only (no value head)
        self.policy_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """Forward pass returning only policy logits."""
        batch_size = x.shape[0]
        
        # Project features
        embeddings = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Transformer
        output = self.transformer(embeddings)
        
        # Policy: logits for each task
        task_outputs = output[:, 1:, :]
        logits = self.policy_head(task_outputs).squeeze(-1)
        
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
            
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
        return None


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
        training_mode=False,
        root_dirichlet_alpha=0.30,
        root_dirichlet_eps=0.25,
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
        )
        
        # Load/Create model
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if self.model is None and TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            self.model = PolicyOnlyTransformer(num_tasks=self.num_tasks).to(self.device)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Filter compatible weights
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            self.model.load_state_dict(model_state)
            self.model.eval()
            print(f"Loaded PolicyOnlyTransformer from {checkpoint_path}")
    
    def adapt_obs(self, obs):
        """Standardize observation to 8-feature radar format used in training."""
        adapted = np.zeros((1, self.num_tasks, 8), dtype=np.float32)

        grid = obs.get('grid', np.zeros((300,), dtype=np.float32))
        grid_mean = float(np.mean(grid))
        grid_min = float(np.min(grid))
        grid_p10 = float(np.percentile(grid, 10))
        grid_neg_frac = float(np.mean(grid < 0.0))
        sensor_id = float(obs.get('sensor_id', 0))

        # SEARCH token gets explicit grid freshness stats.
        adapted[0, 0, :] = [
            0.0,       # t_desired
            grid_min,  # stale-floor proxy
            10.0,      # nominal search dwell
            0.0,       # priority
            1.0,       # active
            grid_p10,  # stale-quantile proxy
            sensor_id,
            grid_neg_frac,
        ]

        az_bin = obs.get('az_bin', np.zeros_like(obs['t_desired'], dtype=np.float32))
        el_bin = obs.get('el_bin', np.zeros_like(obs['t_desired'], dtype=np.float32))
        az_idx = np.clip(np.round(az_bin * 29.0).astype(np.int32), 0, 29)
        el_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32), 0, 9)
        sector_idx = np.clip(el_idx * 30 + az_idx, 0, max(0, len(grid) - 1))
        sector_urgency = grid[sector_idx].astype(np.float32) if len(grid) > 0 else np.full_like(obs['t_desired'], grid_mean, dtype=np.float32)

        # Rows 1..N: TRACK actions.
        adapted[0, 1:, 0] = obs['t_desired']
        adapted[0, 1:, 1] = obs['t_deadline']
        adapted[0, 1:, 2] = obs['t_dwell']
        adapted[0, 1:, 3] = obs.get('priority', np.zeros_like(obs['t_desired']))
        adapted[0, 1:, 4] = obs['active_mask'].astype(np.float32)
        adapted[0, 1:, 5] = sector_urgency
        adapted[0, 1:, 6] = sensor_id
        adapted[0, 1:, 7] = (~obs['active_mask']).astype(np.float32)

        return adapted

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
    
    def _select_child(self, node, priors):
        """
        Child selection for policy-only MCTS.

        Modes:
        - additive (default): normalized-Q + PUCT exploration
        - multiplicative: legacy Q * P / (1 + N)
        """
        best_score = -np.inf
        best_child = node.children[0] if node.children else None

        q_vals = np.array(
            [c.total_reward / max(1, c.visits) for c in node.children],
            dtype=np.float32,
        )
        q_min, q_max = float(np.min(q_vals)), float(np.max(q_vals))
        q_range = q_max - q_min

        for child in node.children:
            q_value = child.total_reward / max(1, child.visits)
            p_value = float(priors[child.action]) if priors is not None else float(child.prior_prob)
            n_value = child.visits

            if self.ucb_mode == "multiplicative":
                score = (q_value + 1e-6) * p_value / (1 + n_value)
            else:
                # Normalize exploit term per sibling set to avoid unstable scaling/negative collapse.
                if q_range > 1e-8:
                    q_norm = (q_value - q_min) / q_range
                else:
                    q_norm = 0.5
                explore = self.pure_mcts.c * p_value * np.sqrt(node.visits + 1) / (1 + n_value)
                score = q_norm + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
    
    def plan(self, obs, budget_ms=200):
        """Generate action plan using policy-guided MCTS with rollout values."""
        if self.model is None:
            return self.pure_mcts.plan(obs, budget_ms=budget_ms)
        
        max_steps = int(budget_ms / 10.0) + 2
        
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask'],
            grid=obs.get('grid', None),
            az_bin=obs.get('az_bin', None),
            el_bin=obs.get('el_bin', None),
        )
        
        # Get root priors from model
        adapted = self.adapt_obs(obs)
        root_priors = self.model.predict(adapted)[0]
        root_priors = self._apply_root_noise(root_priors, root)
        
        # MCTS loop: Policy priors from model, Value from rollouts
        for _ in range(self.num_rollouts):
            node = root
            priors = root_priors
            
            # 1. Select using multiplicative PUCT
            while node.expanded and node.children and not node.is_terminal():
                node = self._select_child(node, priors)
                # Get new priors for child node
                if node.children:
                    adapted = self.adapt_obs({
                        't_desired': node.t_desired,
                        't_deadline': node.t_deadline,
                        't_dwell': node.t_dwell,
                        'priority': node.priority,
                        'active_mask': node.active_mask,
                        'grid': node.grid,
                        'az_bin': node.az_bin,
                        'el_bin': node.el_bin,
                    })
                    priors = self.model.predict(adapted)[0]
            
            # 2. Expand with policy priors
            reward = 0.0
            if not node.is_terminal():
                if not node.expanded:
                    adapted = self.adapt_obs({
                        't_desired': node.t_desired,
                        't_deadline': node.t_deadline,
                        't_dwell': node.t_dwell,
                        'priority': node.priority,
                        'active_mask': node.active_mask,
                        'grid': node.grid,
                        'az_bin': node.az_bin,
                        'el_bin': node.el_bin,
                    })
                    priors = self.model.predict(adapted)[0]
                    self.pure_mcts._expand(node, priors=priors, top_k=10)
                
                # Value from FULL ROLLOUT (not value head)
                reward = self.pure_mcts._simulate(node)
            else:
                reward = 0.0

            # 3. Backprop
            self.pure_mcts._backprop(node, reward)
        
        # Extract plan
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                # Fallback to greedy from policy
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask,
                    'grid': node.grid,
                    'az_bin': node.az_bin,
                    'el_bin': node.el_bin,
                })
                probs = self.model.predict(adapted)[0]
                for i in range(self.max_trackers):
                    if i >= len(node.active_mask) or not node.active_mask[i]:
                        probs[i+1] = 0
                action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                plan.append(action)
                if action > 0:
                    node.active_mask[action-1] = False
                continue
                
            # Use visit count for action selection (AlphaZero-style).
            # More stable than mean-reward argmax when rollouts are noisy.
            best_child = max(node.children, key=lambda c: c.visits)
            plan.append(best_child.action)
            node = best_child
            if node.is_terminal(): break
            
        return plan
    
    def plan_with_policy(self, obs, budget_ms=200):
        """
        Generate action plan AND policy target (visit counts) for training.
        
        Returns:
            plan: List[int] - action sequence
            policy: np.ndarray (num_tasks,) - normalized visit counts at root
        """
        if self.model is None:
            plan = self.pure_mcts.plan(obs, budget_ms=budget_ms)
            return plan, None
        
        max_steps = int(budget_ms / 10.0) + 2
        
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask'],
            grid=obs.get('grid', None),
            az_bin=obs.get('az_bin', None),
            el_bin=obs.get('el_bin', None),
        )
        
        # Get root priors
        adapted = self.adapt_obs(obs)
        root_priors = self.model.predict(adapted)[0]
        root_priors = self._apply_root_noise(root_priors, root)
        
        # MCTS loop
        for _ in range(self.num_rollouts):
            node = root
            priors = root_priors
            
            while node.expanded and node.children and not node.is_terminal():
                node = self._select_child(node, priors)
                if node.children:
                    adapted = self.adapt_obs({
                        't_desired': node.t_desired,
                        't_deadline': node.t_deadline,
                        't_dwell': node.t_dwell,
                        'priority': node.priority,
                        'active_mask': node.active_mask,
                        'grid': node.grid,
                        'az_bin': node.az_bin,
                        'el_bin': node.el_bin,
                    })
                    priors = self.model.predict(adapted)[0]
            
            reward = 0.0
            if not node.is_terminal():
                if not node.expanded:
                    adapted = self.adapt_obs({
                        't_desired': node.t_desired,
                        't_deadline': node.t_deadline,
                        't_dwell': node.t_dwell,
                        'priority': node.priority,
                        'active_mask': node.active_mask,
                        'grid': node.grid,
                        'az_bin': node.az_bin,
                        'el_bin': node.el_bin,
                    })
                    priors = self.model.predict(adapted)[0]
                    self.pure_mcts._expand(node, priors=priors, top_k=10)
                
                reward = self.pure_mcts._simulate(node)
            else:
                reward = 0.0
            
            self.pure_mcts._backprop(node, reward)
        
        # Extract policy target from visit counts
        policy = np.zeros(self.num_tasks, dtype=np.float32)
        total_visits = sum(c.visits for c in root.children)
        if total_visits > 0:
            for child in root.children:
                policy[child.action] = child.visits / total_visits
        
        # Extract plan
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask,
                    'grid': node.grid,
                    'az_bin': node.az_bin,
                    'el_bin': node.el_bin,
                })
                probs = self.model.predict(adapted)[0]
                for i in range(self.max_trackers):
                    if i >= len(node.active_mask) or not node.active_mask[i]:
                        probs[i+1] = 0
                action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                plan.append(action)
                if action > 0:
                    node.active_mask[action-1] = False
                continue
                
            best_child = max(node.children, key=lambda c: c.visits)
            plan.append(best_child.action)
            node = best_child
            if node.is_terminal(): break
            
        return plan, policy
