"""
Transformer-guided MCTS Planner for Radar Task Scheduling.
Uses 4 features per tracker (t_desired, t_deadline, t_dwell, priority).

NOTE: Transformer expects 8-feature format. adapt_obs() converts from 4-feature.
Falls back to pure MCTS if no checkpoint loaded.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

REWARD_SCALE = 20.0

from .mcts import MCTSPlanner, Node
from .planner import Planner


class PretrainedPureTransformer(nn.Module if TORCH_AVAILABLE else object):
    """Transformer for task scheduling (8-feature input)."""
    
    def __init__(self, num_tasks=501, num_features=9, d_model=128, nhead=8, nlayers=4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.d_model = d_model
        
        self.task_embedding = nn.Linear(num_features, d_model)
        # No position embeddings (content-only, order-invariant)
        self.cls_token = nn.Parameter(torch.randn(d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, mask=None):
        """
        x: (B, N, F)
        mask: (B, N) boolean tensor where True indicates padding (inactive tasks to ignore)
        """
        batch_size = x.shape[0]
        embeddings = self.task_embedding(x)
        
        # CLS token creation
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Apply key padding mask so the transformer ignores inactive/padded tasks.
        # The transformer expects mask shape (B, S) where True = ignore.
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Prepend CLS token (never masked)
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            key_padding_mask = None

        output = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)
        
        # Policy: predict logits for each task from task tokens
        task_outputs = output[:, 1:, :]
        logits = self.policy_head(task_outputs).squeeze(-1)
        
        # Value: predict scalar value from CLS token
        cls_output = output[:, 0, :]
        value = self.value_head(cls_output).squeeze(-1)
        
        return logits, value
    
    def predict(self, x):
        x = np.array(x)
        if hasattr(x, 'shape') and len(x.shape) == 2:
            x = torch.from_numpy(x).float().unsqueeze(0)
        elif hasattr(x, 'shape') and len(x.shape) == 3:
            x = torch.from_numpy(x).float()
            
        if TORCH_AVAILABLE:
            # Move to model device
            device = next(self.parameters()).device
            x = x.to(device)
            
            with torch.no_grad():
                logits, value = self.forward(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                # Scale value back to real units
                val = value.cpu().numpy().item() * REWARD_SCALE
                return probs, val
        return None, None


class TransformerMCTSPlanner(Planner):
    """Transformer-guided MCTS (falls back to pure MCTS if no model)."""
    
    def __init__(self, checkpoint_path=None, model=None, max_trackers=500, num_rollouts=50, device='cuda', use_search=True, batch_size=16):
        super().__init__(max_trackers)
        self.max_trackers = max_trackers
        self.num_tasks = max_trackers + 1
        self.SEARCH_ACTION = 0
        self.batch_size = batch_size
        
        # Fallback MCTS
        self.pure_mcts = MCTSPlanner(max_trackers=max_trackers, num_rollouts=num_rollouts)
        
        # Load Transformer if available
        self.model = model
        if self.model is None and TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.model = PretrainedPureTransformer(num_tasks=self.num_tasks).to(self.device)
            
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            self.model.load_state_dict(model_state)
            self.model.eval()
        elif self.model is not None:
            self.device = next(self.model.parameters()).device

            # Optimization: MatMul Precision for TensorCores
            if device == 'cuda':
                torch.set_float32_matmul_precision('high')

            # Optimization: torch.compile failed (PermissionError on nvcc)
            # Reverting to eager mode with Autocast
            # try:
            #      self.model = torch.compile(self.model, mode="reduce-overhead")
            # except Exception as e:
            #      print(f"Warning: torch.compile failed: {e}")
            print(f"Loaded Transformer from {checkpoint_path}")
    
    def adapt_obs(self, obs):
        """Standardize observation to 9-feature radar format used in training."""
        # Adapted from train_sensor_aware.py adapt_obs_sensor_aware
        
        # 1. Feature Extraction
        # Note: 'grid' is (GRID_SIZE,) normally 20 sectors. 
        # In sensor_aware we ignored grid. We only used tracker features.
        
        sensor_id = obs.get('sensor_id', 0)
        
        num_tasks = self.num_tasks # 301
        adapted = np.zeros((1, num_tasks, 9), dtype=np.float32)
        
        # Row 0: SEARCH action
        # Features: [t_des, t_dwell, tardiness, t_deadline, 100.0, 0, 0, active, sensor_id]
        adapted[0, 0, :] = [
            0.0,  # t_desired
            0.0,  # t_dwell (log1p(0))
            0.0, # tardiness
            0.0, # t_deadline
            0.0, # 100.0 (dummy) - wait, trainer uses 0.0 for task 0
            0.0,
            0.0,
            1.0, # active (Search always active)
            sensor_id
        ]
        
        # Rows 1..N: TRACK actions
        t_desired = obs['t_desired']
        t_dwell = obs['t_dwell']
        t_tardiness = np.maximum(0, -obs['t_desired']) * 0.01
        t_deadline = obs['t_deadline']
        
        # NORMALIZATION (Must match train_sensor_aware.py)
        # log1p normalization for time features
        # [0] t_desired
        adapted[0, 1:, 0] = np.sign(t_desired) * np.log1p(np.abs(t_desired))
        
        # [1] t_dwell - Fix for potential -1.0 or negative values
        dwell_safe = np.maximum(0.0, t_dwell) 
        adapted[0, 1:, 1] = np.log1p(dwell_safe)
        
        # [2] tardiness
        adapted[0, 1:, 2] = np.log1p(t_tardiness)
        
        # [3] t_deadline
        adapted[0, 1:, 3] = np.sign(t_deadline) * np.log1p(np.abs(t_deadline))
        
        # [4] Constant 100.0 (This was in train_sensor_aware logic... let's keep consistent)
        adapted[0, 1:, 4] = 100.0 
        
        # [5, 6] unused/grid mean? train_sensor_aware set them to 0.0
        adapted[0, 1:, 5] = 0.0
        adapted[0, 1:, 6] = 0.0
        
        # [7] Inactive Flag (train_sensor_aware used (~active).astype(float))
        adapted[0, 1:, 7] = (~obs['active_mask']).astype(np.float32)
        
        # [8] Sensor ID
        adapted[0, 1:, 8] = sensor_id
        
        # Clip to match training bounds
        adapted = np.clip(adapted, -20.0, 20.0)
        
        return adapted
    
    
    def plan(self, obs, budget_ms=200):
        """Generate action plan using model-guided Batched MCTS."""
        if self.model is None:
            return self.pure_mcts.plan(obs, budget_ms=budget_ms)
        
        # Determine rollouts
        # Use num_rollouts if set, or infer from budget (approx 5ms per batch of 16?)
        # For consistency with training/benchmarking, we rely on num_rollouts.
        num_batches = self.pure_mcts.num_rollouts // self.batch_size
        remainder = self.pure_mcts.num_rollouts % self.batch_size
        
        # --- 1. SETUP ROOT ---
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask']
        )
        
        # Expand Root first (always 1 node)
        self._evaluate_batch([root])
        
        # --- 2. BATCH MCTS LOOP ---
        for _ in range(num_batches):
            leaves = []
            
            # Select 'batch_size' leaves
            for _ in range(self.batch_size):
                leaf = self._select_with_virtual_loss(root)
                leaves.append(leaf)
            
            # Evaluate Batch
            self._evaluate_batch(leaves)
            
        # Handle remainder
        for _ in range(remainder):
             leaf = self.pure_mcts._select(root)
             if not leaf.is_terminal(): 
                 self._evaluate_batch([leaf])

        # --- 3. EXTRACT PLAN ---
        plan = []
        node = root
        max_steps = int(budget_ms / 10.0) + 2
        
        for _ in range(max_steps):
            if not node.children:
                # Fallback to model greedy if search depth reached or no children
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask
                })
                # Using single predict here (slow path, but rare)
                p_batch, _ = self.model.predict(adapted)
                if p_batch is not None:
                    probs = p_batch[0]
                    probs[0] = 0 # No search if possible
                    for i in range(self.max_trackers):
                        if i >= len(node.active_mask) or not node.active_mask[i]:
                            probs[i+1] = 0
                    action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                else:
                    action = self.SEARCH_ACTION
                    
                plan.append(action)
                if action > 0:
                    node.active_mask[action-1] = False
                continue
                
            best_child = max(node.children, key=lambda c: c.total_reward / max(1, c.visits))
            plan.append(best_child.action)
            node = best_child
            if node.is_terminal(): break
            
        return plan

    def _select_with_virtual_loss(self, node):
        """Select a leaf and apply virtual loss (visit count + 1)."""
        while node.expanded and node.children and not node.is_terminal():
            # PUCT Select
            best_score = -float('inf')
            best_child = None
            
            for child in node.children:
                exploit = child.total_reward / max(1, child.visits)
                explore = self.pure_mcts.c * child.prior_prob * np.sqrt(node.visits + 1) / (1 + child.visits)
                score = exploit + explore
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
            
        # Apply Virtual Loss
        node.visits += 1 
        return node

    def _evaluate_batch(self, nodes):
        """Run neural network on a batch of nodes."""
        # Special case: Always allow expanding ROOT (first node in list of 1) if it's the only one
        # This handles the "Cold Start" where we have 0 targets but need to query value/policy for Search
        valid_nodes = []
        for i, n in enumerate(nodes):
             if not n.is_terminal() or (len(nodes) == 1 and i == 0 and not n.expanded):
                 valid_nodes.append(n)

        if not valid_nodes:
            # Handle terminal nodes (undo virtual loss)
            for n in nodes:
                 if n.is_terminal():
                     n.visits -= 1 
                     self.pure_mcts._backprop(n, 0.0) 
            return

        # Prepare Batch Tensor
        batch_obs = []
        for node in valid_nodes:
            adapted = self.adapt_obs({
                't_desired': node.t_desired,
                't_deadline': node.t_deadline,
                't_dwell': node.t_dwell,
                'priority': node.priority,
                'active_mask': node.active_mask
            })
            batch_obs.append(adapted[0]) # adapt_obs returns (1, N, 8)
            
        # Stack
        input_tensor = torch.tensor(np.array(batch_obs), device=self.device)
        
        # Inference
        with torch.no_grad():
            if self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, values = self.model.forward(input_tensor)
            else:
                 logits, values = self.model.forward(input_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # --- SCALE VALUE BACK TO REAL UNITS ---
            values = values.cpu().numpy().flatten() * REWARD_SCALE
            
        # Expand & Backprop
        for i, node in enumerate(valid_nodes):
            # Expand
            self.pure_mcts._expand(node, priors=probs[i], top_k=10, force_engagement=True)
            
            # Backprop Value
            val = values[i]
            
            # Undo Virtual Loss (decrement visit we added)
            node.visits -= 1 
            
            # Valid Backprop
            self.pure_mcts._backprop(node, val)
            
        # Handle terminal nodes in the batch (if any were filtered out)
        for n in nodes:
             if n not in valid_nodes and n.is_terminal():
                 n.visits -= 1
                 self.pure_mcts._backprop(n, 0.0)
    
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
            active_mask=obs['active_mask']
        )
        
        # MCTS loop (Batched for efficiency)
        # Re-use logic from plan() but we need to track root visits
        
        # Calculate batches
        num_batches = self.pure_mcts.num_rollouts // self.batch_size
        remainder = self.pure_mcts.num_rollouts % self.batch_size
        
        # Expand Root first (always 1 node) if not expanded
        if not root.expanded:
            self._evaluate_batch([root])
            
        # Parallel MCTS Loop
        for _ in range(num_batches):
            leaves = []
            # 1. Select 'batch_size' leaves
            for _ in range(self.batch_size):
                leaf = self._select_with_virtual_loss(root)
                leaves.append(leaf)
            
            # 2. Evaluate Batch (Model inference + Expand + Backprop)
            self._evaluate_batch(leaves)
            
        # Handle remainder
        for _ in range(remainder):
             leaf = self._select_with_virtual_loss(root)
             self._evaluate_batch([leaf])
        
        # Extract policy target from root visit counts
        policy = np.zeros(self.num_tasks, dtype=np.float32)
        total_visits = sum(c.visits for c in root.children)
        if total_visits > 0:
            for child in root.children:
                policy[child.action] = child.visits / total_visits
        
        # Extract plan (Time-aware)
        plan = []
        node = root
        accumulated_time = 0.0
        
        # Safety limit (large enough to never hit unless infinite loop)
        for _ in range(100):
            if not node.children:
                # Fallback to model greedy
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask
                })
                p_batch, _ = self.model.predict(adapted)
                if p_batch is not None:
                    probs = p_batch[0]
                    probs[0] = 0
                    for i in range(self.max_trackers):
                        if i >= len(node.active_mask) or not node.active_mask[i]:
                            probs[i+1] = 0
                    action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                else:
                    action = self.SEARCH_ACTION
                
                # Estimate Dwell
                if action == self.SEARCH_ACTION:
                    dwell = 10.0
                else:
                    idx = action - 1
                    dwell = float(node.t_dwell[idx]) if idx < len(node.t_dwell) else 10.0
                
                plan.append(action)
                accumulated_time += dwell
                
                if action > 0:
                    node.active_mask[action-1] = False
                
                if accumulated_time >= budget_ms:
                    break
                continue
                
            best_child = max(node.children, key=lambda c: c.total_reward / max(1, c.visits))
            action = best_child.action
            
            # Estimate Dwell for child
            if action == self.SEARCH_ACTION:
                dwell = 10.0
            else:
                idx = action - 1
                dwell = float(node.t_dwell[idx]) if idx < len(node.t_dwell) else 10.0
            
            plan.append(action)
            accumulated_time += dwell
            
            node = best_child
            
            if accumulated_time >= budget_ms:
                break
            
            if node.is_terminal(): 
                break
            
        return plan, policy
