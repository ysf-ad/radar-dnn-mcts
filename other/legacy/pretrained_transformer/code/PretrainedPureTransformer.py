import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# OPTIMIZATION: Enable TensorFloat32 (TF32) for faster matrix multiplication on Ampere+ GPUs
# This provides ~1.2-1.5x speedup for transformer operations without precision loss
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class PretrainedPureTransformer(nn.Module):
    """
    Pretrained Pure Transformer for Task Scheduling
    Uses masked task modeling for pretraining
    """
    def __init__(self, args):
        super(PretrainedPureTransformer, self).__init__()
        self.args = args
        self.use_cls_token = not args.get("no_cls_token", False)
        # Default to Explicit CLS (Winner) unless implicit_cls is requested
        self.use_explicit_cls = not args.get("implicit_cls", False)
        # Default to Positional Encoding unless disabled
        self.use_pos_encoding = not args.get("no_pos_encoding", False)
        
        if not self.use_cls_token:
            print("ABLATION: Disabling CLS Token (True No-CLS mode)")
        elif not self.use_explicit_cls:
            print("ABLATION: Using Implicit CLS (Baseline mode)")
        else:
            print("CONFIG: Using Explicit CLS Readout (Winner mode)")
            
        if not self.use_pos_encoding:
            print("ABLATION: Disabling Positional Encoding")
        self.num_tasks = args["NumTasks"]
        self.num_features = args["NumFeatures"]
        self.d_model = 128  # Transformer dimension
        self.nhead = 8      # Number of attention heads
        self.nlayers = 4    # Number of transformer layers
        self.mask_ratio = 0.15  # 15% masking like BERT
        
        # Input embedding: each task gets its own token
        self.task_embedding = nn.Linear(self.num_features, self.d_model)
        self.position_embedding = nn.Embedding(self.num_tasks, self.d_model)
        
        # Special tokens
        self.mask_token = nn.Parameter(torch.randn(self.d_model))
        self.cls_token = nn.Parameter(torch.randn(self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.nlayers)
        
        # Ablation: No Attention (PointNet style)
        if args.get("no_attention", False):
            print("ABLATION: Disabling Self-Attention (PointNet mode)")
            single_mlp_mode = bool(args.get("no_attention_single_mlp", False))
            if single_mlp_mode:
                print("ABLATION: Single-MLP no-attention mode enabled")
            # Optional fairness control: widen FFN so no-attention ablation is close in parameter count
            # to the attention model (default uses ~512 hidden dim, matched mode uses 771).
            no_attn_hidden_dim = int(args.get("no_attention_hidden_dim", 512))
            if args.get("no_attention_match_params", False):
                no_attn_hidden_dim = 771
                print("ABLATION: Parameter-matched no-attention mode enabled (hidden_dim=771)")
            else:
                print(f"ABLATION: No-attention hidden_dim={no_attn_hidden_dim}")
            # Replace Transformer with feed-forward token MLP(s) only (no cross-task communication).
            if single_mlp_mode:
                self.transformer = nn.Sequential(
                    nn.Linear(self.d_model, no_attn_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(no_attn_hidden_dim, self.d_model),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            else:
                layers = []
                for _ in range(self.nlayers):
                    layers.append(nn.Linear(self.d_model, no_attn_hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    layers.append(nn.Linear(no_attn_hidden_dim, self.d_model))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    # Optional: Add LayerNorm to match Transformer block structure? 
                    # For simplicity, just standard FFN block.
                self.transformer = nn.Sequential(*layers)
        
        # Reconstruction head for masked tasks
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_features)  # Reconstruct all features of a task
        )
        # Initialize reconstruction head with Xavier uniform for better convergence
        for module in self.reconstruction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Policy head: predict which task to schedule next
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Single output per task
        )
        
        # Training data storage
        self.prob_input_training = np.array([], dtype=np.float64)
        self.labels_training = np.array([], dtype=np.float64)
        
        # Training variables
        self.current_learning_rate = 0.0001
        self.batch_data = torch.tensor(np.empty((args["batch_size"], 1, args["NumFeatures"], 
                                               args["NumTasks"]), dtype=np.float64))
        self.batch_labels = torch.tensor(np.empty((args["batch_size"], args["NumTasks"]), dtype=np.float64))
        
        # Loss and optimizer
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = None
        
        # Checkpointing
        self.checkpoint = None
        self.Epoch_index = 0
        self.loss_total = 0
        self.loss_avg = 0
        self.frozen = False
        
    def create_task_mask(self, x, mask_ratio=0.15):
        """
        Create mask for entire tasks (not individual features)
        x: [batch, tasks, features]
        """
        batch_size, num_tasks, num_features = x.shape
        device = x.device
        
        # Create mask for tasks with adaptive masking
        # Use different masking ratios for more diverse training
        if torch.rand(1).item() < 0.5:
            # Random masking
            mask = torch.rand(batch_size, num_tasks, device=device) < mask_ratio
        else:
            # Sequential masking (mask consecutive tasks)
            mask = torch.zeros(batch_size, num_tasks, device=device, dtype=torch.bool)
            for b in range(batch_size):
                start_idx = torch.randint(0, max(1, num_tasks - 2), (1,)).item()
                end_idx = min(start_idx + int(mask_ratio * num_tasks), num_tasks)
                mask[b, start_idx:end_idx] = True
        
        # Create masked input
        masked_x = x.clone()
        masked_x[mask] = 0.0  # Mask entire tasks with zeros
        
        return masked_x, mask

    def _build_padding_mask(self, x):
        """
        Build a key-padding mask for padded task tokens.
        A token is treated as padding when core features are zero and it is marked dropped.
        x: [batch, tasks, features]
        returns: [batch, tasks] bool mask, True means ignore token.
        """
        # Core task fields are first 6 features in this project setup.
        core_zero = (x[:, :, :6].abs().sum(dim=-1) < 1e-8)
        # Feature 6 = scheduled flag, feature 7 = dropped flag.
        unscheduled = (x[:, :, 6] < 0.5)
        dropped = (x[:, :, 7] > 0.5)
        return core_zero & unscheduled & dropped

    def _sinusoidal_pos_encoding(self, seq_len, batch_size, device, dtype):
        """Fallback positional encoding for sequence lengths beyond learned embedding table."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros((seq_len, self.d_model), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        return pe.to(dtype=dtype)

    def _normalize_input(self, example_input):
        """
        Normalize arbitrary input layouts to [batch, tasks, features] for inference.
        """
        arr = np.array(example_input)

        # Collapse singleton dimensions commonly used in this project.
        if arr.ndim == 5 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)  # e.g., [B, F, 1, T]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = np.squeeze(arr, axis=1)   # e.g., [B, F, T]
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)  # e.g., [B, F, T]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)  # e.g., [F, T]

        # Convert to [batch, tasks, features]
        if arr.ndim == 2:
            # [features, tasks] or [tasks, features]
            if arr.shape[0] == self.num_features and arr.shape[1] != self.num_features:
                arr = arr.transpose(1, 0)
            elif arr.shape[1] != self.num_features:
                raise ValueError(f"Cannot infer feature axis from shape {arr.shape}")
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        elif arr.ndim == 3:
            # [batch, features, tasks] or [batch, tasks, features]
            if arr.shape[-1] == self.num_features:
                pass
            elif arr.shape[1] == self.num_features:
                arr = arr.transpose(0, 2, 1)
            elif arr.shape[0] == self.num_features and arr.shape[-1] != self.num_features:
                # Single sample [features, tasks, ?] is not a valid inference layout in this repo.
                raise ValueError(f"Unsupported 3D inference shape {arr.shape}")
            else:
                raise ValueError(f"Cannot infer feature axis from shape {arr.shape}")
        else:
            raise ValueError(f"Unsupported input shape {arr.shape}")

        return arr.astype(np.float32, copy=False)
    
    def forward_pretrain(self, x):
        """
        Forward pass for pretraining with masked task modeling
        x: [batch, tasks, features]
        """
        if x.dim() == 5:  # [batch, features, 1, tasks, 1]
            x = x.squeeze(-1).squeeze(2)  # [batch, features, tasks]
            x = x.transpose(1, 2)  # [batch, tasks, features]
        elif x.dim() == 4:  # [batch, 1, features, tasks]
            x = x.squeeze(1)  # [batch, features, tasks]
            x = x.transpose(1, 2)  # [batch, tasks, features]
        batch_size, num_tasks, num_features = x.shape
        device = x.device
        
        # Create mask
        masked_x, mask = self.create_task_mask(x, self.mask_ratio)
        
        # Create position indices
        positions = torch.arange(num_tasks, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tasks and add positional encoding
        task_embeddings = self.task_embedding(masked_x)  # [batch, tasks, d_model]
        pos_embeddings = self.position_embedding(positions)  # [batch, tasks, d_model]
        
        # Apply mask tokens
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(batch_size, num_tasks, -1)
        task_embeddings[mask] = mask_tokens[mask]
        
        # Combine embeddings
        embeddings = task_embeddings + pos_embeddings
        
        # Add CLS token at the beginning (only if enabled)
        if self.use_cls_token:
            cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Transformer forward pass
        transformer_output = self.transformer(embeddings)  # [batch, 1+tasks, d_model] OR [batch, tasks, d_model]
        
        # Remove CLS token for reconstruction
        if self.use_cls_token:
            task_outputs = transformer_output[:, 1:, :]  # [batch, tasks, d_model]
        else:
            task_outputs = transformer_output  # [batch, tasks, d_model]
        
        return task_outputs, mask
    
    def forward(self, x):
        """
        Forward pass for policy prediction (no masking)
        x: [batch, tasks, features] (already in correct format from predict)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Input is already in [batch, tasks, features] format
        # No reshaping needed
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tasks
        task_embeddings = self.task_embedding(x)  # [batch, tasks, d_model]
        
        # Add positional encoding (if enabled)
        if self.use_pos_encoding:
            if seq_len <= self.position_embedding.num_embeddings:
                pos_embeddings = self.position_embedding(positions)  # [batch, tasks, d_model]
            else:
                # Dynamic input fallback beyond learned positional table.
                pos_embeddings = self._sinusoidal_pos_encoding(seq_len, batch_size, x.device, task_embeddings.dtype)
            embeddings = task_embeddings + pos_embeddings
        else:
            embeddings = task_embeddings
        
        # Add CLS token (only if enabled)
        if self.use_cls_token:
            cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Build key-padding mask so attention ignores padded/dummy tasks.
        # This is critical when using dynamic input sizes padded to NumTasks.
        src_key_padding_mask = None
        if not self.args.get("no_attention", False):
            task_padding = self._build_padding_mask(x)  # [batch, tasks]
            if self.use_cls_token:
                cls_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_pad, task_padding], dim=1)
            else:
                src_key_padding_mask = task_padding

        # Transformer forward pass
        if src_key_padding_mask is not None:
            transformer_output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        else:
            transformer_output = self.transformer(embeddings)  # [batch, 1+tasks, d_model] OR [batch, tasks, d_model]
        
        # Remove CLS token for policy prediction
        if self.use_cls_token:
            # Split CLS from task tokens
            cls_output = transformer_output[:, 0, :]  # [batch, d_model]
            task_outputs = transformer_output[:, 1:, :]  # [batch, tasks, d_model]
            
            if self.use_explicit_cls:
                # EXPLICIT CLS (Winner): Add global context to every task
                # Expand CLS to match tasks: [batch, 1, d_model] -> [batch, tasks, d_model]
                cls_expanded = cls_output.unsqueeze(1).expand(-1, seq_len, -1)
                # Combine via Addition (as per ablation winner)
                task_outputs = task_outputs + cls_expanded
        else:
            # True No-CLS mode
            task_outputs = transformer_output  # [batch, tasks, d_model]
        
        # Policy prediction for each task
        policy_logits = self.policy_head(task_outputs).squeeze(-1)  # [batch, tasks]
        
        return policy_logits
    
    def predict(self, example_input):
        """Predict method for MCTS compatibility"""
        example_input = self._normalize_input(example_input)
        example_input = torch.from_numpy(example_input).float()
        device = next(self.parameters()).device
        example_input = example_input.to(device)
        
        self.eval()
        with torch.no_grad():
            policy_logits = self.forward(example_input)
            policy_pi = torch.softmax(policy_logits, dim=1)
        # OPTIMIZATION: Keep on GPU to avoid CPU-GPU transfer overhead
        # Convert to numpy only when needed (in policy_select)
        return policy_pi
    
    def predict_batch(self, example_inputs):
        """
        OPTIMIZATION: Batch prediction for multiple inputs at once (much faster on GPU).
        Args:
            example_inputs: List of normalized problem inputs or numpy array
        Returns:
            Batch of predictions as torch.Tensor on GPU (shape [batch, num_tasks])
        """
        device = next(self.parameters()).device
        
        # Handle list of inputs
        if isinstance(example_inputs, list):
            batch_inputs = []
            for example_input in example_inputs:
                batch_inputs.append(self._normalize_input(example_input))
            batch_tensor = np.concatenate(batch_inputs, axis=0)  # [batch, tasks, features]
        else:
            # Already batched or single input
            batch_tensor = self._normalize_input(example_inputs)
        
        batch_tensor = torch.from_numpy(batch_tensor).float().to(device)
        
        self.eval()
        with torch.no_grad():
            policy_logits = self.forward(batch_tensor)
            policy_pi = torch.softmax(policy_logits, dim=1)
        
        return policy_pi
    
    def pretrain_transformer(self, pretrain_data, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Pretrain transformer with masked task modeling
        """
        print(f"Starting PureTransformer pretraining for {epochs} epochs...")
        
        device = next(self.parameters()).device
        self.to(device)
        self.train()
        
        # Convert data to tensor
        pretrain_tensor = torch.from_numpy(pretrain_data).float().to(device)
        
        # Handle different input shapes
        if pretrain_tensor.dim() == 5:  # [batch, 1, features, tasks, 1]
            pretrain_tensor = pretrain_tensor.squeeze(-1).squeeze(1)  # [batch, features, tasks]
        elif pretrain_tensor.dim() == 4:  # [batch, 1, features, tasks]
            pretrain_tensor = pretrain_tensor.squeeze(1)  # [batch, features, tasks]
        
        pretrain_tensor = pretrain_tensor.transpose(1, 2)  # [batch, tasks, features]
        
        # Pretraining optimizer with learning rate scheduling
        pretrain_optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pretrain_optimizer, T_max=epochs, eta_min=learning_rate * 0.1
        )
        
        num_batches = len(pretrain_tensor) // batch_size
        print(f"Training with {num_batches} batches per epoch")
        
        # Create fixed permutation for consistent training across epochs
        fixed_permutation = torch.randperm(len(pretrain_tensor))
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(num_batches):
                indices = fixed_permutation[i * batch_size : (i + 1) * batch_size]
                batch_data = pretrain_tensor[indices]
                
                pretrain_optimizer.zero_grad()
                
                # Forward pass with masking
                task_outputs, mask = self.forward_pretrain(batch_data)
                
                # Reconstruction loss for masked tasks
                reconstruction_loss = 0
                mask_count = 0
                
                for task_idx in range(self.num_tasks):
                    if mask[:, task_idx].any():
                        # Reconstruct task features
                        reconstructed = self.reconstruction_head(task_outputs[:, task_idx, :])
                        original = batch_data[:, task_idx, :]
                        
                        # MSE loss for reconstruction (ensure same shape)
                        if reconstructed.shape != original.shape:
                            continue
                        reconstruction_loss += F.mse_loss(reconstructed, original)
                        mask_count += 1
                
                if mask_count > 0:
                    reconstruction_loss = reconstruction_loss / mask_count
                    reconstruction_loss.backward()
                    pretrain_optimizer.step()
                    total_loss += reconstruction_loss.item()
            
            # Update learning rate
            pretrain_scheduler.step()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if epoch % 50 == 0:  # Less frequent updates
                current_lr = pretrain_optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        print("PureTransformer pretraining complete!")
        torch.save(self.state_dict(), 'pretrained_pure_transformer.pth')
        return self
    
    def train_mcts(self, current_learning_rate, use_pretrain_epochs=False):
        """
        Training method optimized for transformer
        
        Args:
            current_learning_rate: Learning rate for training
            use_pretrain_epochs: If True, use TransformerPretrainEpochs (for initial pretraining).
                                If False, use EpochsMax (for MCTS iteration fine-tuning).
        """
        # FIX: Ensure training data is limited to MaxTrainData before training starts
        # This prevents unbounded growth that causes slowdowns
        initial_size = len(self.labels_training) if self.labels_training.size > 0 else 0
        if initial_size > self.args["MaxTrainData"]:
            excess = initial_size - self.args["MaxTrainData"]
            self.prob_input_training = self.prob_input_training[excess:]
            self.labels_training = self.labels_training[excess:]
            print(f"    Training data trimmed: {initial_size} -> {len(self.labels_training)} samples", flush=True)
        
        # Safety check: Ensure prob_input_training and labels_training have matching sizes
        prob_size = len(self.prob_input_training) if self.prob_input_training.size > 0 else 0
        label_size = len(self.labels_training) if self.labels_training.size > 0 else 0
        if prob_size != label_size:
            min_size = min(prob_size, label_size)
            if min_size > 0:
                self.prob_input_training = self.prob_input_training[:min_size]
                self.labels_training = self.labels_training[:min_size]
                print(f"    Training data size mismatch fixed: prob={prob_size}, label={label_size} -> {min_size}", flush=True)
            else:
                print("No training data available after size mismatch fix!")
                return
        
        num_training_points = len(self.labels_training) if self.labels_training.size > 0 else 0
        if num_training_points == 0:
            print("No training data available!")
            return
        
        # Clear GPU cache before training to prevent memory accumulation
        device = next(self.parameters()).device
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        effective_batch_size = min(self.args["batch_size"], num_training_points)
        num_batches = max(1, int(num_training_points / effective_batch_size))
        
        permutation = np.random.permutation(num_training_points)
        break_point = 0
        self.Epoch_index = 0
        
        device = next(self.parameters()).device
        self.to(device)
        
        # Track whether we're in pretraining mode to decide optimizer type
        # Recreate optimizer when switching modes or LR changes
        need_new_optimizer = (
            not hasattr(self, 'optimizer') or 
            self.optimizer is None or 
            self.optimizer.param_groups[0]['lr'] != current_learning_rate or
            (use_pretrain_epochs and isinstance(self.optimizer, torch.optim.Adam)) or
            (not use_pretrain_epochs and isinstance(self.optimizer, torch.optim.AdamW))
        )
        
        if need_new_optimizer:
            # Use AdamW for pretraining (better for transformers), Adam for MCTS iterations (matches CNN, faster)
            if use_pretrain_epochs:
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=current_learning_rate, weight_decay=0.01)
                # Learning rate scheduler for transformer pretraining
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=2
                )
            else:
                # Use Adam during MCTS iterations to match CNN (simpler, faster)
                self.optimizer = torch.optim.Adam(self.parameters(), lr=current_learning_rate)
                self.scheduler = None  # No scheduler during MCTS iterations
        if not hasattr(self, 'criterion'):
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        
        while break_point == 0:
            batch_index = 0
            self.loss_total = 0
            stop_counter = 0
            
            while batch_index < num_batches:
                offset = batch_index * effective_batch_size
                ind = permutation[offset: offset + effective_batch_size]
                
                # Safety check: Clip indices to valid bounds
                ind = np.clip(ind, 0, num_training_points - 1)
                
                self.batch_data = torch.from_numpy(self.prob_input_training[ind, :, :, :]).float().to(device)
                self.batch_labels = torch.from_numpy(self.labels_training[ind, :]).float().to(device)
                
                x = self.batch_data
                # Reshape from [batch, 1, features, tasks] to [batch, tasks, features]
                x = x.squeeze(1).transpose(1, 2)  # [batch, tasks, features]
                targets = self.batch_labels.argmax(dim=1)
                
                self.optimizer.zero_grad()
                policy_logits = self.forward(x)
                loss = self.criterion(policy_logits, targets)
                
                # L2 regularization
                l2_loss = torch.tensor(0., requires_grad=True).to(device)
                for name, p in self.named_parameters():
                    if 'weight' in name:
                        l2_loss = l2_loss + (p.pow(2.0).sum())
                
                loss += l2_loss * self.args["Beta_L2"]
                
                loss.backward()
                
                # Gradient clipping for stability (always enabled)
                # This prevents large cost spikes during training
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                batch_index += 1
                self.loss_total += loss.item()
            
            # Update learning rate scheduler only during initial pretraining
            # During MCTS iterations, use fixed LR (same as CNN) for speed
            if use_pretrain_epochs and hasattr(self, 'scheduler'):
                self.scheduler.step(self.loss_total / (batch_index + 1))
            
            self.Epoch_index += 1
            self.loss_avg = self.loss_total / (batch_index + 1)
            
            # Use TransformerPretrainEpochs for pretraining, EpochsMax for fine-tuning
            # train_mcts is called both for initial pretraining (should use TransformerPretrainEpochs)
            # and for fine-tuning during MCTS iterations (should use EpochsMax)
            if use_pretrain_epochs:
                max_epochs = self.args.get("TransformerPretrainEpochs", 200)
            else:
                # Use EpochsMax for MCTS iteration fine-tuning (same as CNN)
                max_epochs = self.args.get("EpochsMax", 10)
            
            # Print loss - match CNN's print frequency during MCTS iterations
            if use_pretrain_epochs:
                # During pretraining: print every 50 epochs or every 10 if < 200 epochs
                print_freq = 50 if max_epochs > 200 else 10
                if self.Epoch_index % print_freq == 0 or self.Epoch_index == 1:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Pretrain Epoch {self.Epoch_index}, Avg Loss: {self.loss_avg:.6f}, LR: {current_lr:.6e}")
            else:
                # During MCTS iterations: match CNN's print frequency (every 5 epochs or first/last)
                if self.Epoch_index == 0 or self.Epoch_index == max_epochs - 1 or self.Epoch_index % 5 == 0:
                    print(f'Epoch {self.Epoch_index}/{max_epochs-1}, Batch {batch_index}')
            
            if self.Epoch_index == 1:
                loss_avg_min = self.loss_avg
            elif self.Epoch_index > max_epochs - 1:
                break_point = 1
            elif self.loss_avg >= loss_avg_min:
                stop_counter += 1
                if stop_counter == 3:
                    print(f"  Pretraining stopped at epoch {self.Epoch_index} (loss plateaued)")
                    break_point = 1
            else:
                stop_counter = 0
                loss_avg_min = self.loss_avg
    
    def save_checkpoint(self, folder='checkpoint', filename='transformer_checkpoint.pth'):
        """Persist model checkpoint to disk (mirrors NeuNet API)."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save(self.state_dict(), filepath)
        self.checkpoint = filepath
    
    def load_checkpoint(self, folder='checkpoint', filename='transformer_checkpoint.pth'):
        """Load model checkpoint if it exists on disk (fallback to in-memory copy)."""
        filepath = os.path.join(folder, filename)
        target_path = filepath if os.path.exists(filepath) else getattr(self, "checkpoint", None)
        if target_path is None or not os.path.exists(target_path):
            raise FileNotFoundError(f"No transformer checkpoint found at {filepath}")
        state_dict = torch.load(target_path, map_location=next(self.parameters()).device)
        self.load_state_dict(state_dict)
    
    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False
