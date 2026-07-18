"""
Transformer variant that uses CLS token for policy head input
This is a copy of PretrainedPureTransformer with modified forward() method
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import base class
from PretrainedPureTransformer import PretrainedPureTransformer

class PretrainedPureTransformer_CLS(PretrainedPureTransformer):
    """
    Transformer that uses CLS token output for policy prediction
    Instead of using task outputs directly, we use CLS token as global context
    """
    
    def forward(self, x):
        """
        Forward pass using CLS token for policy prediction
        x: [batch, tasks, features] (already in correct format from predict)
        """
        batch_size = x.shape[0]
        
        # Create position indices
        positions = torch.arange(self.num_tasks, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tasks and add positional encoding
        task_embeddings = self.task_embedding(x)  # [batch, tasks, d_model]
        pos_embeddings = self.position_embedding(positions)  # [batch, tasks, d_model]
        
        # Combine embeddings
        embeddings = task_embeddings + pos_embeddings
        
        # Add CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Transformer forward pass
        transformer_output = self.transformer(embeddings)  # [batch, 1+tasks, d_model]
        
        # USE CLS TOKEN as global context for policy prediction
        cls_output = transformer_output[:, 0, :]  # [batch, d_model] - global representation
        task_outputs = transformer_output[:, 1:, :]  # [batch, tasks, d_model] - task representations
        
        # Option 1: Use CLS to condition task predictions (CLS + task embedding)
        # This allows the model to use global context for each task decision
        cls_expanded = cls_output.unsqueeze(1).expand(-1, self.num_tasks, -1)  # [batch, tasks, d_model]
        
        # Combine CLS and task outputs (element-wise addition or concatenation)
        # Using addition (simpler, same dimension)
        combined_repr = cls_expanded + task_outputs  # [batch, tasks, d_model]
        
        # Policy prediction from combined representation
        policy_logits = self.policy_head(combined_repr).squeeze(-1)  # [batch, tasks]
        
        return policy_logits

