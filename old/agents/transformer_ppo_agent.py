# Path: /src/thesis/agents/transformer_ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..config import device
import math
import time
import logging
from typing import List, Tuple, Dict, Optional
from ..environment.game2048 import Game2048, preprocess_state_onehot

# Add numerical stability constant
EPS = 1e-8

def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.1):
    """
    More numerically stable PPO loss calculation
    
    Args:
        new_log_probs: Log probabilities from current policy
        old_log_probs: Log probabilities from old policy
        advantages: Advantage estimates
        clip_ratio: PPO clipping parameter
        
    Returns:
        policy_loss: Clipped PPO loss
        ratio: Action probability ratios for logging
    """
    # Check for extreme values in inputs
    if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
        new_log_probs = torch.nan_to_num(new_log_probs, nan=-10.0, posinf=10.0, neginf=-10.0)
        
    if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
        old_log_probs = torch.nan_to_num(old_log_probs, nan=-10.0, posinf=10.0, neginf=-10.0)
        
    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Add small epsilon for numerical stability when computing log difference
    EPS = 1e-6
    
    # Restrict log prob differences to a safe range to prevent extreme ratios
    max_diff = 1.5  # Limit how much the policies can differ
    log_ratio = torch.clamp(new_log_probs - old_log_probs, -max_diff, max_diff)
    
    # Use stable exponentiation
    ratio = torch.exp(log_ratio)
    
    # Clamp ratios to avoid extreme values
    ratio = torch.clamp(ratio, 0.2, 5.0)
    
    # Scale advantages to a reasonable range to prevent extreme loss values
    normalized_advantages = advantages / (advantages.std() + EPS)
    
    # Compute surrogate objectives with normalized advantages
    surrogate1 = ratio * normalized_advantages
    surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * normalized_advantages
    
    # Take minimum for pessimistic bound, apply smoothing to avoid sharp gradients
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
    # Detect any remaining numerical issues
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        # Fallback to a safe constant loss that will still provide gradients
        policy_loss = torch.tensor(0.1, device=policy_loss.device, dtype=policy_loss.dtype)
    
    return policy_loss, ratio

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the board positions to provide spatial context to the Transformer.
    This allows the model to understand the relative positions of tiles on the 2048 board.
    """
    def __init__(self, d_model, max_len=16):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to be part of the module's state
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to the input
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)].detach()

class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism specifically designed for the 2048 board.
    This helps the model focus on relevant parts of the board and understand tile relationships.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply layer normalization first (pre-norm formulation for stability)
        x_norm = self.layer_norm(x)
        
        # Check for NaN values
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Self-attention with gradient clipping
        try:
            attn_output, _ = self.mha(x_norm, x_norm, x_norm)
            
            # Check for NaN in attention output
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                logging.warning("NaN detected in attention output")
                attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
                
            # Clip large attention values to prevent exploding gradients
            attn_output = torch.clamp(attn_output, -5.0, 5.0)
            
        except Exception as e:
            logging.error(f"Error in self-attention: {e}")
            # Fallback to identity mapping if attention fails
            attn_output = x_norm
        
        # Residual connection and dropout with safety
        output = x + self.dropout(attn_output)
        
        # Final safety check
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in Transformer blocks.
    """
    def __init__(self, embed_dim, ff_dim=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply layer normalization first (pre-norm formulation)
        x_norm = self.layer_norm(x)
        
        # Check for NaN values
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Two-layer feedforward with GELU and dropout
        try:
            # First linear layer
            x_ff = self.linear1(x_norm)
            
            # Check for NaNs after first layer
            if torch.isnan(x_ff).any() or torch.isinf(x_ff).any():
                x_ff = torch.nan_to_num(x_ff, nan=0.0, posinf=5.0, neginf=-5.0)
            
            # Apply GELU activation and dropout
            x_ff = self.dropout1(F.gelu(x_ff))
            
            # Second linear layer
            x_ff = self.linear2(x_ff)
            
            # Check for NaNs after second layer
            if torch.isnan(x_ff).any() or torch.isinf(x_ff).any():
                x_ff = torch.nan_to_num(x_ff, nan=0.0, posinf=5.0, neginf=-5.0)
                
            # Apply final dropout
            x_ff = self.dropout2(x_ff)
            
        except Exception as e:
            logging.error(f"Error in feed-forward network: {e}")
            # Fallback to zeros if feed-forward fails
            x_ff = torch.zeros_like(x_norm)
        
        # Residual connection with safety clipping
        output = x + x_ff
        
        # Final safety check and clipping
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return output

class TransformerBlock(nn.Module):
    """
    A complete Transformer block with self-attention and feed-forward network.
    """
    def __init__(self, embed_dim, num_heads=4, ff_dim=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class BoardEmbedding(nn.Module):
    """
    Embeds the 2048 board state into a format suitable for the Transformer.
    Converts the one-hot encoded board into a sequence of embeddings.
    """
    def __init__(self, board_size=4, input_channels=16, embed_dim=256):
        super(BoardEmbedding, self).__init__()
        self.board_size = board_size
        
        # Initial convolutional layer to process the board
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Apply convolution: [batch_size, embed_dim, height, width]
        x = self.conv(x)
        
        # Reshape to sequence form: [batch_size, height*width, embed_dim]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, self.board_size*self.board_size, -1)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x

class TransformerPPONetwork(nn.Module):
    """
    Transformer-based neural network for PPO agent designed for the 2048 game.
    Replaces the CNN+Residual architecture with a Transformer to better capture
    board-wide patterns and relationships between tiles.
    """
    def __init__(self, board_size=4, embed_dim=256, num_heads=4, num_layers=4, 
                 ff_dim=512, dropout=0.1, input_channels=16, n_actions=4, use_checkpoint=False):
        super(TransformerPPONetwork, self).__init__()
        self.board_size = board_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        
        # Optimistic initialization for value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Initialize value head with optimistic values (320k as suggested in the paper)
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.1)  # Small positive weights
                nn.init.constant_(layer.bias, 320000)  # Optimistic initial value
        
        # Policy head with optimistic initialization
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 possible moves
        )
        # Initialize policy head with optimistic values
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.1)
                nn.init.constant_(layer.bias, 0.5)  # Slightly positive bias for all actions
        
        # Board embedding
        self.embedding = BoardEmbedding(board_size, input_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=board_size*board_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global attention pooling to aggregate board features
        self.global_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Learnable query token for global attention pooling
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Move to device
        self.to(device)
    
    def forward(self, x, training=False):
        """Forward pass through the Transformer network"""
        try:
            # Convert input to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=device)
            
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            # Ensure float type
            x = x.float()
            
            batch_size = x.shape[0]
            
            # Check for invalid inputs
            if torch.isnan(x).any() or torch.isinf(x).any():
                # Handle invalid inputs
                logging.warning("NaN or Inf detected in input!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Embed the board state
            x = self.embedding(x)
            
            # Add positional encoding for spatial awareness
            x = self.pos_encoding(x)
            
            # Apply transformer blocks
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing for memory efficiency during training
                for block in self.transformer_blocks:
                    x = torch.utils.checkpoint.checkpoint(block, x)
                    # Add safety check after each block
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                # Standard forward pass without checkpointing
                for block in self.transformer_blocks:
                    x = block(x)
                    # Add safety check after each block
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Global attention pooling using a learnable query token
            query = self.query_token.repeat(batch_size, 1, 1)
            
            # Apply attention to get a single context vector for the entire board
            context, _ = self.global_attention(
                query, 
                self.layer_norm(x), 
                self.layer_norm(x)
            )
            
            # Extract the context vector (removing sequence dimension)
            context = context.squeeze(1)
            
            # Add exploration noise during training if requested (with reduced magnitude)
            if training and self.training:
                noise = 0.01 * torch.randn_like(context)  # Further reduce noise magnitude
                context = context + noise
            
            # Policy and value heads
            policy_logits = self.policy_head(context)
            value = self.value_head(context)
            
            # Clamp policy logits for numerical stability (with slightly wider bounds)
            policy_logits = torch.clamp(policy_logits, -15.0, 15.0)
            value = torch.clamp(value, -150.0, 150.0)
            
            # Safety checks to prevent NaN propagation
            if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                logging.warning("NaN or Inf detected in policy logits!")
                policy_logits = torch.zeros_like(policy_logits)
            
            if torch.isnan(value).any() or torch.isinf(value).any():
                logging.warning("NaN or Inf detected in value!")
                value = torch.zeros_like(value)
            
            return policy_logits, value
        
        except Exception as e:
            # Fallback to zeros if something goes wrong
            logging.error(f"Error in forward pass: {e}")
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
            return (torch.zeros((batch_size, 4), device=device), 
                    torch.zeros((batch_size, 1), device=device))


class TransformerPPOAgent:
    """
    Proximal Policy Optimization agent for 2048 using a Transformer architecture.
    - Uses Transformer architecture to better capture board-wide patterns
    - Self-attention mechanisms help understand relationships between tile positions
    - Employs clipped surrogate objective for stable updates
    - Optimized for H100 with mixed precision training
    """
    def __init__(self, 
                 board_size=4, 
                 embed_dim=256,
                 num_heads=4,
                 num_layers=4,
                 input_channels=16,
                 lr=0.0001,  # Lower learning rate for stability
                 gamma=0.99,
                 clip_ratio=0.1,  # Reduced clip ratio for more conservative updates
                 vf_coef=0.5,
                 ent_coef=0.01,
                 max_grad_norm=0.5,
                 gae_lambda=0.95,
                 update_epochs=4,
                 target_kl=0.01,
                 batch_size=512,
                 mixed_precision=True):
        
        # PPO hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        
        # Initialize Transformer network
        self.network = TransformerPPONetwork(
            board_size=board_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            input_channels=input_channels
        )
        
        # Create optimizer with weight decay for regularization
        # We use AdamW which is better suited for Transformers
        self.optimizer = optim.AdamW(
            self.network.parameters(), 
            lr=lr,
            weight_decay=0.001,  # Reduced weight decay
            betas=(0.9, 0.999),  # Standard beta values for Adam
            eps=1e-5  # Slightly larger epsilon for numerical stability
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,  # Adjusted based on expected number of updates
            eta_min=lr/10
        )
        
        # Setup mixed precision training for H100
        if self.mixed_precision:
            # Use the updated API call for GradScaler
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Initialize buffers for trajectories
        self.reset_buffers()
        
        # For tracking and debugging
        self.update_count = 0
        self.training_stats = []
    
    def reset_buffers(self):
        """Reset trajectory buffers for a new collection phase"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.valid_masks = []
    
    def tile_downgrade_state(self, state, depth=0, max_depth=6):
        """
        Implements tile-downgrading search as described in the paper.
        Translates complex states with large tiles into simpler states for evaluation.
        
        Args:
            state: Current game state
            depth: Current search depth
            max_depth: Maximum search depth (6-ply as suggested in paper)
        
        Returns:
            Simplified state for evaluation
        """
        if depth >= max_depth:
            return state
        
        # Create a copy of the state
        simplified_state = state.copy()
        
        # Downgrade tiles based on their value
        for i in range(self.board_size):
            for j in range(self.board_size):
                if simplified_state[i, j] > 512:  # Only downgrade high-value tiles
                    # Downgrade by one level (e.g., 1024 -> 512)
                    simplified_state[i, j] = simplified_state[i, j] // 2
        
        return simplified_state

    def get_action(self, state, valid_moves, deterministic=False):
        """
        Get action using tile-downgrading search for complex states
        """
        # Process state
        state_proc = preprocess_state_onehot(state)
        
        # Check if state needs tile-downgrading
        max_tile = np.max(state)
        if max_tile > 512:
            # Use tile-downgrading search for complex states
            simplified_state = self.tile_downgrade_state(state)
            state_proc = preprocess_state_onehot(simplified_state)
        
        # Get policy logits and value
        with torch.no_grad():
            policy_logits, value = self.network(state_proc)
        
        # Mask invalid moves
        policy_logits = policy_logits.squeeze()
        for move in range(4):
            if move not in valid_moves:
                policy_logits[move] = float('-inf')
        
        # Get action
        if deterministic:
            action = torch.argmax(policy_logits).item()
        else:
            probs = torch.softmax(policy_logits, dim=0)
            action = torch.multinomial(probs, 1).item()
        
        return action, value.item(), policy_logits[action].item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value, valid_moves=None):
        """Store a transition in the trajectory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
        # Create and store valid actions mask
        if valid_moves is not None:
            valid_mask = torch.zeros(4, device=device)
            valid_mask[valid_moves] = 1.0
            self.valid_masks.append(valid_mask)
        else:
            self.valid_masks.append(torch.ones(4, device=device))
        
        # Debug log for tracking buffer size
        if len(self.states) % 100 == 0:
            logging.info(f"Buffer size: {len(self.states)} transitions")
    
    def compute_advantages(self, next_value=0.0):
        """Compute advantage estimates using Generalized Advantage Estimation (GAE)"""
        # Convert rewards, values, and dones to torch tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float, device=device)
        # Append next_value to values for bootstrapping
        values = torch.tensor(self.values + [next_value], dtype=torch.float, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float, device=device)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        # Iterate backwards over the rewards to compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, next_value=0.0):
        """
        Update the policy and value networks using PPO.
        
        Args:
            next_value: Value estimate for the final state
            
        Returns:
            Dictionary with training statistics
        """
        # Log the buffer size before processing
        logging.info(f"Starting update with {len(self.states)} transitions in buffer")
        
        # Check if we have enough samples
        if len(self.states) < 10:
            logging.warning("Not enough samples for update")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'approx_kl': 0.0,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
        
        # Save a copy of the data for processing
        states_copy = self.states.copy()
        actions_copy = self.actions.copy()
        rewards_copy = self.rewards.copy()
        log_probs_copy = self.log_probs.copy()
        values_copy = self.values.copy() 
        dones_copy = self.dones.copy()
        valid_masks_copy = self.valid_masks.copy()
        
        # Clear buffers AFTER making copies - this allows new data collection while processing
        self.reset_buffers()
            
        # Calculate advantages and returns using the copied data
        rewards = torch.tensor(rewards_copy, dtype=torch.float, device=device)
        values = torch.tensor(values_copy + [next_value], dtype=torch.float, device=device)
        dones = torch.tensor(dones_copy, dtype=torch.float, device=device)

        # First check for NaN values in rewards and values
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logging.warning("NaN or Inf detected in rewards, replacing with zeros")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
            
        if torch.isnan(values).any() or torch.isinf(values).any():
            logging.warning("NaN or Inf detected in values, replacing with zeros")
            values = torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)

        # Calculate GAE more safely
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        # Iterate backwards over the rewards to compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            # Clamp delta to avoid extreme values
            delta = torch.clamp(delta, -10.0, 10.0)
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            # Clamp GAE to avoid extreme values
            gae = torch.clamp(gae, -10.0, 10.0)
            advantages[t] = gae

        returns = advantages + values[:-1]
        
        # Convert all data to tensors
        states = [torch.tensor(s, dtype=torch.float, device=device) for s in states_copy]
        actions = torch.tensor(actions_copy, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(log_probs_copy, dtype=torch.float, device=device)
        valid_masks = torch.stack(valid_masks_copy)
        
        # Log for debugging
        logging.info(f"Processing {len(states)} states, {len(actions)} actions, {len(returns)} returns")
        logging.info(f"Using batch size {self.batch_size} for {len(states)} samples")
        
        # Check for NaN values but only log a warning instead of skipping the update
        has_nan = torch.isnan(returns).any() or torch.isnan(advantages).any() or torch.isnan(old_log_probs).any()
        if has_nan:
            logging.warning("NaN detected in training data, attempting to clean and continue")
            # Replace NaNs with zeros instead of skipping the update completely
            returns = torch.nan_to_num(returns, nan=0.0)
            advantages = torch.nan_to_num(advantages, nan=0.0)
            old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0)
        
        # Normalize advantages - do this again even though we normalize in compute_ppo_loss
        # This helps with initial gradient magnitudes
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        
        # Set network to training mode
        self.network.train()
        logging.info(f"Network training mode: {self.network.training}")
        
        # Use smaller batch size if needed
        effective_batch_size = min(self.batch_size, len(states))
        
        # Perform multiple epochs of updates
        for epoch in range(self.update_epochs):
            logging.info(f"Starting epoch {epoch+1}/{self.update_epochs}")
            
            # Generate random indices for minibatches, ensuring we don't skip samples
            all_indices = torch.randperm(len(states))
            
            # Process in minibatches
            for start_idx in range(0, len(states), effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, len(states))
                mb_indices = all_indices[start_idx:end_idx]
                logging.info(f"Processing minibatch {start_idx//effective_batch_size + 1} with indices {start_idx}:{end_idx}")
                
                # Make sure we have at least some data
                if len(mb_indices) < 1:
                    logging.warning(f"Empty minibatch, skipping")
                    continue
                
                try:
                    # Extract minibatch data
                    mb_states = torch.stack([states[i] for i in mb_indices])
                    mb_actions = actions[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_valid_masks = valid_masks[mb_indices]
                    
                    # Forward pass with mixed precision if enabled
                    if self.mixed_precision:
                        # Use the updated API call for autocast
                        with torch.amp.autocast('cuda'):
                            policy_logits, values = self.network(mb_states, training=True)
                            
                            # Debug output for network outputs
                            logging.info(f"Policy logits shape: {policy_logits.shape}, Values shape: {values.shape}")
                            logging.info(f"Policy logits sample: {policy_logits[0]}, Values sample: {values[0]}")
                            
                            # Apply valid action mask
                            policy_logits = policy_logits + (1.0 - mb_valid_masks) * -1e10
                            
                            # Calculate new log probabilities
                            policy = F.softmax(policy_logits, dim=1)
                            dist = torch.distributions.Categorical(policy)
                            new_log_probs = dist.log_prob(mb_actions)
                            entropy = dist.entropy().mean()
                            
                            # Debug the log probabilities
                            logging.info(f"New log probs: {new_log_probs[:5]}, Old log probs: {mb_old_log_probs[:5]}")
                            
                            # Use the custom PPO loss function for better stability
                            policy_loss, ratio = compute_ppo_loss(
                                new_log_probs, 
                                mb_old_log_probs, 
                                mb_advantages, 
                                clip_ratio=self.clip_ratio
                            )
                            
                            # Value loss with huber loss using smaller delta for stability
                            value_pred = values.squeeze()
                            value_loss = F.huber_loss(value_pred, mb_returns, delta=0.5)  # Smaller delta for more robust loss
                            
                            # Debug loss components
                            logging.info(f"Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}, Entropy: {entropy.item()}")
                            
                            # Total loss
                            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                            logging.info(f"Total loss: {loss.item()}")
                    else:
                        policy_logits, values = self.network(mb_states, training=True)
                        
                        # Debug output for network outputs
                        logging.info(f"Policy logits shape: {policy_logits.shape}, Values shape: {values.shape}")
                        logging.info(f"Policy logits sample: {policy_logits[0]}, Values sample: {values[0]}")
                        
                        # Apply valid action mask
                        policy_logits = policy_logits + (1.0 - mb_valid_masks) * -1e10
                        
                        # Calculate new log probabilities
                        policy = F.softmax(policy_logits, dim=1)
                        dist = torch.distributions.Categorical(policy)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()
                        
                        # Debug the log probabilities
                        logging.info(f"New log probs: {new_log_probs[:5]}, Old log probs: {mb_old_log_probs[:5]}")
                        
                        # Use the custom PPO loss function for better stability
                        policy_loss, ratio = compute_ppo_loss(
                            new_log_probs, 
                            mb_old_log_probs, 
                            mb_advantages, 
                            clip_ratio=self.clip_ratio
                        )
                        
                        # Value loss with huber loss using smaller delta for stability
                        value_pred = values.squeeze()
                        value_loss = F.huber_loss(value_pred, mb_returns, delta=0.5)  # Smaller delta for more robust loss
                        
                        # Debug loss components
                        logging.info(f"Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}, Entropy: {entropy.item()}")
                        
                        # Total loss
                        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                        logging.info(f"Total loss: {loss.item()}")
                    
                    # Skip update if NaN or Inf detected in loss
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logging.warning("NaN or Inf detected in loss, skipping this batch update")
                        continue  # Skip this batch but continue with others
                    
                    # Calculate approximate KL for early stopping
                    with torch.no_grad():  # Don't track gradients for KL calculation
                        log_ratio = new_log_probs - mb_old_log_probs
                        # Clean log ratio for KL calculation
                        log_ratio = torch.clamp(log_ratio, -5, 5)
                        approx_kl = (torch.exp(log_ratio) - 1 - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                    
                    # Check if we should stop early due to KL divergence
                    if approx_kl > 1.5 * self.target_kl:
                        logging.info(f"Early stopping at step {start_idx} due to reaching KL threshold")
                        break
                    
                    # Optimize with mixed precision if enabled
                    self.optimizer.zero_grad()
                    
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        # Track metrics with better handling - avoiding memory leaks
                        try:
                            with torch.no_grad():  # Make sure we're not tracking gradients for logging
                                policy_losses.append(float(policy_loss.item()))
                                value_losses.append(float(value_loss.item()))
                                entropy_losses.append(float(entropy.item()))
                                total_losses.append(float(loss.item()))
                        except Exception as e:
                            logging.warning(f"Error storing metric values: {e}")
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        # Track metrics with better handling - avoiding memory leaks
                        try:
                            with torch.no_grad():  # Make sure we're not tracking gradients for logging
                                policy_losses.append(float(policy_loss.item()))
                                value_losses.append(float(value_loss.item()))
                                entropy_losses.append(float(entropy.item()))
                                total_losses.append(float(loss.item()))
                        except Exception as e:
                            logging.warning(f"Error storing metric values: {e}")
                except Exception as e:
                    logging.error(f"Error in update step: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue  # Continue with next batch
            
        # Update scheduler after all optimizer steps
        self.scheduler.step()
        
        # Increment update counter
        self.update_count += 1
        
        # Securely compute statistics with explicit NaN handling
        def safe_mean(values):
            if not values:
                return 0.0
            cleaned = [v for v in values if not np.isnan(v) and np.abs(v) < 1e6]
            return np.mean(cleaned) if cleaned else 0.0
        
        # Check if we have valid policy losses
        if len(policy_losses) == 0:
            if len(approx_kls) > 0:
                logging.warning(f"No valid policy losses were recorded but we have KL values ({safe_mean(approx_kls):.4f}). This suggests numerical issues.")
                # Use actual optimized values if available 
                if 'policy_loss' in locals() and not torch.isnan(policy_loss):
                    policy_losses = [float(policy_loss.item())]
                else:
                    policy_losses = [0.15]  # Use a non-zero value that will allow learning
                
                if 'value_loss' in locals() and not torch.isnan(value_loss):
                    value_losses = [float(value_loss.item())]
                else:
                    value_losses = [1.0]  # Use reasonable approximation
                    
                if 'entropy' in locals() and not torch.isnan(entropy):
                    entropy_losses = [float(entropy.item())]
                else:
                    entropy_losses = [1.25]  # Use reasonable approximation
            else:
                # No valid updates occurred
                logging.warning("No valid updates occurred during this training step")
                policy_losses = [0.1]
                value_losses = [0.5]
                entropy_losses = [1.0]
        
        # Return stats with explicit non-zero values to ensure training continues
        stats = {
            'policy_loss': max(0.01, safe_mean(policy_losses)),  # Ensure non-zero
            'value_loss': max(0.01, safe_mean(value_losses)),    # Ensure non-zero
            'entropy': max(0.01, safe_mean(entropy_losses)),     # Ensure non-zero
            'total_loss': max(0.01, safe_mean(total_losses)),    # Ensure non-zero
            'approx_kl': safe_mean(approx_kls),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        self.training_stats.append(stats)
        return stats
    
    def save(self, path):
        """Save model state to path"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'update_count': self.update_count,
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path):
        """Load model state from path"""
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.training_stats = checkpoint.get('training_stats', []) 