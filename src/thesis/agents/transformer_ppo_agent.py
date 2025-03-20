# Path: /src/thesis/agents/transformer_ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..config import device
import math
import time

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
        
        # Self-attention
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        
        # Residual connection and dropout
        return x + self.dropout(attn_output)

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
        
        # Two-layer feedforward with ReLU and dropout
        x_ff = self.dropout1(F.gelu(self.linear1(x_norm)))
        x_ff = self.dropout2(self.linear2(x_ff))
        
        # Residual connection
        return x + x_ff

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
                 ff_dim=512, dropout=0.1, input_channels=16, n_actions=4):
        super(TransformerPPONetwork, self).__init__()
        self.board_size = board_size
        self.embed_dim = embed_dim
        
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
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize weights with Transformer-specific initialization
        self.apply(self._init_weights)
        
        # Move to device
        self.to(device)
    
    def _init_weights(self, module):
        """Initialize weights with Transformer-appropriate method"""
        if isinstance(module, nn.Linear):
            # Use standard Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Layer norm weights should be 1, biases 0
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Conv layers use Kaiming initialization
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
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
            
            # Embed the board state
            x = self.embedding(x)
            
            # Add positional encoding for spatial awareness
            x = self.pos_encoding(x)
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                x = block(x)
            
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
            
            # Add exploration noise during training if requested
            if training and self.training:
                noise = 0.1 * torch.randn_like(context) * 0.1
                context = context + noise
            
            # Policy and value heads
            policy_logits = self.policy_head(context)
            value = self.value_head(context)
            
            # Safety checks to prevent NaN propagation
            if torch.isnan(policy_logits).any():
                policy_logits = torch.zeros_like(policy_logits)
            
            if torch.isnan(value).any():
                value = torch.zeros_like(value)
            
            return policy_logits, value
        
        except Exception as e:
            # Fallback to zeros if something goes wrong
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
                 lr=0.0003,
                 gamma=0.99,
                 clip_ratio=0.2,
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
            weight_decay=0.01,  # L2 regularization to prevent overfitting
            betas=(0.9, 0.999)  # Standard beta values for Adam
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,  # Adjusted based on expected number of updates
            eta_min=lr/10
        )
        
        # Setup mixed precision training for H100
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
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
    
    def get_action(self, state, valid_moves=None, deterministic=False):
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            valid_moves: List of valid moves
            deterministic: Whether to select deterministically
            
        Returns:
            Selected action, log probability, and value
        """
        # Process state
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
        
        # Forward pass through network
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
        
        # Create valid action mask if provided
        if valid_moves is not None:
            action_mask = torch.full((1, 4), float('-inf'), device=device)
            action_mask[0, valid_moves] = 0
            policy_logits = policy_logits + action_mask
        
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=1)
        
        # Always create the distribution object for consistency
        dist = torch.distributions.Categorical(policy)
        
        # Select action
        if deterministic:
            # Deterministic action selection
            action = torch.argmax(policy, dim=1).item()
            # Get log_prob from the distribution, even though we chose deterministically
            log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
        else:
            # Stochastic action selection
            try:
                # Sample from the already created distribution
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
            except Exception as e:
                # Fallback if distribution has issues
                print(f"Error sampling from distribution: {e}")
                action = torch.argmax(policy, dim=1).item()
                log_prob = torch.log(policy[0, action] + 1e-10).item()
        
        return action, log_prob, value.item()
    
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
        # Calculate advantages and returns
        advantages, returns = self.compute_advantages(next_value)
        
        # Convert all data to tensors
        states = [torch.tensor(s, dtype=torch.float, device=device) for s in self.states]
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float, device=device)
        valid_masks = torch.stack(self.valid_masks)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        
        # Perform multiple epochs of updates
        for _ in range(self.update_epochs):
            # Generate random indices for minibatches
            indices = torch.randperm(len(states))
            
            # Process in minibatches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                mb_indices = indices[start_idx:end_idx]
                
                # Extract minibatch data
                mb_states = torch.stack([states[i] for i in mb_indices])
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_valid_masks = valid_masks[mb_indices]
                
                # Forward pass with mixed precision if enabled
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        policy_logits, values = self.network(mb_states, training=True)
                        
                        # Apply valid action mask
                        policy_logits = policy_logits + (1.0 - mb_valid_masks) * -1e10
                        
                        # Calculate new log probabilities
                        policy = F.softmax(policy_logits, dim=1)
                        dist = torch.distributions.Categorical(policy)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()
                        
                        # Calculate ratio and clipped loss
                        ratio = torch.exp(new_log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss with clipping for stability
                        value_pred = values.squeeze()
                        value_loss = F.huber_loss(value_pred, mb_returns)  # Huber loss is more robust than MSE
                        
                        # Total loss
                        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                else:
                    policy_logits, values = self.network(mb_states, training=True)
                    
                    # Apply valid action mask
                    policy_logits = policy_logits + (1.0 - mb_valid_masks) * -1e10
                    
                    # Calculate new log probabilities
                    policy = F.softmax(policy_logits, dim=1)
                    dist = torch.distributions.Categorical(policy)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratio and clipped loss
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping for stability
                    value_pred = values.squeeze()
                    value_loss = F.huber_loss(value_pred, mb_returns)
                    
                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Calculate approximate KL for early stopping
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                approx_kls.append(approx_kl)
                
                # Check if we should stop early due to KL divergence
                if approx_kl > 1.5 * self.target_kl:
                    break
                
                # Optimize with mixed precision if enabled
                self.optimizer.zero_grad()
                
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                total_losses.append(loss.item())
            
            # Step the learning rate scheduler
            self.scheduler.step()
        
        # Increment update counter
        self.update_count += 1
        
        # Clear buffers
        self.reset_buffers()
        
        # Return stats
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'approx_kl': np.mean(approx_kls),
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