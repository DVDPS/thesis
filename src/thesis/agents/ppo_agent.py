# Path: /src/thesis/agents/ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..config import device
import time

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity  # Residual connection
        x = F.relu(x)
        return x

class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important board regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attn = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))
        
        # Apply attention
        return x * attn.expand_as(x) + x  # Residual connection

class PPONetwork(nn.Module):
    """
    Advanced neural network for PPO agent with residual blocks and spatial attention.
    """
    def __init__(self, board_size=4, hidden_dim=256, input_channels=16, n_actions=4):
        super(PPONetwork, self).__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        
        # Input processing with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(4)
        ])
        
        # Spatial attention after residual blocks
        self.spatial_attention = SpatialAttention(64)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        
        # Skip connection
        self.skip_conv = nn.Conv2d(input_channels, 128, kernel_size=1)
        self.skip_bn = nn.BatchNorm2d(128)
        
        # Calculate size after convolutions
        conv_output_size = 128 * board_size * board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(device)
    
    def _init_weights(self, module):
        """Initialize weights with a stable approach"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x, training=False):
        """Forward pass through the network"""
        try:
            # Convert input to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=device)
            
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            # Ensure float type
            x = x.float()
            
            # Save input for skip connection
            skip = self.skip_bn(self.skip_conv(x))
            
            # Initial convolution
            x = F.relu(self.bn1(self.conv1(x)))
            
            # Apply residual blocks
            for res_block in self.res_blocks:
                x = res_block(x)
            
            # Apply spatial attention
            x = self.spatial_attention(x)
            
            # Second convolution
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Add skip connection
            x = x + skip
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected layers with dropout
            x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
            
            # Add exploration noise during training if requested
            if training and self.training:
                noise = 0.1 * torch.randn_like(x) * 0.1
                x = x + noise
            
            # Policy and value heads
            policy_logits = self.policy_head(x)
            value = self.value_head(x)
            
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


class PPOAgent:
    """
    Proximal Policy Optimization agent for 2048.
    - Uses actor-critic architecture
    - Employs clipped surrogate objective for stable updates
    - Optimized for H100 with mixed precision training
    """
    def __init__(self, 
                 board_size=4, 
                 hidden_dim=256, 
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
        
        # Initialize network
        self.network = PPONetwork(
            board_size=board_size,
            hidden_dim=hidden_dim,
            input_channels=input_channels
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
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
        
        # Select action
        if deterministic:
            action = torch.argmax(policy, dim=1).item()
        else:
            # Sample from distribution
            try:
                dist = torch.distributions.Categorical(policy)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
            except:
                # Fallback if distribution has issues
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
                        policy_logits, values = self.network(mb_states)
                        
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
                        
                        # Value loss
                        value_pred = values.squeeze()
                        value_loss = F.mse_loss(value_pred, mb_returns)
                        
                        # Total loss
                        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                else:
                    policy_logits, values = self.network(mb_states)
                    
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
                    
                    # Value loss
                    value_pred = values.squeeze()
                    value_loss = F.mse_loss(value_pred, mb_returns)
                    
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
            'approx_kl': np.mean(approx_kls)
        }
        
        self.training_stats.append(stats)
        return stats
    
    def save(self, path):
        """Save model state to path"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path):
        """Load model state from path"""
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.training_stats = checkpoint.get('training_stats', [])