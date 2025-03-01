import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from ..config import device

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

class DQNAgent(nn.Module):
    """
    Deep Q-Network agent for 2048.
    - Uses a convolutional network with residual connections
    - Maintains compatibility with MCTS framework by providing policy and value outputs
    - Implements experience replay and target network for stable learning
    """
    def __init__(self, board_size=4, hidden_dim=256, input_channels=16, buffer_size=100000, 
                 batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, 
                 epsilon_decay=0.995, target_update_freq=1000, update_freq=4,
                 learning_rate=0.0001, is_target_network=False):
        super(DQNAgent, self).__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq  # How often to update the network
        self.update_count = 0
        self.step_count = 0  # Count of environment steps
        self.is_target_network = is_target_network
        self.learning_rate = learning_rate
        
        # Experience replay buffer (only for main network)
        if not is_target_network:
            self.replay_buffer = deque(maxlen=buffer_size)
        
        # Input processing with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(4)  # Increased from 3 to 4 blocks
        ])
        
        # Second convolutional layer with increased channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        
        # Skip connection processing
        self.skip_conv = nn.Conv2d(input_channels, 128, kernel_size=1)
        self.skip_bn = nn.BatchNorm2d(128)
        
        # Calculate size after convolutions
        conv_output_size = 128 * board_size * board_size
        
        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)  # Increased from 0.15
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)  # Increased from 0.15
        
        # Q-value head - predict action values
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4)  # 4 actions: UP, RIGHT, DOWN, LEFT
        )
        
        # Value head - for compatibility with MCTS
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
        
        # Create target network (only for main network)
        if not is_target_network:
            self.target_network = DQNAgent(
                board_size=board_size,
                hidden_dim=hidden_dim,
                input_channels=input_channels,
                is_target_network=True  # Prevent infinite recursion
            )
            self.update_target_network()  # Initialize with same weights
            self.target_network.eval()  # Target network is only used for inference
            
            # Create optimizer (only for main network, after all parameters are created)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def _init_weights(self, module):
        """Initialize weights with a stable approach"""
        if isinstance(module, nn.Linear):
            # Xavier initialization for fully connected layers
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            # Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Standard initialization for batch norm
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
            
            # Second convolution
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Add skip connection
            x = x + skip
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected layers with dropout
            x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
            
            # Q-values and state value
            q_values = self.q_head(x)
            state_value = self.value_head(x)
            
            # For MCTS compatibility, return q_values as policy logits and state_value
            return q_values, state_value
            
        except Exception as e:
            # Fallback to zeros if something goes wrong
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
            return (torch.zeros((batch_size, 4), device=device), 
                    torch.zeros((batch_size, 1), device=device))
    
    def get_action(self, state, valid_moves=None, epsilon=None):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            valid_moves: List of valid moves
            epsilon: Override for exploration rate
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Exploration
        if random.random() < epsilon:
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return random.randint(0, 3)
        
        # Exploitation
        with torch.no_grad():
            q_values, _ = self(state)
            
            # Apply action mask if valid_moves provided
            if valid_moves is not None:
                action_mask = torch.full((1, 4), float('-inf'), device=device)
                action_mask[0, valid_moves] = 0
                q_values = q_values + action_mask
                
            return torch.argmax(q_values, dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done, valid_next_moves=None):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            valid_next_moves: Valid moves for next state
        """
        if self.is_target_network:
            return  # Target network doesn't store transitions
            
        # Convert tensors to numpy arrays for storage
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Store transition
        self.replay_buffer.append((state, action, reward, next_state, done, valid_next_moves))
    
    def update(self):
        """Update the network using a batch from replay buffer"""
        if self.is_target_network:
            return 0.0  # Target network doesn't update
            
        # Only update every update_freq steps
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            return 0.0
            
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples
            
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, valid_next_moves_batch = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float, device=device)
        
        # Get current Q values
        current_q_values, _ = self(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            
            # Apply action mask for valid moves
            if valid_next_moves_batch[0] is not None:
                for i, valid_moves in enumerate(valid_next_moves_batch):
                    if valid_moves:
                        # Create mask for invalid moves
                        mask = torch.full((4,), float('-inf'), device=device)
                        mask[valid_moves] = 0
                        next_q_values[i] = next_q_values[i] + mask
            
            # Double DQN: Use online network to select actions
            online_q_values, _ = self(next_states)
            best_actions = torch.argmax(online_q_values, dim=1)
            
            # Use target network to evaluate those actions
            max_next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Calculate loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        if not self.is_target_network:  # Only main network updates target
            # Get state dict without target network parameters
            state_dict = {}
            for name, param in self.state_dict().items():
                if not name.startswith('target_network.'):
                    state_dict[name] = param
            # Load state dict into target network
            self.target_network.load_state_dict(state_dict)
        
    def save(self, path):
        """Save model to path"""
        if not self.is_target_network:  # Only main network saves
            torch.save({
                'model_state_dict': self.state_dict(),
                'target_state_dict': self.target_network.state_dict(),
                'epsilon': self.epsilon,
                'update_count': self.update_count
            }, path)
        
    def load(self, path):
        """Load model from path"""
        if not self.is_target_network:  # Only main network loads
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.update_count = checkpoint.get('update_count', 0)