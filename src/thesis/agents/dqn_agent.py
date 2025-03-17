import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from ..config import device
import logging

#############################################
# Prioritized Replay Buffer Implementation  #
#############################################
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # Validate transition components
        try:
            if not isinstance(state, (np.ndarray, list)):
                logging.error(f"Invalid state type: {type(state)}")
                return
            if not isinstance(action, (int, np.integer)):
                logging.error(f"Invalid action type: {type(action)}")
                return
            if not isinstance(reward, (float, int, np.floating, np.integer)):
                logging.error(f"Invalid reward type: {type(reward)}")
                return
            if not isinstance(next_state, (np.ndarray, list)):
                logging.error(f"Invalid next_state type: {type(next_state)}")
                return
            if not isinstance(done, bool):
                logging.error(f"Invalid done type: {type(done)}")
                return
                
            # Create and validate transition
            transition = (state, action, reward, next_state, done)
            if len(transition) != 5:
                logging.error(f"Invalid transition length: {len(transition)}")
                return
                
            max_prio = self.priorities.max() if self.buffer else 1.0
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
        except Exception as e:
            logging.error(f"Error pushing transition: {str(e)}")
            return

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Ensure priorities are positive and handle zero-sum case
        prios = np.maximum(prios, 1e-8)
        
        try:
            # Calculate probabilities with numerical stability
            probs = prios ** self.alpha
            probs_sum = np.sum(probs)
            
            if not np.isfinite(probs_sum) or probs_sum <= 0:
                probs = np.ones(len(self.buffer)) / len(self.buffer)
            else:
                probs = np.divide(probs, probs_sum, out=np.zeros_like(probs), where=probs_sum!=0)
                probs = np.nan_to_num(probs, nan=1.0/len(self.buffer))
                probs = probs / np.sum(probs)
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = np.power(len(self.buffer) * np.maximum(probs[indices], 1e-10), -beta)
            weights = weights / np.maximum(np.max(weights), 1e-10)
            
            # Validate and collect samples
            samples = []
            valid_indices = []
            for idx in indices:
                transition = self.buffer[idx]
                try:
                    if len(transition) != 5:
                        logging.error(f"Invalid transition at index {idx}: {transition}")
                        continue
                    state, action, reward, next_state, done = transition
                    if not all(x is not None for x in transition):
                        logging.error(f"None values in transition at index {idx}")
                        continue
                    samples.append(transition)
                    valid_indices.append(idx)
                except Exception as e:
                    logging.error(f"Error processing transition at index {idx}: {str(e)}")
                    continue
            
            if len(samples) != batch_size:
                logging.warning(f"Got {len(samples)} valid samples, expected {batch_size}")
                # Pad with duplicates if necessary
                while len(samples) < batch_size:
                    idx = np.random.choice(len(valid_indices))
                    samples.append(samples[idx])
                    valid_indices.append(valid_indices[idx])
            
            return samples, valid_indices, torch.tensor(weights[:len(valid_indices)], dtype=torch.float32, device=device)
            
        except Exception as e:
            logging.error(f"Error during sampling: {str(e)}")
            raise e

    def update_priorities(self, batch_indices, batch_priorities):
        try:
            for idx, prio in zip(batch_indices, batch_priorities):
                if 0 <= idx < self.capacity:
                    self.priorities[idx] = max(1e-8, prio)
                else:
                    logging.error(f"Invalid index for priority update: {idx}")
        except Exception as e:
            logging.error(f"Error updating priorities: {str(e)}")

#############################################
# Residual Block                            #
#############################################
class ResidualBlock(nn.Module):
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
        return F.relu(x)

#############################################
# Dueling DQN Agent with Prioritized Replay and Mixed Precision
#############################################
class DQNAgent(nn.Module):
    """
    Dueling Deep Q-Network agent for 2048.
    This implementation uses:
      - A dueling architecture (separate streams for state-value and advantage)
      - A prioritized replay buffer for better sample efficiency
      - Mixed precision training (using torch.cuda.amp) to reduce memory usage and enable larger batch sizes
    """
    def __init__(self, board_size=4, hidden_dim=1024, input_channels=16,
                 buffer_size=100000, batch_size=512, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 target_update_freq=1000, update_freq=4, learning_rate=0.0001,
                 is_target_network=False):
        super(DQNAgent, self).__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # increased batch size to use more GPU memory
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq  # how often to update the network
        self.update_count = 0
        self.step_count = 0
        self.is_target_network = is_target_network
        self.learning_rate = learning_rate

        # For the main network, use a prioritized replay buffer.
        if not is_target_network:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        # --- Convolutional Feature Extraction ---
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(256)
        # Increase number of residual blocks from 6 to 8
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(8)])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(512)
        self.skip_conv = nn.Conv2d(input_channels, 512, kernel_size=1)
        self.skip_bn = nn.BatchNorm2d(512)
        
        conv_output_size = 512 * board_size * board_size
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)  # Increased dropout for better regularization
        
        # --- Dueling Architecture: Advantage and Value Streams ---
        self.fc_adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 4)  # Output: advantage for each of 4 actions
        )
        self.fc_val = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)  # Output: state-value
        )

        self.apply(self._init_weights)
        self.to(device)

        # Initialize optimizer and scaler after model parameters are set up
        if not is_target_network:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            self.scaler = torch.amp.GradScaler('cuda')  # For mixed precision training
        
        # Create target network for stability
        if not is_target_network:
            self.target_network = DQNAgent(
                board_size=board_size,
                hidden_dim=hidden_dim,
                input_channels=input_channels,
                is_target_network=True
            )
            self.update_target_network()
            self.target_network.eval()

    def _init_weights(self, module):
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
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.float()
            # Save input for skip connection
            skip = self.skip_bn(self.skip_conv(x))
            x = F.relu(self.bn1(self.conv1(x)))
            for res_block in self.res_blocks:
                x = res_block(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = x + skip
            x = x.view(x.size(0), -1)
            x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
            # Dueling streams
            advantage = self.fc_adv(x)
            value = self.fc_val(x)
            # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values, value
        except Exception as e:
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
            return (torch.zeros((batch_size, 4), device=device), 
                    torch.zeros((batch_size, 1), device=device))
    
    def get_action(self, state, valid_moves=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return random.randint(0, 3)
        with torch.no_grad():
            q_values, _ = self(state)
            if valid_moves is not None:
                action_mask = torch.full((1, 4), float('-inf'), device=device)
                action_mask[0, valid_moves] = 0
                q_values = q_values + action_mask
            return torch.argmax(q_values, dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done, valid_next_moves=None):
        if self.is_target_network:
            return
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, beta=0.4, batch_size=None):
        """Update the network weights using experience replay."""
        if self.update_count % self.update_freq != 0:
            self.update_count += 1
            return None

        # Use provided batch size or default to self.batch_size
        current_batch_size = batch_size if batch_size is not None else self.batch_size
        
        if len(self.replay_buffer) < current_batch_size:
            return None
        
        try:
            # Sample from replay buffer with current batch size
            transitions, indices, weights = self.replay_buffer.sample(current_batch_size, beta)
            
            # Convert transitions to numpy arrays
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for transition in transitions:
                state, action, reward, next_state, done = transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float, device=device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
            dones = torch.tensor(np.array(dones), dtype=torch.float, device=device)
            
            # Use mixed precision for forward and loss computation
            with torch.amp.autocast('cuda'):
                current_q_values, _ = self(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values, _ = self.target_network(next_states)
                    online_q_values, _ = self(next_states)
                    best_actions = torch.argmax(online_q_values, dim=1)
                    max_next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                loss = (F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights).mean()
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.update_target_network()
                
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, td_errors)
            return loss.item()
            
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
            return None
    
    def update_target_network(self):
        if not self.is_target_network:
            # Get state dict without target network parameters
            state_dict = {}
            for name, param in self.state_dict().items():
                if not name.startswith('target_network.'):
                    state_dict[name] = param.clone().detach()
            self.target_network.load_state_dict(state_dict)
    
    def save(self, path):
        if not self.is_target_network:
            torch.save({
                'model_state_dict': self.state_dict(),
                'target_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'update_count': self.update_count,
                # Save replay buffer state
                'replay_buffer': self.replay_buffer.buffer,
                'replay_priorities': self.replay_buffer.priorities,
                'replay_pos': self.replay_buffer.pos,
                # Save GradScaler state for mixed precision training
                'scaler_state': self.scaler.state_dict()
            }, path)
    
    def load(self, path):
        if not self.is_target_network:
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                
                # Handle old checkpoint format (just model state dict)
                if not isinstance(checkpoint, dict):
                    self.load_state_dict(checkpoint)
                    print("Loaded legacy checkpoint format (model state only)")
                    return
                
                # Load model state dict
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                
                # Load target network if available
                if 'target_state_dict' in checkpoint:
                    self.target_network.load_state_dict(checkpoint['target_state_dict'])
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load other training states if available
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.update_count = checkpoint.get('update_count', 0)
                
                # Restore replay buffer state if available
                if all(k in checkpoint for k in ['replay_buffer', 'replay_priorities', 'replay_pos']):
                    self.replay_buffer.buffer = checkpoint['replay_buffer']
                    self.replay_buffer.priorities = checkpoint['replay_priorities']
                    self.replay_buffer.pos = checkpoint['replay_pos']
                
                # Restore GradScaler state if available
                if 'scaler_state' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state'])
                
                print(f"Successfully loaded checkpoint with epsilon={self.epsilon:.3f}")
            except Exception as e:
                print(f"Warning: Error during checkpoint loading: {str(e)}")
                print("Initializing with default parameters")