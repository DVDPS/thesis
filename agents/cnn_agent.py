import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler

class Game2048CNN(nn.Module):
    def __init__(self):
        super(Game2048CNN, self).__init__()
        
        # Input: 4x4x16 (one-hot encoded board)
        # Use 'same' padding to maintain spatial dimensions
        self.conv1 = nn.Conv2d(16, 128, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, padding='same')
        
        # Add pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2)
        
        # Additional convolutional layers for pattern recognition
        self.conv3 = nn.Conv2d(128, 64, kernel_size=2, padding='same')
        self.conv4 = nn.Conv2d(64, 32, kernel_size=2, padding='same')
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 2 * 2, 256)  # Adjusted for pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: single value for state evaluation
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Input shape: (batch_size, 16, 4, 4)
        
        # First convolutional block with pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Reduces spatial dimensions by half
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 2 * 2)  # Adjusted for pooling
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output shape: (batch_size, 1)
        
        return x

class CNNAgent:
    def __init__(self, device=None, buffer_size=1000000, batch_size=1024):  # Increased default batch size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Game2048CNN().to(self.device)
        self.target_model = Game2048CNN().to(self.device)  # Add target network
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Force aggressive GPU memory allocation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Request a larger chunk of memory upfront to avoid fragmentation
            temp = torch.zeros((10000, 16, 4, 4), device=self.device)  # Allocate a large tensor
            del temp  # Release the tensor but keep the memory allocation
            torch.cuda.empty_cache()  # Clean up fragmented memory
        
        # Print model device and parameters
        print(f"Model device: {next(self.model.parameters()).device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {total_params:,}")
        
        # Optimizer with fixed learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler()
        
        # Experience replay buffer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = []
        self.buffer_position = 0
        
        # Priority weights for sampling
        self.priorities = np.ones(buffer_size)
        
        # Pre-allocate tensors for batch processing
        self.state_tensors = torch.zeros((batch_size, 16, 4, 4), dtype=torch.float32, device=self.device)
        self.next_state_tensors = torch.zeros((batch_size, 16, 4, 4), dtype=torch.float32, device=self.device)
        self.reward_tensors = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.terminal_tensors = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        # Valid moves cache with size limit
        self.valid_moves_cache = {}
        self.max_cache_size = 10000
        
        # Action evaluation cache
        self.eval_cache = {}
        self.max_eval_cache_size = 10000
        
        # Target network update counter
        self.update_counter = 0
        self.target_update_frequency = 1000  # Update target network every 1000 steps
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        self.target_model.to(device)
        # Move pre-allocated tensors to device
        self.state_tensors = self.state_tensors.to(device)
        self.next_state_tensors = self.next_state_tensors.to(device)
        self.reward_tensors = self.reward_tensors.to(device)
        self.terminal_tensors = self.terminal_tensors.to(device)
        return self
    
    def preprocess_state(self, state):
        """Convert board state to one-hot representation"""
        # Create one-hot encoding (16 channels for values 0 to 2^15)
        onehot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if state[i, j] > 0:
                    power = int(np.log2(state[i, j]))
                    if power < 16:
                        onehot[power, i, j] = 1.0
                else:
                    onehot[0, i, j] = 1.0
        return torch.tensor(onehot, dtype=torch.float32, device=self.device)
    
    def get_valid_moves(self, state, game):
        """Get all valid moves for a state with cache size limit"""
        state_key = state.tobytes()
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
        
        valid_moves = []
        for action in range(4):
            temp_state = state.copy()
            _, _, changed = game._move(temp_state, action)
            if changed:
                valid_moves.append(action)
        
        # Add to cache with size limit
        if len(self.valid_moves_cache) >= self.max_cache_size:
            # Remove oldest entry
            self.valid_moves_cache.pop(next(iter(self.valid_moves_cache)))
        self.valid_moves_cache[state_key] = valid_moves
        
        return valid_moves
    
    def evaluate_all_actions(self, state, game):
        """Evaluate all four actions in one batch"""
        # Try to get from cache first
        state_key = state.tobytes()
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        
        # Create states for all four possible actions
        states = []
        actions = []
        
        for action in range(4):
            temp_state = state.copy()
            new_board, score, changed = game._move(temp_state, action)
            if changed:
                states.append(new_board)
                actions.append((action, score))
        
        if not states:
            return None
        
        # Process all states in one batch
        state_tensors = self.preprocess_batch_states(states)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                values = self.model(state_tensors).squeeze().float()
        
        # Create result
        if len(states) == 1:
            result = [(actions[0][0], actions[0][1], values.item())]
        else:
            result = [(a[0], a[1], v.item()) for a, v in zip(actions, values)]
        
        # Cache result
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = result
        
        return result
    
    def batch_evaluate_actions(self, state, game):
        """Evaluate all possible actions in a single forward pass"""
        # Try to get from cache first
        state_key = state.tobytes()
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        
        # Get valid moves
        valid_actions = self.get_valid_moves(state, game)
        if not valid_actions:
            return None
        
        # Prepare tensor for all possible next states
        next_states = torch.zeros((len(valid_actions), 16, 4, 4), dtype=torch.float32, device=self.device)
        action_info = []
        
        # Generate all possible next states
        for i, action in enumerate(valid_actions):
            temp_state = state.copy()
            new_board, score, _ = game._move(temp_state, action)
            next_states[i] = self.preprocess_state(new_board)
            action_info.append((action, score, new_board))
        
        # Evaluate all states in a single forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                values = self.model(next_states)
                
                # Handle the case when there's only one action (becomes 0-d tensor after squeeze)
                if len(valid_actions) == 1:
                    # If there's only one action, handle as scalar
                    result = [(action_info[0][0], action_info[0][1], action_info[0][2], values.item())]
                else:
                    # Multiple actions - squeeze safely and convert to float
                    values = values.squeeze().float()
                    result = [(action, score, board, val.item()) for (action, score, board), val in zip(action_info, values)]
        
        # Cache result
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = result
        
        return result
    
    def store_experience(self, state, reward, next_state, terminal):
        """Store experience with priority"""
        experience = (state.copy(), reward, next_state.copy(), terminal)
        
        # Add to buffer with highest priority
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(experience)
            self.priorities[len(self.replay_buffer)-1] = 1.0  # New samples get high priority
        else:
            self.replay_buffer[self.buffer_position] = experience
            self.priorities[self.buffer_position] = 1.0  # Reset priority for this position
            self.buffer_position = (self.buffer_position + 1) % self.buffer_size
    
    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_batch_states(self, states):
        """Efficiently preprocess multiple states at once"""
        batch_size = len(states)
        onehot = np.zeros((batch_size, 16, 4, 4), dtype=np.float32)
        
        for b, state in enumerate(states):
            for i in range(4):
                for j in range(4):
                    if state[i, j] > 0:
                        power = int(np.log2(state[i, j]))
                        if power < 16:
                            onehot[b, power, i, j] = 1.0
                    else:
                        onehot[b, 0, i, j] = 1.0
        
        return torch.tensor(onehot, dtype=torch.float32, device=self.device)
    
    def optimize_memory(self):
        """Call periodically to optimize GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()
    
    def update_batch(self, num_batches=1):
        """Update the network using multiple batches"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        total_loss = 0.0
        
        # Determine if we need to use replacement sampling
        total_samples_needed = self.batch_size * num_batches
        if total_samples_needed > len(self.replay_buffer):
            # Use replacement sampling
            replacement = True
        else:
            replacement = False
        
        # Process all batches in one go
        indices_all = np.random.choice(len(self.replay_buffer), 
                                    total_samples_needed, 
                                    replace=replacement)
        
        for batch_idx in range(num_batches):
            # Get batch indices
            batch_indices = indices_all[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            states, rewards, next_states, terminals = [], [], [], []
            
            # Collect batch data
            for i in batch_indices:
                exp = self.replay_buffer[i]
                states.append(exp[0])
                rewards.append(exp[1])
                next_states.append(exp[2])
                terminals.append(exp[3])
            
            # Process whole batch at once
            state_tensors = self.preprocess_batch_states(states)
            next_state_tensors = self.preprocess_batch_states(next_states)
            reward_tensors = torch.tensor(rewards, dtype=torch.float32, device=self.device) / 100.0
            terminal_tensors = torch.tensor(terminals, dtype=torch.float32, device=self.device)
            
            # Get current values with mixed precision
            self.model.train()
            with torch.amp.autocast(device_type='cuda'):
                current_values = self.model(state_tensors).squeeze()
                
                # Get next values using target network
                with torch.no_grad():
                    next_values = torch.zeros_like(reward_tensors, device=self.device)
                    non_terminal_mask = terminal_tensors == 0
                    if non_terminal_mask.any():
                        next_values[non_terminal_mask] = self.target_model(next_state_tensors[non_terminal_mask]).squeeze().float()
                
                # Calculate TD targets with gradient clipping
                targets = reward_tensors + 0.95 * next_values * (1 - terminal_tensors)
                targets = torch.clamp(targets, -100, 100)  # Clip target values to match scaled rewards
                
                # Calculate loss
                loss = self.criterion(current_values, targets)
            
            # Update with gradient scaling and clipping
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update target network if needed
            self.update_counter += 1
            if self.update_counter >= self.target_update_frequency:
                self.update_target_network()
                self.update_counter = 0
        
        return total_loss / num_batches
    
    def evaluate(self, state):
        """Evaluate state value using the CNN"""
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                # Use first position in pre-allocated tensor for evaluation
                self.state_tensors[0] = self.preprocess_state(state)
                value = self.model(self.state_tensors[:1]).squeeze().float().item()
        return value
    
    def update(self, state, reward, next_state, terminal):
        """Store experience in replay buffer and update if enough samples"""
        self.store_experience(state, reward, next_state, terminal)
        
        # Update more frequently when buffer is small
        if len(self.replay_buffer) < self.batch_size * 2:
            return self.update_batch(num_batches=1)  # Single batch when buffer is small
        elif len(self.replay_buffer) >= self.batch_size:
            return self.update_batch(num_batches=1)  # Single batch when buffer is large enough
        return 0.0
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device)) 