import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from torch.amp import autocast

class LargeGame2048CNN(nn.Module):
    def __init__(self):
        super(LargeGame2048CNN, self).__init__()
        
        # Larger network for H100
        self.conv1 = nn.Conv2d(16, 512, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(512, 512, kernel_size=2, padding='same')
        
        # Add pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2)
        
        # Additional convolutional layers
        self.conv3 = nn.Conv2d(512, 256, kernel_size=2, padding='same')
        self.conv4 = nn.Conv2d(256, 128, kernel_size=2, padding='same')
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)  # Larger FC layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # State value estimation
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # First convolutional block with pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ParallelCNNAgent:
    def __init__(self, device=None, buffer_size=500000, batch_size=16384):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LargeGame2048CNN().to(self.device)
        self.target_model = LargeGame2048CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Aggressive GPU memory allocation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Pre-allocating GPU memory...")
            temp = torch.zeros((100000, 16, 4, 4), device=self.device)
            del temp
            torch.cuda.empty_cache()
            print(f"Memory pre-allocation complete.")
        
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
        
        # Action evaluation cache
        self.eval_cache = {}
        self.max_eval_cache_size = 10000
        
        # Target network update counter
        self.update_counter = 0
        self.target_update_frequency = 1000
    
    def preprocess_batch_states(self, states):
        """Process multiple states at once into one-hot encodings"""
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        batch_size = states.shape[0]
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
    
    def preprocess_state(self, state):
        """Convert a single board state to one-hot representation"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
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
    
    def batch_evaluate_actions(self, states, parallel_game):
        """Evaluate all possible actions in a single forward pass"""
        # Try to get from cache first
        state_key = states.tobytes()
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        
        # Get valid moves
        valid_actions = parallel_game.get_valid_moves(0)  # Get moves for first environment
        if not valid_actions:
            return None
        
        # Prepare tensor for all possible next states
        next_states = torch.zeros((len(valid_actions), 16, 4, 4), dtype=torch.float32, device=self.device)
        action_info = []
        
        # Generate all possible next states
        for i, action in enumerate(valid_actions):
            temp_state = states[0].copy()  # Use first state from batch
            new_board, score, _ = parallel_game._move(temp_state, action)
            next_states[i] = self.preprocess_state(new_board)
            action_info.append((action, score, new_board))
        
        # Evaluate all states in a single forward pass
        with torch.no_grad():
            with autocast('cuda', dtype=torch.float16):
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
        experience = (state, reward, next_state, terminal)
        
        # Add to buffer with highest priority
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(experience)
            self.priorities[len(self.replay_buffer)-1] = 1.0
        else:
            self.replay_buffer[self.buffer_position] = experience
            self.priorities[self.buffer_position] = 1.0
            self.buffer_position = (self.buffer_position + 1) % self.buffer_size
    
    def store_batch_experience(self, states, rewards, next_states, terminals):
        """Store a batch of experiences efficiently"""
        batch_size = len(states)
        for i in range(batch_size):
            self.store_experience(states[i], rewards[i], next_states[i], terminals[i])
    
    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def update_batch(self, num_batches=4):
        """Train on multiple large batches"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        total_loss = 0.0
        
        # Determine if we need to use replacement sampling
        total_samples_needed = self.batch_size * num_batches
        replacement = total_samples_needed > len(self.replay_buffer)
        
        # Process all batches in one go
        indices_all = np.random.choice(len(self.replay_buffer), total_samples_needed, replace=replacement)
        
        for batch_idx in range(num_batches):
            # Extract batch data
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            batch_indices = indices_all[batch_start:batch_end]
            
            # Collect batch data (more efficiently)
            states = []
            rewards = []
            next_states = []
            terminals = []
            
            for i in batch_indices:
                exp = self.replay_buffer[i]
                states.append(exp[0])
                rewards.append(exp[1])
                next_states.append(exp[2])
                terminals.append(exp[3])
            
            # Convert to tensors directly for large batch processing
            state_tensors = self.preprocess_batch_states(np.array(states))
            next_state_tensors = self.preprocess_batch_states(np.array(next_states))
            reward_tensors = torch.tensor(rewards, dtype=torch.float32, device=self.device) / 100.0
            terminal_tensors = torch.tensor(terminals, dtype=torch.float32, device=self.device)
            
            # Use mixed precision training for better performance
            self.model.train()
            with autocast('cuda', dtype=torch.float16):
                current_values = self.model(state_tensors).squeeze()
                
                # Get next values using target network
                with torch.no_grad():
                    next_values = torch.zeros_like(reward_tensors, device=self.device)
                    non_terminal_mask = terminal_tensors == 0
                    if non_terminal_mask.any():
                        next_values[non_terminal_mask] = self.target_model(
                            next_state_tensors[non_terminal_mask]
                        ).squeeze().float()
                
                # Calculate TD targets
                targets = reward_tensors + 0.95 * next_values * (1 - terminal_tensors)
                targets = torch.clamp(targets, -100, 100)
                
                # Calculate loss
                loss = self.criterion(current_values, targets)
            
            # Update with gradient scaling for mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
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
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        # Update target network
        self.target_model.load_state_dict(self.model.state_dict())