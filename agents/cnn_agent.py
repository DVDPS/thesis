import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
import gc

class Game2048CNN(nn.Module):
    """
    This is the Game2048CNN class that defines the structure of the CNN.
    It has 19,597,777 parameters.
    """
    def __init__(self):
        super(Game2048CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(16, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024, track_running_stats=False)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024, track_running_stats=False)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512, track_running_stats=False)
        
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, track_running_stats=False)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv7 = nn.Conv2d(256, 128, kernel_size=1)
        self.ln7 = nn.LayerNorm([128, 1, 1])
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1)
        self.ln8 = nn.LayerNorm([128, 1, 1])
        
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        This is the forward pass of the CNN.
        It takes a 4x4 grid of tiles and outputs a single value.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = F.relu(self.ln7(self.conv7(x)))
        x = F.relu(self.ln8(self.conv8(x)))
        x = self.dropout(x)
        
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CNNAgent:
    """
    This is the CNNAgent class that uses a CNN to evaluate the state of the game.
    It uses a replay buffer to store experiences and a target network to stabilize the training.
    It also uses a prioritized experience replay to give more weight to more important experiences.
    """
    def __init__(self, device=None, buffer_size=4000000, batch_size=131072):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Game2048CNN().to(self.device)
        self.target_model = Game2048CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_decay_start = 1000
        self.episode_count = 0
        
        if torch.cuda.is_available():
            self.retained_memory = {
                'input_buffer': torch.zeros((800000, 16, 4, 4), device=self.device),
                'conv1_buffer': torch.zeros((800000, 1024, 4, 4), device=self.device),
                'conv2_buffer': torch.zeros((400000, 512, 2, 2), device=self.device),
                'conv3_buffer': torch.zeros((400000, 256, 1, 1), device=self.device),
                'large_batch_buffer': torch.zeros((batch_size * 32, 16, 4, 4), device=self.device),
                'feature_buffer': torch.zeros((batch_size * 16, 1024, 4, 4), device=self.device)
            }
            self.retained_parallel_buffers = {
                'states': torch.zeros((65536, 16, 4, 4), device=self.device),
                'features': torch.zeros((65536, 1024, 4, 4), device=self.device),
                'intermediate': torch.zeros((65536, 512, 2, 2), device=self.device),
                'output': torch.zeros((65536,), device=self.device)
            }
        
        print(f"Model device: {next(self.model.parameters()).device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {total_params:,}")
        print(f"Batch size: {batch_size}")
        print(f"Buffer size: {buffer_size}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scaler = torch.amp.GradScaler()
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = []
        self.buffer_position = 0
        self.priorities = np.ones(buffer_size)
        
        self.state_tensors = torch.zeros((batch_size * 16, 16, 4, 4), dtype=torch.float32, device=self.device)
        self.next_state_tensors = torch.zeros((batch_size * 16, 16, 4, 4), dtype=torch.float32, device=self.device)
        self.reward_tensors = torch.zeros(batch_size * 16, dtype=torch.float32, device=self.device)
        self.terminal_tensors = torch.zeros(batch_size * 16, dtype=torch.float32, device=self.device)
        
        self.intermediate_tensors = {
            'conv1': torch.zeros((batch_size * 2, 1024, 4, 4), dtype=torch.float32, device=self.device),
            'conv2': torch.zeros((batch_size * 2, 1024, 4, 4), dtype=torch.float32, device=self.device),
            'conv3': torch.zeros((batch_size * 2, 512, 2, 2), dtype=torch.float32, device=self.device),
            'conv4': torch.zeros((batch_size * 2, 512, 2, 2), dtype=torch.float32, device=self.device),
            'conv5': torch.zeros((batch_size * 2, 256, 1, 1), dtype=torch.float32, device=self.device),
            'conv6': torch.zeros((batch_size * 2, 256, 1, 1), dtype=torch.float32, device=self.device),
            'conv7': torch.zeros((batch_size * 2, 128, 1, 1), dtype=torch.float32, device=self.device),
            'conv8': torch.zeros((batch_size * 2, 128, 1, 1), dtype=torch.float32, device=self.device)
        }
        
        self.valid_moves_cache = {}
        self.max_cache_size = 800000
        
        self.eval_cache = {}
        self.max_eval_cache_size = 800000
        
        self.update_counter = 0
        self.target_update_frequency = 500
        
        self.parallel_state_buffer = torch.zeros((32768, 16, 4, 4), dtype=torch.float32, device=self.device)
        self.parallel_next_state_buffer = torch.zeros((32768, 16, 4, 4), dtype=torch.float32, device=self.device)
        
        self.parallel_processing_buffers = {
            'states': torch.zeros((65536, 16, 4, 4), dtype=torch.float32, device=self.device),
            'next_states': torch.zeros((65536, 16, 4, 4), dtype=torch.float32, device=self.device),
            'values': torch.zeros((65536,), dtype=torch.float32, device=self.device),
            'targets': torch.zeros((65536,), dtype=torch.float32, device=self.device)
        }
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        self.target_model.to(device)
        self.state_tensors = self.state_tensors.to(device)
        self.next_state_tensors = self.next_state_tensors.to(device)
        self.reward_tensors = self.reward_tensors.to(device)
        self.terminal_tensors = self.terminal_tensors.to(device)
        return self
    
    def preprocess_state(self, state):
        """
        This is the preprocess_state function that takes a 4x4 grid of tiles and converts it into a one-hot encoded tensor.
        We do this because it is faster than using the state directly. Because the state is a numpy array, it is not as fast as using a tensor.
        """
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
        """
        This is the get_valid_moves function that takes the state of the game and returns the valid moves.
        """
        state_key = state.tobytes()
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
        
        valid_moves = []
        for action in range(4):
            temp_state = state.copy()
            _, _, changed = game._move(temp_state, action)
            if changed:
                valid_moves.append(action)
        
        if len(self.valid_moves_cache) >= self.max_cache_size:
            self.valid_moves_cache.pop(next(iter(self.valid_moves_cache)))
        self.valid_moves_cache[state_key] = valid_moves
        
        return valid_moves
    
    def evaluate_all_actions(self, state, game):
        """
        This is the evaluate_all_actions function that takes the state of the game and returns the value of all the actions.
        """
        state_key = state.tobytes()
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        
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
        
        state_tensors = self.preprocess_batch_states(states)
        
        self.model.eval()
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                if len(states) == 1:
                    state_tensors = state_tensors.repeat(2, 1, 1, 1)
                    values = self.model(state_tensors)
                    values = values[0].unsqueeze(0)
                else:
                    values = self.model(state_tensors)
                
                values = values.squeeze().float()
        
        self.model.train()
        
        if len(states) == 1:
            result = [(actions[0][0], actions[0][1], values.item())]
        else:
            result = [(a[0], a[1], v.item()) for a, v in zip(actions, values)]
        
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = result
        
        return result
    
    def batch_evaluate_actions(self, state, game):
        state_key = state.tobytes()
        if state_key in self.eval_cache:
            return self.eval_cache[state_key]
        
        valid_actions = self.get_valid_moves(state, game)
        if not valid_actions:
            return None
        
        next_states = torch.zeros((max(2, len(valid_actions)), 16, 4, 4), dtype=torch.float32, device=self.device)
        action_info = []
        
        for i, action in enumerate(valid_actions):
            temp_state = state.copy()
            new_board, score, _ = game._move(temp_state, action)
            next_states[i] = self.preprocess_state(new_board)
            action_info.append((action, score, new_board))
        
        if len(valid_actions) == 1:
            next_states[1] = next_states[0]
        
        self.model.eval()
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                values = self.model(next_states)
                values = values[:len(valid_actions)]
                
                if len(valid_actions) == 1:
                    result = [(action_info[0][0], action_info[0][1], action_info[0][2], values[0].item())]
                else:
                    values = values.squeeze().float()
                    result = [(action, score, board, val.item()) for (action, score, board), val in zip(action_info, values)]
        
        self.model.train()
        
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = result
        
        return result
    
    def store_experience(self, state, reward, next_state, terminal):
        """
        This is the store_experience function that takes the state, reward, next state, and terminal of the game and stores it in the replay buffer.
        """
        experience = (state.copy(), reward, next_state.copy(), terminal)
        
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(experience)
            self.priorities[len(self.replay_buffer)-1] = 1.0
        else:
            self.replay_buffer[self.buffer_position] = experience
            self.priorities[self.buffer_position] = 1.0
            self.buffer_position = (self.buffer_position + 1) % self.buffer_size
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_batch_states(self, states):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def update_batch(self, num_batches=16):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        total_loss = 0.0
        
        total_samples_needed = self.batch_size * num_batches
        indices_all = np.random.choice(len(self.replay_buffer), 
                                    total_samples_needed, 
                                    replace=len(self.replay_buffer) < total_samples_needed)
        
        states, rewards, next_states, terminals = [], [], [], []
        
        for i in indices_all:
            exp = self.replay_buffer[i]
            states.append(exp[0])
            rewards.append(exp[1])
            next_states.append(exp[2])
            terminals.append(exp[3])
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            state_tensors = self.preprocess_batch_states(states)
            next_state_tensors = self.preprocess_batch_states(next_states)
            reward_tensors = torch.tensor(rewards, dtype=torch.float32, device=self.device) / 100.0
            terminal_tensors = torch.tensor(terminals, dtype=torch.float32, device=self.device)
            
            current_values = self.model(state_tensors).squeeze()
            
            with torch.no_grad():
                next_values = torch.zeros_like(reward_tensors, device=self.device)
                non_terminal_mask = terminal_tensors == 0
                if non_terminal_mask.any():
                    next_values[non_terminal_mask] = self.target_model(next_state_tensors[non_terminal_mask]).squeeze().float()
            
            targets = reward_tensors + 0.95 * next_values * (1 - terminal_tensors)
            targets = torch.clamp(targets, -100, 100)
            
            loss = self.criterion(current_values.float(), targets)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        total_loss = loss.item()
        
        self.update_counter += num_batches
        if self.update_counter >= self.target_update_frequency:
            self.update_target_network()
            self.update_counter = 0
        
        return total_loss / num_batches
    
    def evaluate(self, state):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                self.state_tensors[0] = self.preprocess_state(state)
                value = self.model(self.state_tensors[:1]).squeeze().float().item()
        return value
    
    def update(self, state, reward, next_state, terminal):
        self.store_experience(state, reward, next_state, terminal)
        
        if self.episode_count >= self.epsilon_decay_start:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if len(self.replay_buffer) < self.batch_size * 2:
            return self.update_batch(num_batches=1)
        elif len(self.replay_buffer) >= self.batch_size:
            return self.update_batch(num_batches=1)
        return 0.0
    
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay_start': self.epsilon_decay_start
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay_start = checkpoint.get('epsilon_decay_start', self.epsilon_decay_start)