import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler

class Game2048CNN(nn.Module):
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
    
    def cleanup_memory(self):
        pass
    
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
                    value = values[0].item()
                else:
                    values = self.model(state_tensors)
                    value_list = values.squeeze().cpu().numpy().tolist()
                    
                    if not isinstance(value_list, list):
                        value_list = [value_list]
        
        results = []
        for idx, (action, score) in enumerate(actions):
            value = value_list[idx] if len(states) > 1 else value
            results.append((action, score, value))
        
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = results
        
        return results
    
    def batch_evaluate_actions(self, state, game):
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
                values = self.model(state_tensors)
                
                if len(values.shape) > 1:
                    value_list = values.squeeze().cpu().numpy().tolist()
                    if not isinstance(value_list, list):
                        value_list = [value_list]
                else:
                    value_list = [values.item()]
                
                if len(value_list) != len(states):
                    value_list = value_list[:len(states)]
        
        results = []
        for idx, (action, score) in enumerate(actions):
            value = value_list[idx] if idx < len(value_list) else 0
            results.append((action, score, value))
        
        if len(self.eval_cache) >= self.max_eval_cache_size:
            self.eval_cache.pop(next(iter(self.eval_cache)))
        self.eval_cache[state_key] = results
        
        return results
    
    def store_experience(self, state, reward, next_state, terminal):
        experience = (state, reward, next_state, terminal)
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(experience)
        else:
            self.replay_buffer[self.buffer_position] = experience
            self.buffer_position = (self.buffer_position + 1) % self.buffer_size
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_batch_states(self, states):
        batch_size = len(states)
        batch_tensor = torch.zeros((batch_size, 16, 4, 4), dtype=torch.float32, device=self.device)
        
        for i, state in enumerate(states):
            batch_tensor[i] = self.preprocess_state(state)
        
        return batch_tensor
    
    def optimize_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def update_batch(self, num_batches=16):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        self.model.train()
        total_loss = 0
        
        for _ in range(num_batches):
            batch_indices = np.random.randint(0, len(self.replay_buffer), self.batch_size)
            batch = [self.replay_buffer[i] for i in batch_indices]
            
            state_batch = [b[0] for b in batch]
            reward_batch = [b[1] for b in batch]
            next_state_batch = [b[2] for b in batch]
            terminal_batch = [b[3] for b in batch]
            
            states = self.preprocess_batch_states(state_batch)
            next_states = self.preprocess_batch_states(next_state_batch)
            rewards = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
            terminals = torch.tensor(terminal_batch, dtype=torch.float32, device=self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                current_q_values = self.model(states).squeeze()
                
                with torch.no_grad():
                    next_q_values = self.target_model(next_states).squeeze()
                
                if next_q_values.dim() == 0:
                    next_q_values = next_q_values.unsqueeze(0)
                if current_q_values.dim() == 0:
                    current_q_values = current_q_values.unsqueeze(0)
                
                target_q_values = rewards + (1 - terminals) * 0.99 * next_q_values
                loss = self.criterion(current_q_values, target_q_values)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_network()
        
        return total_loss / num_batches
    
    def evaluate(self, state):
        state_tensor = self.preprocess_state(state).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                return self.model(state_tensor).item()
    
    def update(self, state, reward, next_state, terminal):
        self.store_experience(state, reward, next_state, terminal)
        
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        return self.update_batch()
    
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {path}") 