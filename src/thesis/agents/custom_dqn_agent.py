import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from ..config import device
import logging

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * 16, 1024)
        self.dense2 = nn.Linear(1024, 4)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(F.dropout(self.dense1(x)))
        return self.dense2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class CustomDQNAgent:
    """
    Custom Deep Q-Network agent for 2048 with the specified architecture.
    """
    def __init__(self, 
                 buffer_size=50000, 
                 batch_size=64, 
                 gamma=0.99,
                 epsilon_start=0.9, 
                 epsilon_end=0.01, 
                 epsilon_decay=0.9999,
                 learning_rate=5e-5):
        
        self.device = device
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.update_count = 0
        self.target_update_freq = 10  # Update target network every 10 episodes
        
    def select_action(self, state, valid_moves=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_moves: List of valid moves
            
        Returns:
            Selected action
        """
        if valid_moves is None or len(valid_moves) == 0:
            return random.randint(0, 3)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            
            # Mask invalid actions with large negative values
            mask = torch.ones(4, device=device) * float('-inf')
            mask[valid_moves] = 0
            
            masked_q_values = q_values + mask
            return torch.argmax(masked_q_values).item()
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """Update the policy network using a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float, device=device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float, device=device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float, device=device)
        done_batch = torch.tensor(batch[4], dtype=torch.float, device=device).unsqueeze(1)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            
        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save model state to path"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
    
    def load(self, path):
        """Load model state from path"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.update_count = checkpoint.get('update_count', 0) 