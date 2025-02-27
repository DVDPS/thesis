import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

class SimpleAgent(nn.Module):
    """
    A simplified, stable agent for 2048 that will actually work.
    Uses a straightforward architecture with proper normalization.
    """
    def __init__(self, board_size=4, hidden_dim=128, input_channels=16):
        super(SimpleAgent, self).__init__()
        self.board_size = board_size
        
        # Input convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate flattened size
        self.flat_size = 64 * (board_size + 2) * (board_size + 2)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(self.flat_size, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Policy and value heads
        self.policy_head = nn.Linear(hidden_dim // 2, 4)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Apply stable initialization
        self.apply(self._init_weights)
        
        # Move to device
        self.to(device)
        
        # Exploration noise - simple linear decay
        self.exploration_noise = 0.5
        self.min_exploration_noise = 0.05
        
    def _init_weights(self, module):
        """Initialize weights with a stable approach"""
        if isinstance(module, nn.Linear):
            # Xavier initialization with a smaller gain for stability
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            # Kaiming initialization for ReLU networks
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
                
    def forward(self, x, training=False):
        """Forward pass with extensive error handling and stability measures"""
        try:
            # Convert input to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=device)
            
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
                
            # Ensure float type
            x = x.float()
            
            # Apply convolutions with batch norm and ReLU
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # First fully connected layer
            x = self.dropout1(F.relu(self.bn3(self.fc1(x))))
            
            # Second fully connected layer
            x = self.dropout2(F.relu(self.bn4(self.fc2(x))))
            
            # Add exploration noise during training if requested
            if training and self.training:
                noise = self.exploration_noise * torch.randn_like(x) * 0.1  # Reduced magnitude
                x = x + noise
                
            # Policy head
            policy_logits = self.policy_head(x)
            
            # Value head
            value = self.value_head(x)
            
            # Safety checks to prevent NaN propagation
            if torch.isnan(policy_logits).any():
                policy_logits = torch.zeros_like(policy_logits)
                
            if torch.isnan(value).any():
                value = torch.zeros_like(value)
                
            return policy_logits, value
            
        except Exception as e:
            # If anything goes wrong, return zeros
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
            return (torch.zeros((batch_size, 4), device=device), 
                    torch.zeros((batch_size, 1), device=device))
    
    def update_exploration(self, progress):
        """Simple linear decay of exploration noise"""
        self.exploration_noise = max(
            self.min_exploration_noise,
            0.5 * (1.0 - progress)
        )