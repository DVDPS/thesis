import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

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

class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing board-wide patterns"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Create query, key, value projections
        query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # B x HW x C/4
        key = self.key(x).view(batch_size, -1, H*W)  # B x C/4 x HW
        value = self.value(x).view(batch_size, -1, H*W)  # B x C x HW
        
        # Calculate attention scores
        energy = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)
        
        # Add weighted attention to input (residual connection)
        out = self.gamma * out + x
        
        return out

class EnhancedAgent(nn.Module):
    """
    Simplified version of the enhanced agent for 2048.
    - Keeps residual connections for better gradient flow
    - Removes the complex attention mechanism
    - Reduces overall model complexity
    - Maintains batch normalization for training stability
    """
    def __init__(self, board_size=4, hidden_dim=256, input_channels=16):
        super(EnhancedAgent, self).__init__()
        self.board_size = board_size
        
        # Input processing with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        
        # Reduced number of residual blocks (from 4 to 2)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(2)  # Reduced from 4 to 2 blocks
        ])
        
        # Removed self-attention mechanism
        
        # Second convolutional layer 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        
        # Calculate size after convolutions
        conv_output_size = 128 * board_size * board_size
        
        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)  # Reduced dropout from 0.2 to 0.1
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.1)  # Reduced dropout from 0.2 to 0.1
        
        # Policy head - predict action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4)
        )
        
        # Value head - predict state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights for stability
        self.apply(self._init_weights)
        
        # Move to device
        self.to(device)
        
        # Increased exploration parameters for better exploration
        self.exploration_noise = 0.75  # Increased from 0.5
        self.min_exploration_noise = 0.1  # Increased from 0.05
        
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
        """Forward pass with residual connections but without attention"""
        try:
            # Convert input to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=device)
            
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
                
            # Ensure float type
            x = x.float()
            
            # Initial convolution
            x = F.relu(self.bn1(self.conv1(x)))
            
            # Apply residual blocks
            for res_block in self.res_blocks:
                x = res_block(x)
            
            # Removed self-attention block
            
            # Second convolution
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected layers
            x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
            
            # Add exploration noise during training if requested
            if training and self.training:
                noise = self.exploration_noise * torch.randn_like(x) * 0.1
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
    
    def update_exploration(self, progress):
        """Update exploration noise with a slower decay schedule"""
        # Slower decay to encourage more exploration
        self.exploration_noise = max(
            self.min_exploration_noise,
            0.75 * (1.0 - progress * 0.6)  # Start higher, decay more slowly (was 0.5 and 0.8)
        )