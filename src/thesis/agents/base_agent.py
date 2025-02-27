import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

class ResidualBlock(nn.Module):
    """Residual block with normalization and parametric ReLU"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.gn1 = nn.GroupNorm(4, channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.gn2 = nn.GroupNorm(4, channels)
        self.prelu2 = nn.PReLU(channels)
        
    def forward(self, x):
        residual = x
        x = self.prelu1(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x += residual
        return self.prelu2(x)

class AttentionLayer(nn.Module):
    """Simple self-attention mechanism for the game board"""
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Reshape for attention calculation
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

class PPOAgent(nn.Module):
    def __init__(self, board_size: int = 4, hidden_dim: int = 256, simple: bool = False, 
                 use_exploration_noise: bool = True, input_channels: int = 1,
                 optimistic: bool = False, Vinit: float = 320000.0):
        super(PPOAgent, self).__init__()
        self.board_size = board_size
        self.use_exploration_noise = use_exploration_noise
        self.simple = simple
        self.optimistic = optimistic
        self.input_channels = input_channels

        if self.simple:
            input_size = board_size * board_size * input_channels
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.policy_head = nn.Linear(64, 4)
            self.value_head = nn.Linear(64, 1)
        else:
            self.input_norm = nn.LayerNorm([input_channels, board_size, board_size])
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding='same')
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
            self.ln1 = nn.GroupNorm(4, 32)
            self.ln2 = nn.GroupNorm(4, 64)
            self.shortcut = nn.Conv2d(input_channels, 64, kernel_size=1)
            self.fc1 = nn.Linear(64 * board_size * board_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.dropout = nn.Dropout(0.15)
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 4)
            )
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        self.apply(self._init_weights)
        if self.optimistic:
            self._apply_optimistic_init(Vinit)
        self.to(device)

        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.05

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def _apply_optimistic_init(self, Vinit):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, Vinit)
        if self.simple:
            init_layer(self.value_head)
        else:
            for layer in self.value_head:
                init_layer(layer)

    def forward(self, x, training: bool = False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        if self.simple:
            x = x.view(x.size(0), -1)
        else:
            x = self.input_norm(x)
            identity = self.shortcut(x)
            x = F.relu(self.ln1(self.conv1(x)))
            x = F.relu(self.ln2(self.conv2(x)))
            x = x + identity
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if training and self.training and self.use_exploration_noise:
            noise = self.exploration_noise * torch.randn_like(x)
            x = x + noise
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def update_exploration(self, progress: float) -> None:
        effective_progress = min(progress / 0.8, 1.0)
        self.exploration_noise = max(self.min_exploration_noise, 1.0 * (1 - 0.4 * effective_progress))

class EnhancedPPOAgent(nn.Module):
    def __init__(self, board_size=4, hidden_dim=256, input_channels=16, Vinit=320000.0):
        super(EnhancedPPOAgent, self).__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        
        # Input processing
        self.input_norm = nn.LayerNorm([input_channels, board_size, board_size])
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.gn1 = nn.GroupNorm(4, 64)
        self.prelu1 = nn.PReLU(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(4)  # Increased depth
        ])
        
        # Attention mechanism
        self.attention = AttentionLayer(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * board_size * board_size, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4)  # 4 possible actions
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply optimistic initialization to value head
        self._apply_optimistic_init(Vinit)
        
        # Move to device
        self.to(device)
        
        # Exploration parameters
        self.exploration_noise = 1.5  # Higher initial noise
        self.min_exploration_noise = 0.1  # Higher minimum noise
        self.exploration_decay_fraction = 0.7  # Slower decay (70% of training)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def _apply_optimistic_init(self, Vinit):
        for module in self.value_head:
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, Vinit / len(self.value_head))
                
    def forward(self, x, training=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        
        # Input normalization
        x = self.input_norm(x)
        
        # First conv layer
        x = self.prelu1(self.gn1(self.conv1(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Attention mechanism
        x = self.attention(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.ln1(F.relu(self.fc1(x))))
        x = self.dropout2(self.ln2(F.relu(self.fc2(x))))
        
        # Apply exploration noise if in training mode
        if training and self.training:
            noise = self.exploration_noise * torch.randn_like(x)
            x = x + noise
            
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
        
    def update_exploration(self, progress):
        """
        Update exploration noise with slower decay schedule.
        """
        if progress < self.exploration_decay_fraction:
            # Slower linear decay during the majority of training
            decay_progress = progress / self.exploration_decay_fraction
            self.exploration_noise = max(
                self.min_exploration_noise,
                1.5 * (1 - 0.8 * decay_progress)  # Higher multiplier and slower decay
            )
        else:
            # Hold at minimum for the remainder of training
            self.exploration_noise = self.min_exploration_noise