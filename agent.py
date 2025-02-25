import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.squeeze(x).view(b, c)
        excitation = self.excitation(squeeze).view(b, c, 1, 1)
        return x * excitation

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([channels, 4, 4])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([channels, 4, 4])
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

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
            self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
            self.ln1 = nn.LayerNorm([128, board_size, board_size])
            self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(3)])
            self.global_pool = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1),
                nn.LayerNorm([256, board_size, board_size]),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.policy_attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
            self.value_network = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.dropout = nn.Dropout(0.1)

        self.apply(self._init_weights)
        if self.optimistic:
            self._apply_optimistic_init(Vinit)
        self.to(device)

        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.01

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def _apply_optimistic_init(self, Vinit):
        if not self.simple:
            final_layer = self.value_network[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.constant_(final_layer.bias, Vinit)

    def forward(self, x, training: bool = False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()

        if self.simple:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            policy_logits = self.policy_head(x)
            value = self.value_head(x)
        else:
            x = self.input_norm(x)
            x = F.relu(self.ln1(self.conv1(x)))
            
            for res_block in self.res_blocks:
                x = res_block(x)
            
            features = self.global_pool(x).squeeze(-1).squeeze(-1)
            
            if training and self.training and self.use_exploration_noise:
                noise = self.exploration_noise * torch.randn_like(features)
                features = features + noise
            
            policy_logits = self.policy_attention(features)
            value = self.value_network(features)

        return policy_logits, value

    def update_exploration(self, progress: float) -> None:
        effective_progress = min(progress / 0.7, 1.0)
        self.exploration_noise = max(
            self.min_exploration_noise,
            1.0 * (1 - effective_progress) ** 2
        )