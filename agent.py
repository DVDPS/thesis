import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import device

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
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding='same')
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
            self.ln1 = nn.GroupNorm(8, 64)
            self.ln2 = nn.GroupNorm(8, 128)
            self.ln3 = nn.GroupNorm(8, 128)
            self.shortcut = nn.Conv2d(input_channels, 128, kernel_size=1)
            self.fc1 = nn.Linear(128 * board_size * board_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.dropout = nn.Dropout(0.1)
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
        # Apply optimistic initialization if flag is set.
        if self.optimistic:
            self._apply_optimistic_init(Vinit)
        self.to(device)

        # Exploration noise parameters.
        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.01

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def _apply_optimistic_init(self, Vinit):
        # For each linear layer in the value head, set the bias to Vinit.
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
            x = F.relu(self.ln3(self.conv3(x))) + identity
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if self.simple:
            x = x.view(x.size(0), -1)
        else:
            if training and self.training and self.use_exploration_noise:
                noise = self.exploration_noise * torch.randn_like(x)
                x = x + noise
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


    def update_exploration(self, progress: float) -> None:
    # For example, decay noise linearly over 70% of training instead of 100%
        effective_progress = min(progress / 0.7, 1.0)
        self.exploration_noise = max(self.min_exploration_noise, 1.0 * (1 - 0.5 * effective_progress))