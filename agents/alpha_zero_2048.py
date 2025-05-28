import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from src.thesis.environment.game2048 import Game2048, compute_monotonicity

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        
        self.conv1 = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.policy_conv = nn.Conv2d(256, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * 4 * 4, 4)
        
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(4 * 4, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 4 * 4 * 4)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 4 * 4)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MCTSNode:
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
        self.is_terminal = False

class AlphaZeroAgent:
    def __init__(self, num_simulations=800, c_puct=1.0, use_gpu=True):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        self.model.eval()
        
        try:
            checkpoint = torch.load("alpha_zero_model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded AlphaZero model weights.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Starting with fresh model.")
        
        self.game = Game2048()
        
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        onehot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if state[i, j] > 0:
                    power = int(np.log2(state[i, j]))
                    if power < 16:
                        onehot[power, i, j] = 1.0
                else:
                    onehot[0, i, j] = 1.0
        return torch.tensor(onehot, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def to_numpy(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    
    def evaluate_state(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            policy, value = self.model(state_tensor)
            
            if isinstance(state, torch.Tensor):
                max_tile = state.max().item()
                empty_cells = torch.sum(state == 0).item()
            else:
                max_tile = np.max(state)
                empty_cells = np.sum(state == 0)
                
            monotonicity = compute_monotonicity(self.to_numpy(state))
            
            immediate_reward = (np.log2(max_tile) / 15.0) + (empty_cells / 16.0) + monotonicity
            final_value = 0.7 * value[0].item() + 0.3 * immediate_reward
            
            return policy[0].cpu().numpy(), final_value
    
    def calculate_monotonicity(self, state: np.ndarray) -> float:
        h_mono = 0
        for i in range(4):
            for j in range(3):
                if state[i, j] >= state[i, j+1]:
                    h_mono += 1
        
        v_mono = 0
        for j in range(4):
            for i in range(3):
                if state[i, j] >= state[i+1, j]:
                    v_mono += 1
        
        return (h_mono + v_mono) / 24.0
    
    def select_node(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.is_terminal:
            puct_values = {}
            total_visits = node.visits
            sqrt_total_visits = np.sqrt(total_visits)
            
            for action, child in node.children.items():
                q_value = child.value / (child.visits + 1e-8)
                puct = q_value + self.c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
                puct_values[action] = puct
            
            best_action = max(puct_values.items(), key=lambda x: x[1])[0]
            node = node.children[best_action]
        return node
    
    def expand_node(self, node: MCTSNode) -> MCTSNode:
        policy, value = self.evaluate_state(node.state)
        
        if isinstance(node.state, torch.Tensor):
            self.game.board = node.state.clone()
        else:
            self.game.board = node.state.copy()
        
        valid_moves = []
        move_values = []
        
        for action in range(4):
            new_board, score, changed = self.game._move(self.game.board, action)
            if changed:
                valid_moves.append(action)
                child_state = new_board
                child = MCTSNode(child_state, parent=node, action=action)
                child.prior = policy[action]
                
                if isinstance(new_board, torch.Tensor):
                    max_tile = new_board.max().item()
                    empty_cells = torch.sum(new_board == 0).item()
                else:
                    max_tile = np.max(new_board)
                    empty_cells = np.sum(new_board == 0)
                
                monotonicity = compute_monotonicity(self.to_numpy(new_board))
                move_value = (np.log2(max_tile) / 15.0) + (empty_cells / 16.0) + (monotonicity / 10.0)
                move_values.append(move_value)
                
                node.children[action] = child
        
        if not valid_moves:
            node.is_terminal = True
            return node
        
        best_move_idx = np.argmax(move_values)
        return node.children[valid_moves[best_move_idx]]
    
    def backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def get_move(self, state: np.ndarray) -> int:
        root = MCTSNode(state)
        
        for sim_idx in range(self.num_simulations):
            node = self.select_node(root)
            
            if not node.is_terminal and node.visits > 0:
                node = self.expand_node(node)
            
            if node.is_terminal:
                value = 0.0
            else:
                if isinstance(node.state, torch.Tensor):
                    max_tile = node.state.max().item()
                    empty_cells = torch.sum(node.state == 0).item()
                else:
                    max_tile = np.max(node.state)
                    empty_cells = np.sum(node.state == 0)
                
                monotonicity = compute_monotonicity(self.to_numpy(node.state))
                value = (np.log2(max_tile) / 15.0) + (empty_cells / 16.0) + (monotonicity / 10.0)
            
            self.backpropagate(node, value)
        
        if not root.children:
            return 0
            
        best_action = max(root.children.items(),
                         key=lambda x: (x[1].visits, x[1].value))[0]
        
        return best_action

    def save_model(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) 