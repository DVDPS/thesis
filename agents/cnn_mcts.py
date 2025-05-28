import numpy as np
import torch
from typing import Dict, List, Tuple
from .cnn_agent import Game2048CNN
from src.thesis.environment.game2048 import Game2048

class MCTSNode:
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False

class CNNMCTSAgent:
    def __init__(self, num_simulations=100, exploration_constant=1.41, use_gpu=True):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = Game2048CNN().to(self.device)
        self.model.eval()
        
        try:
            checkpoint = torch.load("best_cnn_model.pth", map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Successfully loaded CNN model weights.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        self.batch_size = 8192
        self.parallel_batches = 4
        self.warmup_steps = 100
        
        self.state_tensors = [
            torch.zeros((self.batch_size, 16, 4, 4),
                       dtype=torch.float32).to(self.device)
            for _ in range(self.parallel_batches)
        ]
        
        self.state_buffers = [[] for _ in range(self.parallel_batches)]
        self.current_buffer = 0
        
        self.temp_storage = {
            'features': torch.zeros((self.batch_size * self.parallel_batches, 512, 4, 4),
                                  dtype=torch.float32).to(self.device),
            'intermediate': torch.zeros((self.batch_size * self.parallel_batches, 256, 2, 2),
                                     dtype=torch.float32).to(self.device),
            'output': torch.zeros((self.batch_size * self.parallel_batches,),
                                dtype=torch.float32).to(self.device)
        }
        
        self.cuda_graphs = {}
        if use_gpu and torch.cuda.is_available():
            self._init_cuda_graphs()
            
        self.streams = [torch.cuda.Stream() for _ in range(self.parallel_batches)] if use_gpu and torch.cuda.is_available() else []

    def _init_cuda_graphs(self):
        print("Initializing CUDA graphs for optimized processing...")
        
        sample_input = torch.zeros((self.batch_size, 16, 4, 4),
                                 dtype=torch.float32,
                                 device=self.device)
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        _ = self.model(sample_input)
        torch.cuda.current_stream().wait_stream(s)
        
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    self.static_output = self.model(sample_input)
        
        self.cuda_graphs['forward'] = g
        self.static_input = sample_input
        print("CUDA graphs initialized successfully.")

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        onehot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = state[i, j]
                if val > 0:
                    power = int(np.log2(val))
                    if power < 16:
                        onehot[power, i, j] = 1.0
                else:
                    onehot[0, i, j] = 1.0
        return torch.tensor(onehot, dtype=torch.float32, device=self.device)

    def evaluate_state(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            value = self.model(state_tensor).item()
        return value

    def evaluate_states_batch(self, states: List[np.ndarray]) -> List[float]:
        if not states:
            return []
        
        values = []
        num_states = len(states)
        
        for start_idx in range(0, num_states, self.batch_size * self.parallel_batches):
            end_idx = min(start_idx + self.batch_size * self.parallel_batches, num_states)
            current_batch_states = states[start_idx:end_idx]
            
            batch_splits = np.array_split(current_batch_states, 
                                        min(self.parallel_batches, 
                                            len(current_batch_states)))
            
            for i, (state_batch, stream) in enumerate(zip(batch_splits, self.streams)):
                if not len(state_batch) > 0:
                    continue
                    
                stream.wait_stream(torch.cuda.current_stream())
                
                with torch.cuda.stream(stream):
                    for j, state in enumerate(state_batch):
                        preprocessed = self.preprocess_state(state)
                        self.state_tensors[i][j].copy_(preprocessed, non_blocking=True)
                    
                    if len(state_batch) == self.batch_size and 'forward' in self.cuda_graphs:
                        self.static_input.copy_(self.state_tensors[i][:len(state_batch)], non_blocking=True)
                        self.cuda_graphs['forward'].replay()
                        batch_output = self.static_output
                    else:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda'):
                                batch_output = self.model(self.state_tensors[i][:len(state_batch)])
                    
                    batch_output = batch_output.squeeze()
                    if batch_output.ndim == 0:
                        values.append(batch_output.item())
                    else:
                        values.extend(batch_output.cpu().numpy().tolist())
            
            for stream in self.streams:
                torch.cuda.current_stream().wait_stream(stream)
        
        return values

    def select_node(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.is_terminal:
            if not all(child.visits > 0 for child in node.children.values()):
                unexplored = [a for a, child in node.children.items() 
                            if child.visits == 0]
                return node.children[np.random.choice(unexplored)]
            
            ucb_values = {}
            for action, child in node.children.items():
                exploitation = child.value / child.visits
                exploration = self.exploration_constant * np.sqrt(np.log(node.visits) / child.visits)
                ucb_values[action] = exploitation + exploration
            
            best_action = max(ucb_values.items(), key=lambda x: x[1])[0]
            node = node.children[best_action]
        return node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        game = Game2048()
        game.board = torch.tensor(node.state, dtype=torch.float32)
        
        for action in range(4):
            new_board, reward, changed = game._move(game.board, action)
            if changed:
                if isinstance(new_board, torch.Tensor):
                    child_state = new_board.cpu().numpy()
                else:
                    child_state = new_board
                if action not in node.children:
                    child = MCTSNode(child_state, parent=node, action=action)
                    node.children[action] = child
        
        if not node.children:
            node.is_terminal = True
            return node
        
        return node.children[np.random.choice(list(node.children.keys()))]

    def simulate(self, node: MCTSNode, depth: int = 4) -> float:
        if depth == 0 or node.is_terminal:
            return self.evaluate_state(node.state)
        
        game = Game2048()
        game.board = torch.tensor(node.state, dtype=torch.float32)
        
        total_reward = 0
        current_state = node.state.copy()
        states_to_evaluate = []
        
        for _ in range(depth):
            valid_moves = []
            move_results = []
            
            for action in range(4):
                board_tensor = torch.tensor(current_state, dtype=torch.float32)
                new_board, reward, changed = game._move(board_tensor, action)
                
                if changed:
                    valid_moves.append(action)
                    if isinstance(new_board, torch.Tensor):
                        new_state = new_board.cpu().numpy()
                    else:
                        new_state = new_board
                    move_results.append((new_state, reward))
                    states_to_evaluate.append(new_state)
            
            if not valid_moves:
                break
                
            values = self.evaluate_states_batch(states_to_evaluate) if states_to_evaluate else []
            
            if not values:
                break
                
            best_idx = np.argmax(values)
            best_state, reward = move_results[best_idx]
            
            total_reward += reward
            current_state = best_state
            
            game = Game2048()
            game.board = torch.tensor(best_state, dtype=torch.float32)
            game.add_random_tile()
            current_state = game.board.cpu().numpy()
            
            states_to_evaluate = []
        
        return total_reward + self.evaluate_state(current_state)

    def backpropagate(self, node: MCTSNode, value: float):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def get_move(self, state: np.ndarray) -> int:
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            node = self.select_node(root)
            
            if node.visits > 0 and not node.is_terminal:
                node = self.expand_node(node)
            
            value = self.simulate(node)
            self.backpropagate(node, value)
        
        if not root.children:
            return 0
            
        best_action = max(root.children.items(),
                        key=lambda x: x[1].value / x[1].visits if x[1].visits > 0 else 0)[0]
        
        return best_action 