import numpy as np
import torch
from typing import Dict, List, Tuple
from .cnn_agent import Game2048CNN
from src.thesis.environment.game2048 import Game2048

class MCTSNode:
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
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

        # GPU optimization settings
        self.batch_size = 8192  # Increased from 64 to 8192
        self.parallel_batches = 4
        self.warmup_steps = 100
        
        # Pre-allocate tensors for batch evaluation
        self.state_tensors = [
            torch.zeros((self.batch_size, 16, 4, 4),
                       dtype=torch.float32).to(self.device)
            for _ in range(self.parallel_batches)
        ]
        
        # Buffers for states
        self.state_buffers = [[] for _ in range(self.parallel_batches)]
        self.current_buffer = 0
        
        # Pre-allocate tensors for computation
        self.temp_storage = {
            'features': torch.zeros((self.batch_size * self.parallel_batches, 512, 4, 4),
                                  dtype=torch.float32).to(self.device),
            'intermediate': torch.zeros((self.batch_size * self.parallel_batches, 256, 2, 2),
                                     dtype=torch.float32).to(self.device),
            'output': torch.zeros((self.batch_size * self.parallel_batches,),
                                dtype=torch.float32).to(self.device)
        }
        
        # Initialize CUDA graphs for repeated operations
        self.cuda_graphs = {}
        if use_gpu and torch.cuda.is_available():
            self._init_cuda_graphs()
            
        # Enable CUDA stream usage
        self.streams = [torch.cuda.Stream() for _ in range(self.parallel_batches)] if use_gpu and torch.cuda.is_available() else []

    def _init_cuda_graphs(self):
        """Initialize CUDA graphs for repeated operations"""
        print("Initializing CUDA graphs for optimized processing...")
        
        # Create a sample input for graph capture
        sample_input = torch.zeros((self.batch_size, 16, 4, 4),
                                 dtype=torch.float32,
                                 device=self.device)
        
        # Warmup and capture forward pass
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # Warmup
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        _ = self.model(sample_input)
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    self.static_output = self.model(sample_input)
        
        self.cuda_graphs['forward'] = g
        self.static_input = sample_input
        print("CUDA graphs initialized successfully.")

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert board state to one-hot representation"""
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
        """Evaluate a single state using the CNN"""
        with torch.no_grad():
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            value = self.model(state_tensor).item()
        return value

    def evaluate_states_batch(self, states: List[np.ndarray]) -> List[float]:
        """Evaluate multiple states in a batch using parallel processing"""
        if not states:
            return []
        
        values = []
        num_states = len(states)
        
        # Process multiple batches in parallel using CUDA streams
        for start_idx in range(0, num_states, self.batch_size * self.parallel_batches):
            end_idx = min(start_idx + self.batch_size * self.parallel_batches, num_states)
            current_batch_states = states[start_idx:end_idx]
            
            # Split into parallel batches
            batch_splits = np.array_split(current_batch_states, 
                                        min(self.parallel_batches, 
                                            len(current_batch_states)))
            
            # Process batches in parallel streams
            for i, (state_batch, stream) in enumerate(zip(batch_splits, self.streams)):
                if not len(state_batch) > 0:
                    continue
                    
                stream.wait_stream(torch.cuda.current_stream())
                
                with torch.cuda.stream(stream):
                    # Prepare batch
                    for j, state in enumerate(state_batch):
                        preprocessed = self.preprocess_state(state)
                        self.state_tensors[i][j].copy_(preprocessed, non_blocking=True)
                    
                    # Process batch using CUDA graph if possible
                    if len(state_batch) == self.batch_size and 'forward' in self.cuda_graphs:
                        self.static_input.copy_(self.state_tensors[i][:len(state_batch)], non_blocking=True)
                        self.cuda_graphs['forward'].replay()
                        batch_output = self.static_output
                    else:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda'):
                                batch_output = self.model(self.state_tensors[i][:len(state_batch)])
                    
                    # Handle both single value and batch outputs
                    batch_output = batch_output.squeeze()
                    if batch_output.ndim == 0:  # Single value
                        values.append(batch_output.item())
                    else:  # Batch of values
                        values.extend(batch_output.cpu().numpy().tolist())
            
            # Synchronize streams
            for stream in self.streams:
                torch.cuda.current_stream().wait_stream(stream)
        
        return values

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1"""
        while node.children and not node.is_terminal:
            if not all(child.visits > 0 for child in node.children.values()):
                # If some children are unexplored, select one randomly
                unexplored = [a for a, child in node.children.items() 
                            if child.visits == 0]
                return node.children[np.random.choice(unexplored)]
            
            # UCB1 formula
            ucb_values = {}
            for action, child in node.children.items():
                exploitation = child.value / child.visits
                exploration = self.exploration_constant * np.sqrt(np.log(node.visits) / child.visits)
                ucb_values[action] = exploitation + exploration
            
            best_action = max(ucb_values.items(), key=lambda x: x[1])[0]
            node = node.children[best_action]
        return node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding all possible children"""
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
        
        # Return a random child node
        return node.children[np.random.choice(list(node.children.keys()))]

    def simulate(self, node: MCTSNode, depth: int = 4) -> float:
        """Simulate a game from a node using the CNN for evaluation"""
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
            
            # Try all possible moves
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
            
            # Batch evaluate all states
            if states_to_evaluate:
                values = self.evaluate_states_batch(states_to_evaluate)
                move_values = []
                for (_, reward), value in zip(move_results, values):
                    move_values.append(value + reward)
                
                best_idx = np.argmax(move_values)
                action = valid_moves[best_idx]
                current_state, reward = move_results[best_idx]
                total_reward += reward
                
                # Clear states buffer
                states_to_evaluate = []
            
            # Add random tile
            game.board = torch.tensor(current_state, dtype=torch.float32)
            game.add_random_tile()
            current_state = game.board.cpu().numpy()
        
        return total_reward + self.evaluate_state(current_state)

    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def get_move(self, state: np.ndarray) -> int:
        """Get the best move for a given state"""
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            # Selection
            node = self.select_node(root)
            
            # Expansion
            if not node.is_terminal and node.visits > 0:
                node = self.expand_node(node)
            
            # Simulation
            value = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, value)
        
        # Choose the move with highest average value
        if not root.children:
            return 0  # No valid moves
            
        avg_values = {action: child.value / child.visits 
                     for action, child in root.children.items()}
        return max(avg_values.items(), key=lambda x: x[1])[0] 