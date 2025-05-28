import numpy as np
import torch
from typing import List, Tuple, Dict
from .cnn_agent import Game2048CNN
from src.thesis.environment.game2048 import Game2048

class BeamState:
    def __init__(self, board: np.ndarray, score: float, action_sequence: List[int]):
        self.board = board
        self.score = score
        self.action_sequence = action_sequence

class CNNBeamSearchAgent:
    def __init__(self, beam_width=10, search_depth=4, use_gpu=True):
        self.beam_width = beam_width
        self.search_depth = search_depth
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
        
        self.state_tensors = [
            torch.zeros((self.batch_size, 16, 4, 4),
                       dtype=torch.float32).to(self.device)
            for _ in range(self.parallel_batches)
        ]
        
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

    def get_next_states(self, state: np.ndarray) -> List[Tuple[np.ndarray, int, float]]:
        game = Game2048()
        game.board = torch.tensor(state, dtype=torch.float32)
        next_states = []
        
        for action in range(4):
            new_board, reward, changed = game._move(game.board, action)
            if changed:
                if isinstance(new_board, torch.Tensor):
                    new_state = new_board.cpu().numpy()
                else:
                    new_state = new_board
                next_states.append((new_state, action, reward))
        
        return next_states

    def beam_search(self, state: np.ndarray) -> List[int]:
        initial_beam = [BeamState(state, 0, [])]
        
        for depth in range(self.search_depth):
            candidates = []
            states_to_evaluate = []
            state_map = {}
            
            for beam_state in initial_beam:
                next_states = self.get_next_states(beam_state.board)
                for new_state, action, reward in next_states:
                    game = Game2048()
                    game.board = torch.tensor(new_state, dtype=torch.float32)
                    game.add_random_tile()
                    new_state = game.board.cpu().numpy()
                    
                    states_to_evaluate.append(new_state)
                    state_map[len(states_to_evaluate) - 1] = (
                        new_state,
                        beam_state.score + reward,
                        beam_state.action_sequence + [action]
                    )
            
            if not states_to_evaluate:
                break
            
            values = self.evaluate_states_batch(states_to_evaluate)
            
            for idx, value in enumerate(values):
                new_state, score, action_sequence = state_map[idx]
                candidates.append(BeamState(
                    new_state,
                    score + value,
                    action_sequence
                ))
            
            candidates.sort(key=lambda x: x.score, reverse=True)
            initial_beam = candidates[:self.beam_width]
        
        if not initial_beam:
            return 0
        
        best_sequence = max(initial_beam, key=lambda x: x.score).action_sequence
        return best_sequence[0] if best_sequence else 0

    def get_move(self, state: np.ndarray) -> int:
        return self.beam_search(state) 