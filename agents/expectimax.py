import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from src.thesis.environment.game2048 import Game2048, preprocess_state
import pickle

def apply_tile_downgrading(state: np.ndarray) -> np.ndarray:
    """
    If the state contains very large tiles that are hard to evaluate,
    this function downgrades them by halving all tiles larger than the largest missing tile.
    """
    max_tile = np.max(state)
    unique_tiles = np.unique(state[state > 0])
    missing_tiles = []
    potential_tile = 2
    while potential_tile < max_tile:
        if potential_tile not in unique_tiles:
            missing_tiles.append(potential_tile)
        potential_tile *= 2
    if not missing_tiles:
        return state
    largest_missing = max(missing_tiles)
    downgraded_state = state.copy()
    downgraded_state[downgraded_state > largest_missing] //= 2
    return downgraded_state

class ExpectimaxAgent:
    def __init__(self, depth: int = 15, use_gpu: bool = True):
        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.transposition_table = {}  # Cache for previously evaluated states
        
        # Example heuristic: corner weight matrix
        self.corner_weight_matrix = torch.tensor([
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64]
        ], dtype=torch.float32, device=self.device)
        
        # Snake pattern weights (descending from bottom-right)
        self.snake_pattern_matrix = torch.tensor([
            [7, 6, 5, 4],
            [8, 9, 10, 3],
            [13, 12, 11, 2],
            [14, 15, 16, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Different stage models can be loaded here
        self.stage_thresholds = [0, 16384]  # Thresholds for different stages
        self.stage_models = {}  # Will store value networks for different stages
        # To use the TD model properly, you would also pass in your n-tuple configuration.
        self.n_tuples = None  # Set this if you plan to load a TD model

    def get_move(self, game_state: np.ndarray) -> int:
        max_value = float("-inf")
        best_action = 0
        actions = [0, 1, 2, 3]  # [Up, Right, Down, Left]
        for action in actions:
            next_state, reward, changed = Game2048.simulate_move(game_state, action)
            if changed:
                eval_state = apply_tile_downgrading(next_state)
                value = reward + self._expectimax(eval_state, self.depth - 1, is_max=False)
                if value > max_value:
                    max_value = value
                    best_action = action
        return best_action
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Create a hash for the state for the transposition table"""
        return state.tobytes()
    
    def _determine_stage(self, state: np.ndarray) -> int:
        """Determine which learning stage to use based on the max tile"""
        max_tile = np.max(state)
        stage = 0
        for i, threshold in enumerate(self.stage_thresholds):
            if max_tile >= threshold:
                stage = i
        return stage
    
    def _expectimax(self, state: np.ndarray, depth: int, is_max: bool) -> float:
        if depth == 0 or self._is_terminal(state):
            return self._evaluate_state(state)
        
        state_hash = self._hash_state(state)
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]
        
        if is_max:  # Player's move: maximize reward.
            value = float("-inf")
            for action in [0, 1, 2, 3]:
                temp_state = state.copy()
                next_state, reward, changed = Game2048.simulate_move(temp_state, action)
                if changed:
                    value = max(value, reward + self._expectimax(next_state, depth - 1, is_max=False))
        else:  # Chance node: average over possible new tile placements.
            value = 0
            empty_cells = np.transpose(np.where(state == 0))
            if len(empty_cells) == 0:
                return self._evaluate_state(state)
            p = 1.0 / len(empty_cells)
            for i, j in empty_cells:
                original = state[i, j]
                state[i, j] = 2
                value += 0.9 * p * self._expectimax(state, depth - 1, is_max=True)
                state[i, j] = 4
                value += 0.1 * p * self._expectimax(state, depth - 1, is_max=True)
                state[i, j] = original  # Restore cell.
        self.transposition_table[state_hash] = value
        return value
    
    def _is_terminal(self, state: np.ndarray) -> bool:
        for action in range(4):
            next_state, _, changed = Game2048.simulate_move(state, action)
            if changed:
                return False
        return True
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        """
        Evaluate state using the loaded TD model if available, otherwise use heuristic.
        """
        state_1d = state.flatten()
        stage = self._determine_stage(state)
        if stage in self.stage_models and self.n_tuples is not None:
            weights = self.stage_models[stage]
            value = 0
            for n_tuple in self.n_tuples:
                features = [state_1d[i] for i in n_tuple]
                key = tuple(features)
                if key in weights:
                    value += weights[key]
            return value
        # Fallback to heuristic evaluation
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        corner_score = torch.sum(state_tensor * self.corner_weight_matrix)
        return corner_score.item()
    
    def _calculate_monotonicity(self, state: torch.Tensor) -> torch.Tensor:
        monotonicity_score = torch.tensor(0.0, device=self.device)
        for i in range(4):
            for j in range(3):
                if state[i, j+1] >= state[i, j] and state[i, j+1] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i, j+1])
                elif state[i, j] > state[i, j+1] and state[i, j] > 0 and state[i, j+1] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        for j in range(4):
            for i in range(3):
                if state[i+1, j] >= state[i, j] and state[i+1, j] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i+1, j])
                elif state[i, j] > state[i+1, j] and state[i, j] > 0 and state[i+1, j] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        return monotonicity_score
    
    def _calculate_smoothness(self, state: torch.Tensor) -> torch.Tensor:
        smoothness_score = torch.tensor(0.0, device=self.device)
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j+1] > 0:
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i, j+1]))
        for j in range(4):
            for i in range(3):
                if state[i, j] > 0 and state[i+1, j] > 0:
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i+1, j]))
        return smoothness_score
    
    def _calculate_snake_pattern(self, state: torch.Tensor) -> torch.Tensor:
        snake_score = torch.sum(state * self.snake_pattern_matrix)
        return snake_score
    
    def _apply_tile_downgrading(self, state: torch.Tensor) -> torch.Tensor:
        max_tile = torch.max(state)
        unique_tiles = torch.unique(state[state > 0])
        missing_tiles = []
        potential_tile = 2
        while potential_tile < max_tile:
            if potential_tile not in unique_tiles:
                missing_tiles.append(potential_tile)
            potential_tile *= 2
        if not missing_tiles:
            return state
        largest_missing = max(missing_tiles)
        downgraded_state = state.clone()
        downgraded_state[state > largest_missing] = state[state > largest_missing] / 2
        return downgraded_state
    
    def load_model(self, stage: int, model_path: str):
        """Load a trained model for a specific stage"""
        try:
            with open(model_path, "rb") as f:
                weights = pickle.load(f)
            self.stage_models[stage] = weights
            print(f"Successfully loaded model weights for stage {stage}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found")
        except Exception as e:
            print(f"Error loading model: {e}")
