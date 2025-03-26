import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from src.thesis.environment.game2048 import Game2048, preprocess_state

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
    def __init__(self, depth: int = 3, use_gpu: bool = True):
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
        
    def get_move(self, game_state: np.ndarray) -> int:
        max_value = float("-inf")
        best_action = 0
        actions = [0, 1, 2, 3]  # [Up, Right, Down, Left]
        game = Game2048()
        for action in actions:
            next_state = game_state.copy()
            state_before = next_state.copy()
            next_state, reward, _, _ = game.step(action)
            if not np.array_equal(next_state, state_before):
                # Optionally apply tile downgrading before evaluation
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
            game = Game2048()
            for action in [0, 1, 2, 3]:
                temp_state = state.copy()
                next_state, reward, _, _ = game.step(action)
                if not np.array_equal(next_state, temp_state):
                    value = max(value, reward + self._expectimax(next_state, depth - 1, is_max=False))
        else:  # Chance node: average over possible new tile placements.
            value = 0
            empty_cells = np.transpose(np.where(state == 0))
            if len(empty_cells) == 0:
                return self._evaluate_state(state)
            p = 1.0 / len(empty_cells)
            for i, j in empty_cells:
                original = state[i, j]
                # Consider tile 2 with 90% probability.
                state[i, j] = 2
                value += 0.9 * p * self._expectimax(state, depth - 1, is_max=True)
                # Consider tile 4 with 10% probability.
                state[i, j] = 4
                value += 0.1 * p * self._expectimax(state, depth - 1, is_max=True)
                state[i, j] = original  # Restore cell.
        self.transposition_table[state_hash] = value
        return value
    
    def _is_terminal(self, state: np.ndarray) -> bool:
        game = Game2048()
        for action in range(4):
            temp_state = state.copy()
            next_state, _, _, _ = game.step(action)
            if not np.array_equal(temp_state, next_state):
                return False
        return True
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        """
        A simple heuristic evaluation based on the corner strategy.
        (In practice, the evaluation can combine multiple factors such as monotonicity, smoothness, free cells, etc.)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        corner_score = torch.sum(state_tensor * self.corner_weight_matrix)
        return corner_score.item()
    
    def _calculate_monotonicity(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate how monotonic the state is (decreasing from corner)"""
        # Calculate monotonicity along rows and columns
        monotonicity_score = torch.tensor(0.0, device=self.device)
        
        # For rows (decreasing from right to left)
        for i in range(4):
            for j in range(3):
                # If values are decreasing from right to left
                if state[i, j+1] >= state[i, j] and state[i, j+1] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i, j+1])
                # Penalty for non-monotonic arrangement
                elif state[i, j] > state[i, j+1] and state[i, j] > 0 and state[i, j+1] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        
        # For columns (decreasing from bottom to top)
        for j in range(4):
            for i in range(3):
                # If values are decreasing from bottom to top
                if state[i+1, j] >= state[i, j] and state[i+1, j] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i+1, j])
                # Penalty for non-monotonic arrangement
                elif state[i, j] > state[i+1, j] and state[i, j] > 0 and state[i+1, j] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        
        return monotonicity_score
    
    def _calculate_smoothness(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate smoothness of the state (adjacent tiles should be similar)"""
        smoothness_score = torch.tensor(0.0, device=self.device)
        
        # For rows
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j+1] > 0:
                    # Negative score for difference (we want to minimize difference)
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i, j+1]))
        
        # For columns
        for j in range(4):
            for i in range(3):
                if state[i, j] > 0 and state[i+1, j] > 0:
                    # Negative score for difference (we want to minimize difference)
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i+1, j]))
        
        return smoothness_score
    
    def _calculate_snake_pattern(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate how well the state follows a snake pattern"""
        # Snake pattern is rewarded by multiplying each cell by its position in the snake
        snake_score = torch.sum(state * self.snake_pattern_matrix)
        return snake_score
    
    def _apply_tile_downgrading(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply tile-downgrading technique to handle states with large tiles
        by translating them into downgraded states
        """
        # Find the largest missing tile
        max_tile = torch.max(state)
        unique_tiles = torch.unique(state[state > 0])
        
        # Find the largest missing tile that's smaller than max_tile
        missing_tiles = []
        potential_tile = 2
        while potential_tile < max_tile:
            if potential_tile not in unique_tiles:
                missing_tiles.append(potential_tile)
            potential_tile *= 2
        
        if not missing_tiles:
            return state  # No downgrading needed
        
        largest_missing = max(missing_tiles)
        
        # Create downgraded state by halving tiles larger than largest_missing
        downgraded_state = state.clone()
        downgraded_state[state > largest_missing] = state[state > largest_missing] / 2
        
        return downgraded_state
    
    def load_model(self, stage: int, model_path: str):
        """Load a trained model for a specific stage"""
        # This method would load your trained TD learning model for evaluation
        pass  # Implement based on your model format


