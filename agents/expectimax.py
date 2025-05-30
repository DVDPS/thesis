import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from src.thesis.environment.game2048 import Game2048, preprocess_state
import pickle

def apply_tile_downgrading(state: np.ndarray) -> np.ndarray:
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
        self.transposition_table = {}
        
        self.corner_weight_matrix = torch.tensor([
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64]
        ], dtype=torch.float32, device=self.device)
        
        self.snake_pattern_matrix = torch.tensor([
            [7, 6, 5, 4],
            [8, 9, 10, 3],
            [13, 12, 11, 2],
            [14, 15, 16, 1]
        ], dtype=torch.float32, device=self.device)
        
        self.stage_thresholds = [0, 16384]
        self.stage_models = {}
        self.n_tuples = None

    def get_move(self, game_state: np.ndarray) -> int:
        max_value = float("-inf")
        best_action = 0
        actions = [0, 1, 2, 3]
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
        return state.tobytes()
    
    def _determine_stage(self, state: np.ndarray) -> int:
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
        
        if is_max:
            value = float("-inf")
            for action in [0, 1, 2, 3]:
                temp_state = state.copy()
                next_state, reward, changed = Game2048.simulate_move(temp_state, action)
                if changed:
                    value = max(value, reward + self._expectimax(next_state, depth - 1, is_max=False))
        else:
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
                state[i, j] = original
        self.transposition_table[state_hash] = value
        return value
    
    def _is_terminal(self, state: np.ndarray) -> bool:
        for action in range(4):
            next_state, _, changed = Game2048.simulate_move(state, action)
            if changed:
                return False
        return True
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        state = apply_tile_downgrading(state)
        
        model_value = self.value_model.evaluate(state)
        
        heuristic_value = 0
        
        corners = [state[0,0], state[0,3], state[3,0], state[3,3]]
        max_corner = max(corners)
        if max_corner > 0:
            heuristic_value += max_corner * 0.3
        
        for i in range(4):
            row = state[i,:]
            col = state[:,i]
            if all(row[j] >= row[j+1] for j in range(3)):
                heuristic_value += sum(row) * 0.25
            if all(col[j] >= col[j+1] for j in range(3)):
                heuristic_value += sum(col) * 0.25
        
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if i < 3:
                    smoothness -= abs(state[i,j] - state[i+1,j])
                if j < 3:
                    smoothness -= abs(state[i,j] - state[i,j+1])
        heuristic_value += smoothness * 0.15
        
        empty_cells = np.sum(state == 0)
        heuristic_value += empty_cells * 150
        
        merge_potential = 0
        for i in range(4):
            for j in range(4):
                if i < 3 and state[i,j] == state[i+1,j] and state[i,j] > 0:
                    merge_potential += state[i,j] * 3
                if j < 3 and state[i,j] == state[i,j+1] and state[i,j] > 0:
                    merge_potential += state[i,j] * 3
        heuristic_value += merge_potential * 0.15
        
        max_tile = np.max(state)
        if max_tile >= 512:
            max_tile_pos = np.where(state == max_tile)
            if len(max_tile_pos[0]) > 0:
                i, j = max_tile_pos[0][0], max_tile_pos[1][0]
                if (i == 0 or i == 3) and (j == 0 or j == 3):
                    heuristic_value += max_tile * 0.4
                elif (i == 0 or i == 3) or (j == 0 or j == 3):
                    heuristic_value += max_tile * 0.2
        
        snake_score = 0
        for i in range(4):
            for j in range(4):
                if state[i,j] > 0:
                    if i % 2 == 0:
                        if j == 0 or state[i,j] >= state[i,j-1]:
                            snake_score += state[i,j]
                    else:
                        if j == 3 or state[i,j] >= state[i,j+1]:
                            snake_score += state[i,j]
        heuristic_value += snake_score * 0.2
        
        scaled_model_value = model_value * 0.15
        return scaled_model_value + heuristic_value
    
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
    
    class value_model:
        @staticmethod
        def evaluate(state: np.ndarray) -> float:
            return 0
