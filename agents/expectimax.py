import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from src.thesis.environment.game2048 import Game2048, preprocess_state
import pickle

def apply_tile_downgrading(state: np.ndarray) -> np.ndarray:
    """
    This function applies tile downgrading to the state.
    It is used to convert the state into a more manageable format for the expectimax algorithm. 
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
    """
    This is the ExpectimaxAgent class that uses the expectimax algorithm to evaluate the state of the game.
    """
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
        """
        This is the get_move function that takes the state of the game and returns the best action.
        """
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
        return best_action

    def _hash_state(self, state: np.ndarray) -> str:
        return state.tobytes()

    def _determine_stage(self, state: np.ndarray) -> int:
        """
        This function determines the stage of the game.
        """
        max_tile = np.max(state)
        stage = 0
        for i, threshold in enumerate(self.stage_thresholds):
            if max_tile >= threshold:
                stage = i
        return stage

    def _expectimax(self, state: np.ndarray, depth: int, is_max: bool) -> float:
        """
        This function implements the expectimax algorithm. 
        """
        if depth == 0 or self._is_terminal(state):
            return self._evaluate_state(state)

        state_hash = self._hash_state(state)
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]

        if is_max:
            value = float("-inf")
            possible_moves = 0
            for action in [0, 1, 2, 3]:
                temp_state = state.copy()
                next_state, reward, changed = Game2048.simulate_move(temp_state, action)
                if changed:
                    possible_moves += 1
                    value = max(value, reward + self._expectimax(next_state, depth - 1, is_max=False))
            if possible_moves == 0:
                 return self._evaluate_state(state)
        else:
            value = 0
            empty_cells = np.transpose(np.where(state == 0))
            num_empty = len(empty_cells)
            if num_empty == 0:
                return self._evaluate_state(state)

            p = 1.0 / num_empty
            value_2 = 0
            value_4 = 0
            for i, j in empty_cells:
                state[i, j] = 2
                value_2 += self._expectimax(state, depth - 1, is_max=True)
                state[i, j] = 4
                value_4 += self._expectimax(state, depth - 1, is_max=True)
                state[i, j] = 0 
            value = p * (0.9 * value_2 + 0.1 * value_4)

        self.transposition_table[state_hash] = value
        return value

    def _is_terminal(self, state: np.ndarray) -> bool:
        if np.any(state == 0):
            return False
        for action in range(4):
            _, _, changed = Game2048.simulate_move(state, action)
            if changed:
                return False
        return True

    def _evaluate_state(self, state: np.ndarray) -> float:
        state = apply_tile_downgrading(state)
        heuristic_value = 0.0
        corners = [state[0,0], state[0,3], state[3,0], state[3,3]]
        max_corner = max(corners) if corners else 0
        if max_corner > 0:
            heuristic_value += max_corner * 0.3
        for i in range(4):
            row = state[i,:]
            col = state[:,i]
            if all(row[j] >= row[j+1] for j in range(3)):
                heuristic_value += np.sum(row) * 0.1
            if all(col[j] >= col[j+1] for j in range(3)):
                heuristic_value += np.sum(col) * 0.1

        smoothness = 0
        for i in range(4):
            for j in range(4):
                val = state[i, j]
                if val > 0:
                    log_val = np.log2(val)
                    if j < 3 and state[i, j + 1] > 0:
                        smoothness -= abs(log_val - np.log2(state[i, j + 1]))
                    if i < 3 and state[i + 1, j] > 0:
                        smoothness -= abs(log_val - np.log2(state[i + 1, j]))
        heuristic_value += smoothness * 0.1
        empty_cells = np.sum(state == 0)
        heuristic_value += empty_cells * 50

        merge_potential = 0
        """
        heuristic value = it is the sum of the state and the merge potential.
        the merge potential is the sum of the values of the tiles that can be merged.
        the values of the tiles that can be merged are the values of the tiles that are adjacent to each other.
        the merge potential is multiplied by 0.1 to give it less weight.
        the sum of the state is added to the heuristic value to give it more weight.
        """
        for i in range(4):
            for j in range(4):
                val = state[i,j]
                if val > 0:
                    if i < 3 and val == state[i+1,j]:
                        merge_potential += val * 1.0
                    if j < 3 and val == state[i,j+1]:
                        merge_potential += val * 1.0
        heuristic_value += merge_potential * 0.1
        heuristic_value += np.sum(state)

        return heuristic_value

    def _calculate_monotonicity(self, state: torch.Tensor) -> torch.Tensor:
        """
        This function calculates the monotonicity of the state.
        The monotonicity is the sum of the absolute differences between the values of the tiles in the state.
        """
        monotonicity_score = torch.tensor(0.0, device=self.device)
        log_state = torch.log2(torch.clamp(state, min=1))

        for i in range(4):
             monotonicity_score -= torch.sum(torch.abs(torch.diff(log_state[i, log_state[i,:] > 0])))
        for j in range(4):
             monotonicity_score -= torch.sum(torch.abs(torch.diff(log_state[log_state[:,j] > 0, j])))
        return monotonicity_score

    def _calculate_smoothness(self, state: torch.Tensor) -> torch.Tensor:
        smoothness_score = torch.tensor(0.0, device=self.device)
        log_state = torch.log2(torch.clamp(state, min=1))

        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j+1] > 0:
                    smoothness_score -= torch.abs(log_state[i, j] - log_state[i, j+1])
        for j in range(4):
            for i in range(3):
                if state[i, j] > 0 and state[i+1, j] > 0:
                    smoothness_score -= torch.abs(log_state[i, j] - log_state[i+1, j])
        return smoothness_score

    def _calculate_snake_pattern(self, state: torch.Tensor) -> torch.Tensor:
        """
        This function calculates the snake pattern of the state.
        The snake pattern is the sum of the values of the tiles in the state multiplied by the snake pattern matrix.
        """
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
        try:
            with open(model_path, "rb") as f:
                weights = pickle.load(f)
            self.stage_models[stage] = weights
            print(f"Successfully loaded model weights for stage {stage}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found")
        except Exception as e:
            print(f"Error loading model: {e}")