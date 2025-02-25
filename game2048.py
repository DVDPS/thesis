import numpy as np
import random
import math
from typing import Tuple, List, Optional



def compute_monotonicity(board):
    """
    Computes a monotonicity score by analyzing both row and column-wise patterns.
    Higher scores indicate better tile arrangements.
    """
    board = board.astype(np.float32)
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    
    def directional_monotonicity(arr):
        diffs = np.diff(arr)
        return np.sum(np.maximum(diffs, 0)), np.sum(np.maximum(-diffs, 0))
    
    total_score = 0
    for i in range(4):
        row = log_board[i]
        col = log_board[:, i]
        row_left, row_right = directional_monotonicity(row)
        col_up, col_down = directional_monotonicity(col)
        total_score -= min(row_left, row_right) + min(col_up, col_down)
    
    return total_score

def compute_smoothness(board):
    """
    Measures how smooth the board is by calculating differences between adjacent tiles.
    Lower scores indicate a smoother board with more merge opportunities.
    """
    board = board.astype(np.float32)
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    
    smoothness = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] != 0:
                if j < 3 and board[i, j+1] != 0:
                    smoothness -= abs(log_board[i, j] - log_board[i, j+1])
                if i < 3 and board[i+1, j] != 0:
                    smoothness -= abs(log_board[i, j] - log_board[i+1, j])
    return smoothness

def compute_merge_potential(board):
    """
    Calculates the potential for future merges by looking at adjacent equal values.
    Higher scores indicate more merge opportunities.
    """
    merge_score = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] != 0:
                if j < 3 and board[i, j] == board[i, j+1]:
                    merge_score += board[i, j]
                if i < 3 and board[i, j] == board[i+1, j]:
                    merge_score += board[i, j]
    return merge_score

class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        self.previous_max_tile = 0
        # Enhanced weights for different heuristics
        self.monotonicity_weight = 0.2
        self.smoothness_weight = 0.1
        self.merge_potential_weight = 0.15
        self.empty_cell_weight = 0.3
        self.corner_weight = 0.25
        
        # Enhanced corner weights that favor a snake-like pattern
        self.corner_weights = np.array([
            [7.0, 6.0, 5.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0]
        ])

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.previous_max_tile = 0
        self.moves_without_merge = 0  # Track moves without merges
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    def add_random_tile(self) -> None:
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            cell = random.choice(empty_cells)
            # Slightly increased probability of 4 (20% instead of original 10%)
            self.board[cell] = 2 if random.random() < 0.8 else 4

    def get_possible_moves(self) -> List[int]:
        moves = []
        for action in range(4):
            temp_board = self.board.copy()
            _, _, changed = self._move(temp_board, action, test_only=True)
            if changed:
                moves.append(action)
        return moves

    def _merge_row(self, row):
        filtered = row[row != 0]
        merged = []
        score = 0
        i = 0
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i+1]:
                merged_val = filtered[i] * 2
                merged.append(merged_val)
                score += merged_val
                i += 2
            else:
                merged.append(filtered[i])
                i += 1
        new_row = np.array(merged, dtype=np.int32)
        new_row = np.pad(new_row, (0, self.size - len(new_row)), 'constant')
        changed = not np.array_equal(new_row, row)
        return new_row, score, changed

    def _move(self, board, action, test_only = False):
        rotated = np.rot90(board.copy(), k=action)
        total_score = 0
        changed = False
        for i in range(self.size):
            new_row, score, row_changed = self._merge_row(rotated[i])
            if row_changed:
                changed = True
            rotated[i] = new_row
            total_score += score
        new_board = np.rot90(rotated, k=-action)
        return new_board, total_score, changed

    def step(self, action):
        old_board = self.board.copy()
        old_max = np.max(old_board)
        new_board, score_gain, valid_move = self._move(self.board, action)
        
        if valid_move:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()
            if score_gain > 0:
                self.moves_without_merge = 0
            else:
                self.moves_without_merge += 1
        
        new_max_tile = int(np.max(self.board))
        empty_cells = np.sum(self.board == 0)
        
        # Base reward from merging
        reward = score_gain
        
        # Penalty for invalid moves and stagnation
        if not valid_move:
            reward -= 10  # Increased penalty for invalid moves
        if self.moves_without_merge > 5:
            reward -= 5  # Penalty for too many moves without merges
        
        # Bonus for reaching new max tile
        if new_max_tile > old_max:
            reward += 50 * np.log2(new_max_tile)  # Logarithmic bonus for higher tiles
        
        # Add weighted heuristic components
        monotonicity = compute_monotonicity(self.board)
        smoothness = compute_smoothness(self.board)
        merge_potential = compute_merge_potential(self.board)
        
        # Calculate positional score using corner weights
        positional_score = np.sum(self.board * self.corner_weights)
        
        # Combine all heuristics with their weights
        heuristic_reward = (
            self.monotonicity_weight * monotonicity +
            self.smoothness_weight * smoothness +
            self.merge_potential_weight * merge_potential +
            self.empty_cell_weight * empty_cells +
            self.corner_weight * positional_score / np.max(self.board)  # Normalize positional score
        )
        
        # Add heuristic reward to base reward
        reward += heuristic_reward
        
        # Game over penalty
        if self.is_game_over():
            reward -= 200  # Increased penalty for game over
        
        info = {
            'score': self.score,
            'max_tile': new_max_tile,
            'valid_move': valid_move,
            'empty_cells': empty_cells,
            'merge_score': score_gain,
            'monotonicity': monotonicity,
            'smoothness': smoothness,
            'merge_potential': merge_potential,
            'positional_score': positional_score
        }
        
        return self.board.copy(), reward, self.is_game_over(), info

    def is_game_over(self) -> bool:
        return len(self.get_possible_moves()) == 0

def preprocess_state(state):
    """Convert board state to log2 scale; zeros remain zeros."""
    state = state.astype(np.float32)
    mask = state > 0
    state[mask] = np.log2(state[mask])
    return state 


def preprocess_state_onehot(state):
    """
    Converts the 4x4 board into a one-hot representation with 16 channels.
    Channel 0 represents an empty tile (0).
    Channels 1-15 represent tiles with values 2^1 through 2^15.
    """
    board = state.astype(np.int32)
    channels = 16
    onehot = np.zeros((channels, board.shape[0], board.shape[1]), dtype=np.float32)
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] > 0:
                power = int(np.log2(board[i, j]))
                if power < channels:
                    onehot[power, i, j] = 1.0
            else:
                onehot[0, i, j] = 1.0  # Empty tiles are represented in channel 0
    
    return onehot
