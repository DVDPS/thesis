import numpy as np
import random
import math
from typing import Tuple, List, Optional



def compute_monotonicity(board):
    """
    Computes a basic monotonicity score by summing the negative differences
    between adjacent tiles (in log2 space) along rows and columns.
    Higher scores (less negative) indicate a more monotonic board.
    This came from the paper 
    """
    board = board.astype(np.float32)
    # Avoid log(0) by setting zeros to a small positive value.
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    mono_score = 0.0
    # rows
    for row in log_board:
        mono_score -= np.sum(np.abs(np.diff(row)))
    # columns
    for col in log_board.T:
        mono_score -= np.sum(np.abs(np.diff(col)))
    return mono_score


class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        self.previous_max_tile = 0
        # Factor for the corner and border heuristic bonus
        self.corner_factor = 0.1  # Increased from 0.05 to give more weight to positioning
        # Enhanced weight matrix that favors both corners and edges
        self.corner_weights = np.array([
            [3.0, 2.0, 2.0, 3.0],
            [2.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0, 2.0],
            [3.0, 2.0, 2.0, 3.0]
        ])

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.previous_max_tile = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    def add_random_tile(self) -> None:
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            cell = random.choice(empty_cells)
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
        """Merge a single row; returns new row, score gained, and a flag if changed."""
        # Use a simple list-based merge without calling np.delete repeatedly.
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
        # Determine if the row has changed compared to the original.
        changed = not np.array_equal(new_row, row)
        return new_row, score, changed

    def _move(self, board, action, test_only = False):
        """
        Executes a move on a given board. action: 0=up, 1=right, 2=down, 3=left.
        Returns new board, score gained, and a flag whether the board changed.
        """
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

    def corner_heuristic(self) -> float:
        """
        Compute a bonus based on the positioning of high-value tiles on borders and corners.
        Higher values on the edges and especially corners will receive larger bonuses.
        """
        board_log = self.board.astype(np.float32)
        mask = board_log > 0
        board_log[mask] = np.log2(board_log[mask])
        
        # Calculate the weighted sum using the enhanced corner weights
        weighted_sum = np.sum(board_log * self.corner_weights)
        
        # Additional bonus for the highest values being on borders
        max_value = np.max(board_log)
        if max_value > 0:
            borders = np.concatenate([
                board_log[0, :],    # top row
                board_log[-1, :],   # bottom row
                board_log[1:-1, 0], # left column (excluding corners)
                board_log[1:-1, -1] # right column (excluding corners)
            ])
            border_bonus = 0.5 * np.sum(borders == max_value)  # bonus for each max value on border
            weighted_sum += border_bonus * max_value
        
        # Normalize by the sum of weights plus potential border bonus
        normalization = np.sum(self.corner_weights)
        return weighted_sum / normalization
    


    
    

    def step(self, action):
        old_board = self.board.copy()
        new_board, score_gain, valid_move = self._move(self.board, action)
        if valid_move:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()
        new_max_tile = int(np.max(self.board))
        
        # Use merged tile sum as base reward.
        reward = score_gain  # This is the sum of merged tile values.
        
        # Penalize invalid moves and game over.
        if not valid_move:
            reward -= 2
        if self.is_game_over():
            reward -= 100

        # Optionally, add a bonus for having empty tiles.
        empty_bonus = 0.2 * np.sum(self.board == 0)
        
        # Add a bonus based on monotonicity.
        mono_bonus = 0.01 * compute_monotonicity(self.board)
        
        reward += empty_bonus + mono_bonus

        info = {
            'score': self.score,
            'max_tile': new_max_tile,
            'valid_move': valid_move,
            'empty_cells': np.sum(self.board == 0),
            'merge_score': score_gain,
            'empty_bonus': empty_bonus,
            'monotonicity_bonus': mono_bonus
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
