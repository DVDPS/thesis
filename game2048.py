import numpy as np
import random
import math
from typing import Tuple, List, Optional

class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        self.previous_max_tile = 0
        # Factor for the corner heuristic bonus.
        self.corner_factor = 0.01
        # Parameterize the corner heuristic with a weight matrix;
        # you might later experiment with alternative corners or normalization.
        self.corner_weights = np.array([
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0, 0.5],
            [2.0, 1.0, 0.5, 0.25],
            [1.0, 0.5, 0.25, 0.1]
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
        Compute a bonus based on the positioning of high-value tiles in the top-left corner.
        """
        # Use the parameterized corner weights.
        board_log = self.board.astype(np.float32)
        mask = board_log > 0
        board_log[mask] = np.log2(board_log[mask])
        weighted_sum = np.sum(board_log * self.corner_weights)
        # Normalize the heuristic by the sum of the weights.
        normalization = np.sum(self.corner_weights)
        return weighted_sum / normalization

    def step(self, action):
        """Execute an action, update state and return (state, reward, done, info)."""
        old_board = self.board.copy()
        old_empty = np.sum(old_board == 0)
        new_board, score_gain, valid_move = self._move(self.board, action)
        if valid_move:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()
        new_empty = np.sum(self.board == 0)
        new_max_tile = int(np.max(self.board))
        
        # Base reward components.
        merge_reward = score_gain / 50.0
        max_tile_reward = 0.0
        if new_max_tile > self.previous_max_tile:
            max_tile_reward = np.log2(new_max_tile) * 40
            self.previous_max_tile = new_max_tile
        empty_bonus = 0.2 * new_empty
        
        reward = merge_reward + max_tile_reward + empty_bonus
        if not valid_move:
            reward -= 2
        if self.is_game_over():
            reward -= 100
        corner_bonus = self.corner_heuristic() * self.corner_factor
        reward += corner_bonus
        
        info = {
            'score': self.score,
            'max_tile': new_max_tile,
            'valid_move': valid_move,
            'empty_cells': new_empty,
            'merge_reward': merge_reward,
            'max_tile_reward': max_tile_reward,
            'empty_bonus': empty_bonus,
            'corner_bonus': corner_bonus
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