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
        old_max_tile = np.max(old_board)
        new_board, score_gain, valid_move = self._move(self.board, action)
        
        if valid_move:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()
        
        new_max_tile = int(np.max(self.board))
        
        # --- REWARD CALCULATION ---
        # Base reward from merged tiles
        reward = score_gain
        
        # IMPROVEMENT 1: Exponential rewards for higher tiles
        # Give exponentially higher rewards for creating new max tiles
        if new_max_tile > old_max_tile:
            # log2(new_max_tile) is the power of 2 for the tile (e.g., 8=3, 16=4, etc.)
            # This makes rewards grow exponentially with tile value
            reward += 10 * math.log2(new_max_tile) ** 2
        
        # IMPROVEMENT 2: Better penalties
        if not valid_move:
            reward -= 4  # Increased penalty for invalid moves
        
        if self.is_game_over():
            # Scale game over penalty with board state - less penalty if you have high tiles
            highest_power = math.log2(new_max_tile) if new_max_tile > 0 else 0
            reward -= max(200 - highest_power * 10, 50)  # Lower penalty for higher achievements
        
        # IMPROVEMENT 3: Enhanced strategic rewards
        # Reward for empty tiles (space to maneuver)
        empty_count = np.sum(self.board == 0)
        empty_bonus = 0.5 * empty_count
        
        # Improved monotonicity - reward organized boards
        mono_bonus = 0.05 * compute_monotonicity(self.board)
        
        # Enhanced corner placement - more significant reward
        corner_bonus = 0.2 * self.corner_heuristic()
        
        # IMPROVEMENT 4: Snaking pattern bonus (common in expert play)
        snake_bonus = 0.1 * self.compute_snake_pattern()
        
        # IMPROVEMENT 5: Merge potential bonus
        merge_potential = 0.1 * self.compute_merge_potential()
        
        # Add all components
        reward += empty_bonus + mono_bonus + corner_bonus + snake_bonus + merge_potential
        
        info = {
            'score': self.score,
            'max_tile': new_max_tile,
            'valid_move': valid_move,
            'empty_cells': empty_count,
            'merge_score': score_gain,
            'empty_bonus': empty_bonus,
            'monotonicity_bonus': mono_bonus,
            'corner_bonus': corner_bonus,
            'snake_bonus': snake_bonus,
            'merge_potential': merge_potential
        }
        
        return self.board.copy(), reward, self.is_game_over(), info

    def compute_snake_pattern(self):
        """
        Calculate a score for how well the board follows a snake pattern.
        Snake pattern is a zigzag arrangement with decreasing values:
        
        high → → → →
        ↑ → → → → ↓
        ↑ ← ← ← ← ↓
        ↑ ← ← ← low
        """
        # Define the ideal snake path order
        snake_path = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 3), (3, 2), (3, 1), (3, 0)
        ]
        
        # Get values along the snake path
        values = []
        for i, j in snake_path:
            if self.board[i, j] > 0:
                values.append(math.log2(self.board[i, j]))
            else:
                values.append(0)
        
        # Calculate how monotonically decreasing the values are along the path
        score = 0
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                score += 1
                # Bonus for large differences between adjacent tiles
                score += min(values[i] - values[i + 1], 3)
            else:
                # Penalty for increasing values along the path
                score -= 1
        
        return score

    def compute_merge_potential(self):
        """
        Calculate the potential for future merges (adjacency of similar values).
        Higher score means more potential future merges.
        """
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    continue
                    
                # Check horizontally and vertically for same values
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        if self.board[i, j] == self.board[ni, nj]:
                            # Award more points for higher value matching pairs
                            match_value = math.log2(self.board[i, j])
                            score += match_value
        
        return score

    def is_game_over(self) -> bool:
        return len(self.get_possible_moves()) == 0

    def render(self):
        """
        Render the game board to the console.
        """
        print("\n" + "-" * 25)
        print(f"Score: {self.score}")
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("".center(5), end="|")
                else:
                    print(str(cell).center(5), end="|")
            print("\n" + "-" * 25)
        print("")

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
