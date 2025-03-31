import numpy as np
import random
import math
from typing import Tuple, List, Optional
import torch

def compute_monotonicity(board):
    
    board = board.astype(np.float32)
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    mono_score = 0.0
    
    for row in log_board:
        diff = np.diff(row)
        mono_score -= np.sum(np.abs(diff[diff != 0]))
    for col in log_board.T:
        diff = np.diff(col)
        mono_score -= np.sum(np.abs(diff[diff != 0]))

    snake_score = 0
    for i in range(4):
        row_log = log_board[i]
        if i % 2 == 0:
            diffs = row_log[:-1] - row_log[1:]
            snake_score += np.sum(diffs >= 0)
        else:
             diffs = row_log[1:] - row_log[:-1]
             snake_score += np.sum(diffs >= 0)

    corner_score = 0
    corners = [(0,0), (0,3), (3,0), (3,3)]
    max_tile_log = np.max(log_board)
    if max_tile_log > 0:
        for r, c in corners:
            if log_board[r, c] == max_tile_log:
                corner_score += 2
            elif log_board[r, c] >= max_tile_log - 1:
                corner_score += 1

    return mono_score + 0.5 * snake_score + 0.3 * corner_score

class Game2048:
    """
    This is the Game2048 class that implements the game of 2048.
    """
    def __init__(self, seed=None):
        self.size = 4
        self.corner_weights = np.array([
            [16, 8, 4, 2],
            [ 8, 4, 2, 1],
            [ 4, 2, 1, 0],
            [ 2, 1, 0, 0]
        ]) * 4.0
        self.reset(seed)

    def reset(self, seed=None):
        """
        This is the reset function that resets the game, with a random seed if specified.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            else:
                 torch.manual_seed(seed)

        self.board = torch.zeros((self.size, self.size), dtype=torch.int32)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.cpu().numpy()

    def add_random_tile(self):
        """
        This is the add_random_tile function that adds a random tile to the board.
        """
        empty_cells = torch.where(self.board == 0)
        if len(empty_cells[0]) > 0:
            idx = torch.randint(0, len(empty_cells[0]), (1,)).item()
            row, col = empty_cells[0][idx], empty_cells[1][idx]
            self.board[row, col] = 2 if torch.rand(1).item() < 0.9 else 4

    def _move_line(self, line):
        """
        This is the _move_line function that moves and merges a single line (row/column).
        """
        non_zeros = line[line != 0]
        new_line = torch.zeros_like(line)
        merged_score = 0
        target_idx = 0
        skip_next = False
        
        for i in range(len(non_zeros)):
            if skip_next:
                skip_next = False
                continue
            
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i+1]:
                merged_val = non_zeros[i] * 2
                new_line[target_idx] = merged_val
                merged_score += merged_val.item()
                target_idx += 1
                skip_next = True
            else:
                new_line[target_idx] = non_zeros[i]
                target_idx += 1
                
        changed = not torch.equal(line, new_line)
        return new_line, merged_score, changed

    def _move_gpu(self, board_tensor: torch.Tensor, action: int):

         original_board = board_tensor.clone()
         rotated_board = torch.rot90(board_tensor, k=action)
         
         total_score_gain = 0
         any_changed = False
         new_rotated_board = torch.zeros_like(rotated_board)

         for i in range(self.size):
             line = rotated_board[i, :]
             new_line, score_gain, changed = self._move_line(line)
             new_rotated_board[i, :] = new_line
             total_score_gain += score_gain
             if changed:
                 any_changed = True

         final_board = torch.rot90(new_rotated_board, k=-action)

         return final_board, total_score_gain, any_changed

    def _move(self, board, action):
         if isinstance(board, np.ndarray):
             board_tensor = torch.from_numpy(board).to(dtype=torch.int32)
         elif isinstance(board, torch.Tensor):
             board_tensor = board.to(dtype=torch.int32)
         else:
             raise TypeError("Board must be a numpy array or torch tensor")

         rot_map = {0: 1, 1: 2, 2: 3, 3: 0}
         k_rot = rot_map.get(action, 0)
         final_board_tensor, score_gain, changed = self._move_gpu(board_tensor, k_rot)
         return final_board_tensor.cpu().numpy(), score_gain, changed

    def is_game_over(self):
        if torch.any(self.board == 0):
            return False
    
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
                    
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
                    
        return True

    def get_valid_moves(self):
        valid_moves = []
        current_board_np = self.board.cpu().numpy()
        for action in range(4):
            _, _, changed = Game2048.simulate_move(current_board_np, action)
            if changed:
                valid_moves.append(action)
        return valid_moves

    def get_state(self):
        return self.board.cpu().numpy()

    def get_score(self):
        return self.score

    def corner_heuristic(self) -> float:
        board_np = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        
        board_log = board_np.astype(np.float32)
        mask = board_log > 0
        board_log[mask] = np.log2(board_log[mask])
        corner_weights_np = self.corner_weights if isinstance(self.corner_weights, np.ndarray) else self.corner_weights.cpu().numpy()

        weighted_sum = np.sum(board_log * corner_weights_np)

        max_value = np.max(board_log)
        border_bonus = 0
        if max_value > 0:
            borders = np.concatenate([
                board_log[0, :], board_log[-1, :],
                board_log[1:-1, 0], board_log[1:-1, -1]
            ])
            border_bonus = 0.5 * np.sum(borders == max_value) * max_value

        normalization = np.sum(corner_weights_np) + 1e-6
        return (weighted_sum + border_bonus) / normalization

    def count_potential_high_value_merges(self):
        board_np = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        bonus = 0.0
        threshold = 64

        for i in range(self.size):
            for j in range(self.size):
                val = board_np[i, j]
                if val >= threshold:
                    val_weight = math.log2(val)

                    if j < self.size - 1 and val == board_np[i, j + 1]:
                        bonus += val_weight * 2.0
                    
                    if i < self.size - 1 and val == board_np[i + 1, j]:
                        bonus += val_weight * 2.0
        return bonus

    def compute_snake_pattern(self):
        board_np = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        score = 0.0
        
        for i in range(self.size):
             row_vals = [math.log2(x) if x > 0 else 0 for x in board_np[i, :]]
             if i % 2 == 0:
                  for j in range(self.size - 1):
                      diff = row_vals[j] - row_vals[j+1]
                      if diff >= 0:
                           score += 1 + min(diff, 3)
                      else:
                           score -= 1
             else:
                  for j in range(self.size - 1, 0, -1):
                      diff = row_vals[j] - row_vals[j-1]
                      if diff >= 0:
                           score += 1 + min(diff, 3)
                      else:
                           score -= 1
        return score

    def compute_merge_potential(self):
        board_np = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        score = 0.0
        for i in range(self.size):
            for j in range(self.size):
                val = board_np[i, j]
                if val > 0:
                    log_val = math.log2(val)
                    if j < self.size - 1 and val == board_np[i, j + 1]:
                        score += log_val
                    if i < self.size - 1 and val == board_np[i + 1, j]:
                        score += log_val
        return score

    def render(self):
        board_to_render = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        print("\n" + "-" * (self.size * 6 + 1))
        print(f"Score: {self.score}")
        for row in board_to_render:
            print("|", end="")
            for cell in row:
                val_str = str(int(cell)) if cell != 0 else ""
                print(val_str.center(5), end="|")
            print("\n" + "-" * (self.size * 6 + 1))
        print("")

    def check_adjacent_512_tiles(self):
        board_np = self.board.cpu().numpy() if isinstance(self.board, torch.Tensor) else self.board
        bonus = 0
        tile_val = 512
        positions = np.argwhere(board_np == tile_val)

        if len(positions) < 2:
            return 0

        bonus += 400

        checked_pairs = set()
        for i in range(len(positions)):
            for k in range(i + 1, len(positions)):
                 r1, c1 = positions[i]
                 r2, c2 = positions[k]
                 if abs(r1 - r2) + abs(c1 - c2) == 1:
                      bonus += 1000
        return bonus

    @staticmethod
    def merge_row_static(row: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        size = len(row)
        filtered = row[row != 0]
        merged = []
        score = 0
        i = 0
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i+1]:
                merged_val = filtered[i] * 2
                merged.append(merged_val)
                score += int(merged_val)
                i += 2
            else:
                merged.append(filtered[i])
                i += 1
        
        new_row = np.zeros(size, dtype=row.dtype)
        if merged:
             new_row[:len(merged)] = np.array(merged, dtype=row.dtype)
             
        changed = not np.array_equal(new_row, row)
        return new_row, score, changed

    @staticmethod
    def simulate_move(board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
         size = board.shape[0]
         new_board = board.copy()
         total_score = 0
         changed = False
         if action in [0, 2]:
             axis = 0
             for j in range(size):
                 col = new_board[:, j]
                 processed_col, score, col_changed = Game2048.merge_row_static(col if action == 2 else col[::-1])
                 new_board[:, j] = processed_col if action == 2 else processed_col[::-1]
                 total_score += score
                 if col_changed: changed = True
         elif action in [1, 3]:
              axis = 1
              for i in range(size):
                  row = new_board[i, :]

                  processed_row, score, row_changed = Game2048.merge_row_static(row if action == 3 else row[::-1])
                  new_board[i, :] = processed_row if action == 3 else processed_row[::-1]
                  total_score += score
                  if row_changed: changed = True
         else:
              raise ValueError("Invalid action specified.")

         return new_board, total_score, changed

def preprocess_state(state):
    state = state.astype(np.float32)
    mask = state > 0
    state[mask] = np.log2(state[mask])
    return state

def preprocess_state_onehot(state):
    board = state.astype(np.int32)
    channels = 16
    size = board.shape[0]
    onehot = np.zeros((channels, size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            val = board[i, j]
            if val > 0:
                power = int(math.log2(val))
                if 0 < power < channels:
                    onehot[power, i, j] = 1.0

            else:
                onehot[0, i, j] = 1.0 



    return onehot