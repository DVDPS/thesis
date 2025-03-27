import numpy as np
import random
import math
from typing import Tuple, List, Optional
import torch

def compute_monotonicity(board):
    """
    Computes an enhanced monotonicity score that considers:
    1. Basic monotonicity along rows and columns
    2. Snake pattern monotonicity
    3. Corner strategy
    """
    board = board.astype(np.float32)
    # Avoid log(0) by setting zeros to a small positive value
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    mono_score = 0.0
    
    # 1. Basic monotonicity (rows and columns)
    for row in log_board:
        mono_score -= np.sum(np.abs(np.diff(row)))
    for col in log_board.T:
        mono_score -= np.sum(np.abs(np.diff(col)))
    
    # 2. Snake pattern monotonicity
    snake_score = 0
    for i in range(4):
        if i % 2 == 0:
            # Left to right
            for j in range(3):
                if log_board[i,j] >= log_board[i,j+1]:
                    snake_score += 1
        else:
            # Right to left
            for j in range(3):
                if log_board[i,j] <= log_board[i,j+1]:
                    snake_score += 1
    
    # 3. Corner strategy
    corner_score = 0
    corners = [(0,0), (0,3), (3,0), (3,3)]
    max_tile = np.max(log_board)
    for corner in corners:
        if log_board[corner] == max_tile:
            corner_score += 2
        elif log_board[corner] >= max_tile - 1:
            corner_score += 1
    
    # Combine scores with weights
    return mono_score + 0.5 * snake_score + 0.3 * corner_score

class Game2048:
    def __init__(self, seed=None):
        self.reset(seed)
    
    def reset(self, seed=None):
        """Reset the game with optional seed"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize board as tensor
        self.board = torch.zeros((4, 4), dtype=torch.float32)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.cpu().numpy()
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        # Convert board to tensor if it's numpy array
        if isinstance(self.board, np.ndarray):
            self.board = torch.from_numpy(self.board).float()
            
        empty_cells = torch.where(self.board == 0)
        if len(empty_cells[0]) > 0:
            idx = torch.randint(0, len(empty_cells[0]), (1,))
            row, col = empty_cells[0][idx], empty_cells[1][idx]
            self.board[row, col] = 2 if torch.rand(1) < 0.9 else 4
    
    def _move_gpu(self, board, action):
        """Move tiles in the specified direction using GPU operations"""
        # Ensure board is a tensor
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float()
            
        # Store original board for comparison
        original = board.clone()
        board = board.clone()  # Work with a copy
        score_gain = 0
        
        # Rotate board based on action (0: left, 1: up, 2: right, 3: down)
        rotated = torch.rot90(board, k=action)
        
        for i in range(4):
            # Extract row and remove zeros
            row = rotated[i]
            filtered = row[row != 0]
            
            # No need to merge if 0 or 1 tile
            if len(filtered) <= 1:
                # Just put non-zero values at the beginning
                new_row = torch.zeros(4, device=board.device)
                new_row[:len(filtered)] = filtered
                rotated[i] = new_row
                continue
                
            # Create new row by merging
            merged = []
            skip = False
            
            for j in range(len(filtered) - 1):
                if skip:
                    skip = False
                    continue
                    
                if filtered[j] == filtered[j + 1]:
                    merged_val = filtered[j] * 2
                    merged.append(merged_val)
                    score_gain += int(merged_val)
                    skip = True
                else:
                    merged.append(filtered[j])
                    
            # Add the last tile if not merged
            if not skip and len(filtered) > 0:
                merged.append(filtered[-1])
                
            # Pad with zeros and update row
            new_row = torch.zeros(4, device=board.device)
            new_row[:len(merged)] = torch.tensor(merged, device=board.device)
            rotated[i] = new_row
        
        # Rotate back
        result = torch.rot90(rotated, k=-action)
        
        # Check if board changed
        changed = not torch.all(result == original)
        
        return result.cpu().numpy(), score_gain, changed
    
    def _move(self, board, action):
        """Wrapper for GPU move operation"""
        return self._move_gpu(board, action)
    
    def is_game_over(self):
        """Check if the game is over"""
        # Ensure board is a tensor
        if isinstance(self.board, np.ndarray):
            self.board = torch.from_numpy(self.board).float()
            
        # Check for empty cells
        if torch.any(self.board == 0):
            return False
        
        # Check for possible merges
        for i in range(4):
            for j in range(4):
                current = self.board[i, j]
                # Check right
                if j < 3 and current == self.board[i, j + 1]:
                    return False
                # Check down
                if i < 3 and current == self.board[i + 1, j]:
                    return False
        return True
    
    def get_valid_moves(self):
        """Get list of valid moves"""
        valid_moves = []
        for action in range(4):
            _, _, changed = self._move(self.board, action)
            if changed:
                valid_moves.append(action)
        return valid_moves
    
    def get_state(self):
        """Get current game state"""
        if isinstance(self.board, np.ndarray):
            return self.board
        return self.board.cpu().numpy()
    
    def get_score(self):
        """Get current score"""
        return self.score

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

    def count_potential_high_value_merges(self):
        """Count the number of potential merges for high-value tiles"""
        board = self.board.copy()
        bonus = 0
        
        # Check for adjacent same values (horizontally and vertically)
        for i in range(self.size):
            for j in range(self.size):
                if board[i, j] == 0:
                    continue
                value_weight = math.log2(board[i, j]) if board[i, j] > 0 else 0
                if j < self.size - 1 and board[i, j] == board[i, j + 1] and board[i, j] >= 64:
                    bonus += value_weight * 2.0
                if i < self.size - 1 and board[i, j] == board[i + 1, j] and board[i, j] >= 64:
                    bonus += value_weight * 2.0
        return bonus

    def compute_snake_pattern(self):
        """
        Calculate a score for how well the board follows a snake pattern.
        Snake pattern is a zigzag arrangement with decreasing values:
        
        high → → → →
        ↑ → → → → ↓
        ↑ ← ← ← ← ↓
        ↑ ← ← ← low
        """
        snake_path = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 3), (3, 2), (3, 1), (3, 0)
        ]
        values = []
        for i, j in snake_path:
            if self.board[i, j] > 0:
                values.append(math.log2(self.board[i, j]))
            else:
                values.append(0)
        score = 0
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                score += 1
                score += min(values[i] - values[i + 1], 3)
            else:
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
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        if self.board[i, j] == self.board[ni, nj]:
                            match_value = math.log2(self.board[i, j])
                            score += match_value
        return score

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

    def check_adjacent_512_tiles(self):
        """Check for adjacent 512 tiles and provide a substantial reward bonus"""
        board = self.board.copy()
        bonus = 0
        tile_512_positions = np.where(board == 512)
        tile_512_coords = list(zip(tile_512_positions[0], tile_512_positions[1]))
        num_512_tiles = len(tile_512_coords)
        if num_512_tiles >= 2:
            bonus += 400
            for i, (r1, c1) in enumerate(tile_512_coords):
                for j, (r2, c2) in enumerate(tile_512_coords[i+1:], i+1):
                    if (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2):
                        bonus += 1000
                        return bonus
        return bonus

    # --- NEW STATIC METHODS FOR SIMULATING MOVES ---

    @staticmethod
    def merge_row_static(row: np.ndarray, size: int) -> Tuple[np.ndarray, int, bool]:
        """
        Merge a single row in a static context.
        Returns the new row, score gained, and a flag indicating if the row changed.
        """
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
        new_row = np.pad(new_row, (0, size - len(new_row)), 'constant')
        changed = not np.array_equal(new_row, row)
        return new_row, score, changed

    @staticmethod
    def simulate_move(board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
        """
        Simulate executing a move from a given board state without modifying the internal state.
        action: 0=up, 1=right, 2=down, 3=left.
        Returns the new board, score gained, and a flag indicating if the board changed.
        """
        size = board.shape[0]
        rotated = np.rot90(board.copy(), k=action)
        total_score = 0
        changed = False
        for i in range(size):
            new_row, score, row_changed = Game2048.merge_row_static(rotated[i], size)
            if row_changed:
                changed = True
            rotated[i] = new_row
            total_score += score
        new_board = np.rot90(rotated, k=-action)
        return new_board, total_score, changed

def preprocess_state(state):
    """Enhanced state preprocessing that includes strategic features"""
    state = state.astype(np.float32)
    mask = state > 0
    state[mask] = np.log2(state[mask])
    
    # Add strategic features
    strategic_features = np.zeros_like(state)
    
    # 1. Corner importance
    corners = [(0,0), (0,3), (3,0), (3,3)]
    for corner in corners:
        if state[corner] > 0:
            strategic_features[corner] += 0.5
    
    # 2. Edge importance
    edges = [(0,1), (0,2), (1,0), (2,0), (3,1), (3,2), (1,3), (2,3)]
    for edge in edges:
        if state[edge] > 0:
            strategic_features[edge] += 0.3
    
    # 3. High-value tile importance
    max_tile = np.max(state)
    if max_tile > 0:
        high_value_mask = state >= max_tile - 1
        strategic_features[high_value_mask] += 0.4
    
    # Combine base state with strategic features
    return state + strategic_features

def preprocess_state_onehot(state):
    """
    Enhanced one-hot representation that includes strategic features
    """
    board = state.astype(np.int32)
    channels = 16
    onehot = np.zeros((channels, board.shape[0], board.shape[1]), dtype=np.float32)
    
    # Basic one-hot encoding
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] > 0:
                power = int(np.log2(board[i, j]))
                if power < channels:
                    onehot[power, i, j] = 1.0
            else:
                onehot[0, i, j] = 1.0
    
    # Add strategic features
    max_tile = np.max(board)
    if max_tile > 0:
        # Corner importance
        corners = [(0,0), (0,3), (3,0), (3,3)]
        for corner in corners:
            if board[corner] > 0:
                onehot[:, corner[0], corner[1]] *= 1.5
        
        # Edge importance
        edges = [(0,1), (0,2), (1,0), (2,0), (3,1), (3,2), (1,3), (2,3)]
        for edge in edges:
            if board[edge] > 0:
                onehot[:, edge[0], edge[1]] *= 1.3
        
        # High-value tile importance
        high_value_mask = board >= max_tile - 1
        onehot[:, high_value_mask] *= 1.4
    
    return onehot
