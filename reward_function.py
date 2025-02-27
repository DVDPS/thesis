import numpy as np
import math
from game2048 import Game2048, compute_monotonicity

def compute_snake_pattern(board):
    """
    Calculate how well the board follows a snake pattern.
    Snake pattern is a zigzag arrangement with decreasing values.
    """
    # Define the ideal snake path
    snake_path = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0)
    ]
    
    # Get values along the snake path
    values = []
    for i, j in snake_path:
        if board[i, j] > 0:
            values.append(math.log2(board[i, j]))
        else:
            values.append(0)
    
    # Calculate monotonicity score along the path
    score = 0
    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            score += 1
        else:
            score -= 0.5
    
    return score

def improved_step(self, action):
    """
    A more effective reward function for 2048.
    
    Rewards:
    1. Merging tiles (with exponential rewards for higher tiles)
    2. Creating new max tiles
    3. Maintaining empty spaces
    4. Following monotonic patterns (like the snake pattern)
    5. Having tiles arranged in a corner
    
    Penalties:
    1. Invalid moves
    2. Game over
    """
    # Store old state for comparison
    old_board = self.board.copy()
    old_max_tile = np.max(old_board)
    old_empty_count = np.sum(old_board == 0)
    
    # Execute the move
    new_board, score_gain, valid_move = self._move(self.board, action)
    
    if valid_move:
        self.board = new_board
        self.score += score_gain
        self.add_random_tile()
    
    # Get new state info
    new_max_tile = np.max(self.board)
    new_empty_count = np.sum(self.board == 0)
    
    # --- REWARD CALCULATION ---
    
    # 1. Base reward from score gain (merged tiles)
    reward = score_gain * 0.1  # Scale down raw score for better balance
    
    # 2. New max tile bonus (exponential scale)
    if new_max_tile > old_max_tile:
        tile_level = int(np.log2(new_max_tile))
        # Exponential bonus: higher tiles get much higher rewards
        # 8 -> 9 points, 16 -> 16 points, 32 -> 25 points, 64 -> 36 points, etc.
        reward += tile_level ** 2
    
    # 3. Empty cells bonus - more empty cells is better (room to maneuver)
    empty_bonus = 0.5 * new_empty_count
    
    # 4. Monotonicity bonus - reward organized boards
    mono_bonus = 0.2 * compute_monotonicity(self.board)
    
    # 5. Snake pattern bonus - reward the snake/zigzag formation
    snake_bonus = 0.5 * compute_snake_pattern(self.board)
    
    # 6. Corner utilization - reward having high values in corners
    corner_values = np.array([
        self.board[0, 0], self.board[0, 3],
        self.board[3, 0], self.board[3, 3]
    ])
    corner_bonus = 0.1 * np.sum(corner_values > 0) * np.log2(np.max(corner_values) + 1)
    
    # Penalties
    # 1. Invalid move penalty
    if not valid_move:
        reward -= 1.0  # Mild penalty for invalid moves
    
    # 2. Game over penalty (with scaling)
    if self.is_game_over():
        # Scale penalty based on achievement - less penalty for higher max tiles
        max_tile_level = int(np.log2(new_max_tile)) if new_max_tile > 0 else 0
        game_over_penalty = max(50 - max_tile_level * 3, 10)
        reward -= game_over_penalty
    
    # Add all components
    reward += empty_bonus + mono_bonus + snake_bonus + corner_bonus
    
    # Create info dict
    info = {
        'score': self.score,
        'max_tile': new_max_tile,
        'valid_move': valid_move,
        'empty_cells': new_empty_count,
        'merge_score': score_gain,
        'empty_bonus': empty_bonus,
        'monotonicity_bonus': mono_bonus,
        'snake_bonus': snake_bonus,
        'corner_bonus': corner_bonus
    }
    
    return self.board.copy(), reward, self.is_game_over(), info

def apply_reward_function(game_class):
    """Apply the improved reward function to the Game2048 class"""
    game_class.step = improved_step
    game_class.compute_snake_pattern = compute_snake_pattern
    return game_class