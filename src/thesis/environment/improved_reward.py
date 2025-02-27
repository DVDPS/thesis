import numpy as np
import math
from .game2048 import Game2048, compute_monotonicity

def analyze_snake_pattern(board):
    """
    Analyze how well the board follows the snake pattern, with more lenient
    evaluation that is suitable for early learning stages.
    
    The ideal snake pattern looks like:
    
    high → → → →
    ←  ← ← ← ↓
    ↑  → → → ↓
    ↑  ← ← ← low
    
    Returns a more forgiving score to help guide early learning.
    """
    # Define the ideal snake path
    snake_path = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0)
    ]
    
    # Extract values along the snake path
    values = []
    for i, j in snake_path:
        if board[i, j] > 0:
            values.append(math.log2(board[i, j]))
        else:
            values.append(0)
    
    # Count how many adjacent pairs follow the decreasing pattern
    score = 0
    pattern_broken = False
    
    # Check for monotonically decreasing values along path - more lenient evaluation
    for i in range(len(values) - 1):
        # Reward decreases (good pattern following)
        if values[i] > values[i + 1]:
            # Less aggressive reward for decreases
            diff = values[i] - values[i + 1]
            score += diff * 1.5  # Reduced from 2
        # Penalize increases (pattern breaking), but less severely
        elif values[i] < values[i + 1]:
            # Only consider significant increases as pattern-breaking
            # More lenient position weighting
            position_weight = max(0.5, 1.5 - i/8)  # Much less aggressive penalty
            
            # Only mark as pattern broken if it's a large increase in a critical position
            if i < 6 and values[i] > 0 and values[i+1] > 0 and (values[i+1] - values[i] > 2):
                pattern_broken = True
                
            # More moderate penalty for increases
            diff = values[i + 1] - values[i]
            score -= diff * 1.5 * position_weight  # Reduced from 3
    
    # Less aggressive reward for having highest values at the beginning
    highest_value_pos = values.index(max(values))
    if highest_value_pos == 0:  # Highest value in top-left corner
        score += 5  # Reduced from 10
    elif highest_value_pos < 4:  # Highest value in top row
        score += 3  # Reduced from 5
    elif highest_value_pos > 12:  # Highest value near end of snake (bad)
        score -= 2  # Reduced from 5
    
    # Check if the top-left corner has a significant value
    if values[0] >= 7:  # 128 or higher
        score += values[0]  # Reduced from values[0] * 2
    
    return score, pattern_broken

def compute_merge_potential(board):
    """
    Calculate the potential for future merges (adjacency of similar values).
    Higher score means more potential future merges.
    """
    score = 0
    size = board.shape[0]
    
    for i in range(size):
        for j in range(size):
            if board[i, j] == 0:
                continue
                
            # Check horizontally and vertically for same values
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if board[i, j] == board[ni, nj]:
                        # Award more points for higher value matching pairs
                        match_value = math.log2(board[i, j])
                        score += match_value
    
    return score

def improved_step(self, action):
    """
    Enhanced reward function with more balanced rewards and penalties.
    
    Key changes:
    1. Reduced penalties for breaking the snake pattern
    2. More moderate scaling of rewards and penalties
    3. Focus on positive reinforcement
    4. Less volatile reward structure overall
    """
    # Store old state for comparison
    old_board = self.board.copy()
    old_max_tile = np.max(old_board)
    old_empty_count = np.sum(old_board == 0)
    old_snake_score, _ = analyze_snake_pattern(old_board)
    
    # Execute the move
    new_board, score_gain, valid_move = self._move(self.board, action)
    
    if valid_move:
        self.board = new_board
        self.score += score_gain
        self.add_random_tile()
    
    # Get new state info
    new_max_tile = np.max(self.board)
    new_empty_count = np.sum(self.board == 0)
    new_snake_score, pattern_broken = analyze_snake_pattern(self.board)
    
    # === REWARD CALCULATION (MORE BALANCED) ===
    
    # 1. Base reward from merges (moderately scaled)
    if score_gain > 0:
        # Less aggressive scaling for merge rewards
        scaled_score = math.log(score_gain + 1) * 2.5
        reward = scaled_score
    else:
        reward = 0
    
    # 2. New max tile bonus (less steep exponential)
    if new_max_tile > old_max_tile:
        tile_level = int(np.log2(new_max_tile))
        # More moderate exponential scaling: still rewards higher tiles more but not as extremely
        reward += tile_level * 3
        
        # Milestone bonuses for key tiles - still significant but not as extreme
        if new_max_tile == 256:
            reward += 25  # Reduced bonus for 256
        elif new_max_tile == 512:
            reward += 50  # Reduced bonus for 512
        elif new_max_tile == 1024:
            reward += 100  # Reduced bonus for 1024
        elif new_max_tile == 2048:
            reward += 250  # Reduced bonus for 2048
    
    # 3. Snake pattern rewards/penalties - significantly reduced penalties
    # Less weight on absolute snake score
    snake_bonus = new_snake_score * 0.5
    
    # Less aggressive reward for improving snake pattern
    if new_snake_score > old_snake_score:
        snake_improvement = (new_snake_score - old_snake_score) * 1.0  # Reduced multiplier
        snake_bonus += snake_improvement
    
    # Significantly reduced penalty for breaking a good snake pattern
    if pattern_broken and old_snake_score > 10:
        snake_bonus -= 10  # Much lower penalty (was 30)
    
    # 4. Empty cells bonus - critical for maintaining playability
    empty_bonus = new_empty_count * 1.5  # Increased to encourage keeping spaces open
    
    # 5. Merge potential - moderate reward for future merges
    merge_potential = compute_merge_potential(self.board) * 0.75  # Slightly increased
    
    # 6. Corner utilization - moderate bonus for corners
    corner_values = [
        self.board[0, 0], self.board[0, 3],
        self.board[3, 0], self.board[3, 3]
    ]
    max_corner = max(corner_values)
    if max_corner > 0:
        corner_bonus = np.log2(max_corner) * 1.5  # Reduced multiplier
        # Moderate bonus if highest value is in top-left
        if self.board[0, 0] == max_corner and max_corner >= 64:
            corner_bonus *= 1.5  # Reduced multiplier (was 2)
    else:
        corner_bonus = 0
    
    # Add all positive components
    reward += snake_bonus + empty_bonus + merge_potential + corner_bonus
    
    # === PENALTIES (REDUCED) ===
    
    # Invalid move penalty - more moderate
    if not valid_move:
        reward -= 2.0  # Reduced penalty (was 5.0)
    
    # Game over penalty (less extreme)
    if self.is_game_over():
        # Reduce penalty if we achieved high tiles - much more moderate
        max_tile_level = int(np.log2(new_max_tile)) if new_max_tile > 0 else 0
        game_over_penalty = max(40 - max_tile_level * 3, 10)  # Reduced penalty
        reward -= game_over_penalty
    
    # Create info dict with components for debugging
    info = {
        'score': self.score,
        'max_tile': new_max_tile,
        'valid_move': valid_move,
        'empty_cells': new_empty_count,
        'merge_score': score_gain,
        'empty_bonus': empty_bonus,
        'snake_bonus': snake_bonus,
        'merge_potential': merge_potential,
        'corner_bonus': corner_bonus,
        'snake_pattern_broken': pattern_broken
    }
    
    return self.board.copy(), reward, self.is_game_over(), info

def apply_improved_reward(game_class):
    """Apply the improved reward function to the Game2048 class"""
    # Add new methods
    game_class.analyze_snake_pattern = analyze_snake_pattern
    game_class.compute_merge_potential = compute_merge_potential
    
    # Replace the step method with our improved version
    game_class.step = improved_step
    
    return game_class