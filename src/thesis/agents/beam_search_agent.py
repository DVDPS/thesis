import numpy as np
import random
import torch
from typing import List, Tuple, Dict
import logging

class BeamSearchAgent:
    """
    Beam search agent for 2048 game that uses human-like heuristics to evaluate states.
    Implements the beam search algorithm with depth d=20 and beam width k=10.
    """
    def __init__(self, board_size: int = 4, beam_width: int = 10, search_depth: int = 20):
        self.board_size = board_size
        self.beam_width = beam_width
        self.search_depth = search_depth
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
    
    def get_action(self, state: np.ndarray, valid_moves: List[int]) -> int:
        """
        Get the best action using beam search.
        
        Args:
            state: Current game state
            valid_moves: List of valid moves
            
        Returns:
            Best action to take
        """
        # Convert state to tensor and move to device
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Initialize beam with current state
        beam = [(state_tensor, [], 0)]  # (state, action_history, score)
        
        # Run beam search for specified depth
        for depth in range(self.search_depth):
            new_beam = []
            
            # Expand each state in current beam
            for current_state, action_history, _ in beam:
                # Try each valid action
                for action in valid_moves:
                    # Simulate action and get next state
                    next_state = self._simulate_action(current_state, action)
                    if next_state is None:
                        continue
                        
                    # Calculate score for new state
                    score = self._evaluate_state(next_state)
                    
                    # Add to new beam with updated action history
                    new_action_history = action_history + [action]
                    new_beam.append((next_state, new_action_history, score))
            
            # Select top k states for next iteration
            if new_beam:
                beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:self.beam_width]
            else:
                break
        
        # Return first action from best state's action history
        if beam:
            best_state, best_actions, _ = beam[0]
            return best_actions[0] if best_actions else random.choice(valid_moves)
        else:
            return random.choice(valid_moves)
    
    def _simulate_action(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """
        Simulate an action on the state and return the resulting state.
        Randomly places a new tile (2 or 4) in an empty position.
        
        Args:
            state: Current game state
            action: Action to simulate (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            New state after action
        """
        # Create a copy of the state
        new_state = state.clone()
        
        # Get empty positions
        empty_positions = torch.where(new_state == 0)
        if len(empty_positions[0]) == 0:
            return None
        
        # Randomly place a new tile (2 or 4)
        idx = random.randint(0, len(empty_positions[0]) - 1)
        new_pos = (empty_positions[0][idx], empty_positions[1][idx])
        new_state[new_pos] = 2 if random.random() < 0.9 else 4
        
        return new_state
    
    def _evaluate_state(self, state: torch.Tensor) -> float:
        """
        Evaluate state using the heuristic function that considers:
        1. Number of empty tiles
        2. Maximum tile value
        3. Smoothness
        4. Monotonicity
        5. Corner strategy (prefer high values in corners)
        6. Snake pattern (prefer snake-like arrangement)
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Heuristic score
        """
        # Count empty tiles
        empty_count = torch.sum(state == 0).item()
        
        # Get maximum tile value
        max_tile = torch.max(state).item()
        
        # Calculate smoothness (penalize adjacent tiles with large differences)
        smoothness = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i, j] != 0:
                    # Check right neighbor
                    if j < self.board_size - 1:
                        smoothness -= abs(state[i, j].item() - state[i, j + 1].item())
                    # Check bottom neighbor
                    if i < self.board_size - 1:
                        smoothness -= abs(state[i, j].item() - state[i + 1, j].item())
        
        # Calculate monotonicity (prefer states where tiles are arranged in decreasing order)
        monotonicity = 0
        # Check horizontal monotonicity
        for i in range(self.board_size):
            for j in range(self.board_size - 1):
                if state[i, j].item() >= state[i, j + 1].item():
                    monotonicity += 1
                else:
                    monotonicity -= 1
        # Check vertical monotonicity
        for j in range(self.board_size):
            for i in range(self.board_size - 1):
                if state[i, j].item() >= state[i + 1, j].item():
                    monotonicity += 1
                else:
                    monotonicity -= 1
        
        # Calculate corner strategy (prefer high values in corners)
        corner_score = 0
        corners = [(0, 0), (0, self.board_size-1), (self.board_size-1, 0), (self.board_size-1, self.board_size-1)]
        for corner in corners:
            if state[corner].item() == max_tile:
                corner_score += 1
            elif state[corner].item() >= max_tile / 2:
                corner_score += 0.5
        
        # Calculate snake pattern score (prefer snake-like arrangement)
        snake_score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i, j] != 0:
                    # Check if tile is part of a snake pattern
                    if i % 2 == 0:
                        if j == 0 or state[i, j].item() >= state[i, j-1].item():
                            snake_score += 1
                    else:
                        if j == self.board_size-1 or state[i, j].item() >= state[i, j+1].item():
                            snake_score += 1
        
        # Combine all factors with adjusted weights
        weights = {
            'empty': 2.0,        # Increased weight for empty tiles
            'max_tile': 1.0,     # Increased weight for max tile
            'smoothness': 0.2,   # Increased weight for smoothness
            'monotonicity': 0.5, # Increased weight for monotonicity
            'corner': 1.5,       # New weight for corner strategy
            'snake': 0.3         # New weight for snake pattern
        }
        
        # Normalize scores
        max_possible_empty = self.board_size * self.board_size
        max_possible_smoothness = max_tile * (self.board_size * (self.board_size - 1) * 2)
        max_possible_monotonicity = (self.board_size * (self.board_size - 1) * 2)
        max_possible_corner = 4
        max_possible_snake = self.board_size * self.board_size
        
        score = (
            weights['empty'] * (empty_count / max_possible_empty) +
            weights['max_tile'] * (max_tile / 2048) +  # Normalize by target tile
            weights['smoothness'] * (smoothness / max_possible_smoothness) +
            weights['monotonicity'] * (monotonicity / max_possible_monotonicity) +
            weights['corner'] * (corner_score / max_possible_corner) +
            weights['snake'] * (snake_score / max_possible_snake)
        )
        
        return score 