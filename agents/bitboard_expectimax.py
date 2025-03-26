import numpy as np
import torch
from typing import Dict
from .bitboard2048 import Bitboard2048, add_random_tile, pack_row

class BitboardExpectimaxAgent:
    def __init__(self, depth: int = 4, use_gpu: bool = True):
        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.transposition_table = {}  # Cache for previously evaluated states
        
        # Weight matrices for heuristics
        self.corner_weight_matrix = torch.tensor([
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128]
        ], dtype=torch.float32, device=self.device)
        
        # Snake pattern weights (descending from bottom-right)
        self.snake_pattern_matrix = torch.tensor([
            [14, 13, 12, 11],
            [15, 16, 17, 10],
            [20, 19, 18, 9],
            [21, 22, 23, 8]
        ], dtype=torch.float32, device=self.device)

    def get_move(self, game_state: np.ndarray) -> int:
        """Determine the best move using expectimax search with bitboard representation"""
        # Clear transposition table at the start of each move
        self.transposition_table.clear()
        
        # Convert numpy array to bitboard
        bitboard = self._numpy_to_bitboard(game_state)
        
        max_value = float("-inf")
        best_action = 0
        actions = [0, 1, 2, 3]  # [Up, Right, Down, Left]
        
        for action in actions:
            # Try the move
            next_bitboard, reward = self._apply_move(bitboard, action)
            if next_bitboard.board != bitboard.board:  # If move is valid
                value = reward + self._expectimax(next_bitboard, self.depth - 1, is_max=False)
                if value > max_value:
                    max_value = value
                    best_action = action
        
        return best_action

    def _apply_move(self, bitboard: Bitboard2048, action: int) -> tuple[Bitboard2048, float]:
        """Apply a move to the bitboard and return the new state and reward"""
        # Store original state for verification
        original_state = bitboard.to_numpy()
        
        # Apply the move
        if action == 0:  # Up
            next_bitboard, score = bitboard.move_up()
        elif action == 1:  # Right
            next_bitboard, score = bitboard.move_right()
        elif action == 2:  # Down
            next_bitboard, score = bitboard.move_down()
        else:  # Left
            next_bitboard, score = bitboard.move_left()
        
        # Get the state after the move
        next_state = next_bitboard.to_numpy()
        
        # Verify the move was valid and actually changed the state
        if next_bitboard.board != bitboard.board:
            # Check if the move actually changed the state
            if np.array_equal(original_state, next_state):
                return bitboard, 0
            
            # Add random tile only if the move was valid and changed the state
            next_bitboard = add_random_tile(next_bitboard)
            
            # Verify the random tile was added correctly
            final_state = next_bitboard.to_numpy()
            empty_cells_before = np.sum(original_state == 0)
            empty_cells_after = np.sum(final_state == 0)
            if empty_cells_after >= empty_cells_before:
                print(f"Warning: Random tile not added after move {['Up', 'Right', 'Down', 'Left'][action]}")
            
            return next_bitboard, score
        
        return bitboard, 0  # Return original state if move was invalid

    def _expectimax(self, bitboard: Bitboard2048, depth: int, is_max: bool) -> float:
        """Expectimax search algorithm using bitboard representation"""
        if depth == 0 or self._is_terminal(bitboard):
            return self._evaluate_state(bitboard)
        
        state_hash = bitboard.board
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]
        
        if is_max:  # Player's move: maximize reward
            value = float("-inf")
            for action in [0, 1, 2, 3]:
                next_bitboard, reward = self._apply_move(bitboard, action)
                if next_bitboard.board != bitboard.board:  # If move is valid
                    value = max(value, reward + self._expectimax(next_bitboard, depth - 1, is_max=False))
        else:  # Chance node: average over possible new tile placements
            value = 0
            empty_cells = self._get_empty_cells(bitboard)
            if not empty_cells:
                return self._evaluate_state(bitboard)
            
            p = 1.0 / len(empty_cells)
            for i, j in empty_cells:
                # Consider tile 2 with 90% probability
                next_bitboard = self._add_tile(bitboard, i, j, 1)  # 1 represents 2^1 = 2
                value += 0.9 * p * self._expectimax(next_bitboard, depth - 1, is_max=True)
                
                # Consider tile 4 with 10% probability
                next_bitboard = self._add_tile(bitboard, i, j, 2)  # 2 represents 2^2 = 4
                value += 0.1 * p * self._expectimax(next_bitboard, depth - 1, is_max=True)
        
        self.transposition_table[state_hash] = value
        return value

    def _is_terminal(self, bitboard: Bitboard2048) -> bool:
        """Check if the state is terminal (no moves possible)"""
        for action in range(4):
            next_bitboard, _ = self._apply_move(bitboard, action)
            if next_bitboard.board != bitboard.board:
                return False
        return True

    def _evaluate_state(self, bitboard: Bitboard2048) -> float:
        """Evaluate a state using multiple heuristics"""
        state = bitboard.to_numpy()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Apply tile downgrading for large tiles
        max_tile = torch.max(state_tensor)
        if max_tile > 1024:  # If we have tiles larger than 1024
            state_tensor = self._apply_tile_downgrading(state_tensor)
        
        # Corner strategy score (weighted more heavily)
        corner_score = torch.sum(state_tensor * self.corner_weight_matrix)
        
        # Snake pattern score (helps maintain order)
        snake_score = torch.sum(state_tensor * self.snake_pattern_matrix)
        
        # Free cells score (important for maneuverability)
        empty_cells = torch.sum(state_tensor == 0)
        free_cells_score = empty_cells * 30.0  # Increased weight for free cells
        
        # Monotonicity score (how well values decrease from corner)
        monotonicity_score = self._calculate_monotonicity(state_tensor)
        
        # Smoothness score (how similar adjacent tiles are)
        smoothness_score = self._calculate_smoothness(state_tensor)
        
        # Merge potential score (how many tiles can be merged)
        merge_potential = self._calculate_merge_potential(state_tensor)
        
        # Additional heuristic: prefer states with higher tiles in the corner
        corner_tile_score = 0
        if state_tensor[0, 3] > 0:  # Top-right corner
            corner_tile_score = torch.log2(state_tensor[0, 3]) * 100
        
        # Combine scores with adjusted weights
        total_score = (
            corner_score * 3.0 +          # Corner strategy is most important
            snake_score * 2.0 +           # Snake pattern helps maintain order
            free_cells_score * 1.5 +      # Free cells are important but not critical
            monotonicity_score * 1.2 +    # Monotonicity helps with merges
            smoothness_score * 0.8 +      # Smoothness is a secondary concern
            merge_potential * 1.5 +       # Merge potential is important for progress
            corner_tile_score             # High corner tile is very valuable
        )
        
        return total_score.item()
    
    def _calculate_monotonicity(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate how monotonic the state is (decreasing from corner)"""
        monotonicity_score = torch.tensor(0.0, device=self.device)
        
        # For rows (decreasing from right to left)
        for i in range(4):
            for j in range(3):
                if state[i, j+1] >= state[i, j] and state[i, j+1] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i, j+1])
                elif state[i, j] > state[i, j+1] and state[i, j] > 0 and state[i, j+1] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        
        # For columns (decreasing from bottom to top)
        for j in range(4):
            for i in range(3):
                if state[i+1, j] >= state[i, j] and state[i+1, j] > 0 and state[i, j] > 0:
                    monotonicity_score += torch.log2(state[i+1, j])
                elif state[i, j] > state[i+1, j] and state[i, j] > 0 and state[i+1, j] > 0:
                    monotonicity_score -= torch.log2(state[i, j])
        
        return monotonicity_score
    
    def _calculate_smoothness(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate smoothness of the state (adjacent tiles should be similar)"""
        smoothness_score = torch.tensor(0.0, device=self.device)
        
        # For rows
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j+1] > 0:
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i, j+1]))
        
        # For columns
        for j in range(4):
            for i in range(3):
                if state[i, j] > 0 and state[i+1, j] > 0:
                    smoothness_score -= torch.abs(torch.log2(state[i, j]) - torch.log2(state[i+1, j]))
        
        return smoothness_score
    
    def _calculate_merge_potential(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate potential for future merges"""
        merge_score = torch.tensor(0.0, device=self.device)
        
        # Check horizontal merges
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j] == state[i, j+1]:
                    merge_score += torch.log2(state[i, j])
        
        # Check vertical merges
        for i in range(3):
            for j in range(4):
                if state[i, j] > 0 and state[i, j] == state[i+1, j]:
                    merge_score += torch.log2(state[i, j])
        
        return merge_score

    def _apply_tile_downgrading(self, state: torch.Tensor) -> torch.Tensor:
        """Apply tile downgrading to handle large tiles"""
        max_tile = torch.max(state)
        unique_tiles = torch.unique(state[state > 0])
        
        # Find the largest missing tile
        missing_tiles = []
        potential_tile = 2
        while potential_tile < max_tile:
            if potential_tile not in unique_tiles:
                missing_tiles.append(potential_tile)
            potential_tile *= 2
        
        if not missing_tiles:
            return state
        
        largest_missing = max(missing_tiles)
        
        # Create downgraded state
        downgraded_state = state.clone()
        downgraded_state[state > largest_missing] = state[state > largest_missing] / 2
        
        return downgraded_state

    def _numpy_to_bitboard(self, state: np.ndarray) -> Bitboard2048:
        """Convert numpy array to bitboard representation"""
        new_board = np.uint64(0)
        for r in range(4):
            row_vals = []
            for c in range(4):
                val = state[r, c]
                if val == 0:
                    row_vals.append(0)
                else:
                    # Convert value to exponent (e.g., 2 -> 1, 4 -> 2, 8 -> 3)
                    exp = int(val).bit_length() - 1
                    if exp < 0:  # Handle edge case where value is 1
                        exp = 0
                    row_vals.append(exp)
            
            # Pack the row values into a 16-bit integer
            row_packed = pack_row(row_vals)
            new_board |= (np.uint64(row_packed) << (16 * r))
        
        # Create bitboard and verify conversion
        bitboard = Bitboard2048(new_board)
        converted_state = bitboard.to_numpy()
        
        # Only print warning if there's a mismatch
        if not np.array_equal(state, converted_state):
            print("Warning: State conversion mismatch!")
        
        return bitboard

    def _get_empty_cells(self, bitboard: Bitboard2048) -> list[tuple[int, int]]:
        """Get list of empty cell coordinates"""
        state = bitboard.to_numpy()
        return [(i, j) for i in range(4) for j in range(4) if state[i, j] == 0]

    def _add_tile(self, bitboard: Bitboard2048, i: int, j: int, exponent: int) -> Bitboard2048:
        """Add a tile with given exponent at position (i,j)"""
        state = bitboard.to_numpy()
        state[i, j] = 1 << exponent
        return self._numpy_to_bitboard(state)

    def _pack_row(self, tiles: list[int]) -> int:
        """Pack a row of 4-bit values into a 16-bit integer"""
        result = 0
        for i, tile in enumerate(tiles):
            result |= (tile & 0xF) << (4 * i)
        return result 