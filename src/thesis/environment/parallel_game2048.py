import torch
import numpy as np
import random

class ParallelGame2048:
    def __init__(self, num_envs=64, seed=None):
        """Initialize multiple 2048 game environments"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_all()
        
    def reset_all(self):
        """Reset all environments"""
        # Initialize boards for all environments (4x4 grid)
        self.boards = torch.zeros((self.num_envs, 4, 4), dtype=torch.float32, device=self.device)
        self.scores = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Add two random tiles to each board
        self.add_random_tiles()
        self.add_random_tiles()
        
        return self.boards.clone().cpu().numpy()
        
    def add_random_tiles(self):
        """Add random tiles (2 or 4) to empty cells in all environments"""
        # Find empty cells in all environments
        empty_mask = (self.boards == 0)
        
        # Skip full boards
        active_envs = (~self.done) & torch.any(empty_mask.view(self.num_envs, -1), dim=1)
        if not torch.any(active_envs):
            return
            
        for env_idx in torch.where(active_envs)[0]:
            empty_cells = torch.where(empty_mask[env_idx])
            if len(empty_cells[0]) == 0:
                continue
                
            # Randomly select an empty cell
            idx = torch.randint(0, len(empty_cells[0]), (1,), device=self.device)
            row, col = empty_cells[0][idx], empty_cells[1][idx]
            
            # Place a 2 (90%) or 4 (10%)
            self.boards[env_idx, row, col] = 2.0 if torch.rand(1, device=self.device) < 0.9 else 4.0
    
    def _move_batch(self, boards, actions):
        """Execute moves for all environments based on action vectors"""
        batch_size = boards.shape[0]
        new_boards = boards.clone()
        scores = torch.zeros(batch_size, device=self.device)
        changed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Process each action type separately
        for action in range(4):
            # Get environments using this action
            action_mask = (actions == action)
            if not torch.any(action_mask):
                continue
                
            # Get the boards for these environments
            action_boards = boards[action_mask]
            action_indices = action_mask.nonzero().flatten()
            
            # Rotate boards based on action
            rotated = torch.rot90(action_boards, k=action)
            
            # Process each environment's board separately
            for env_idx in range(action_boards.size(0)):
                # Process each row in this environment's board
                for i in range(4):
                    row = rotated[env_idx, i].clone()
                    
                    # Skip empty rows
                    if not torch.any(row > 0):
                        continue
                    
                    # Extract non-zero elements
                    non_zero = row[row > 0]
                    if len(non_zero) == 0:
                        continue
                        
                    # Create a merged row
                    merged_row = torch.zeros_like(row)
                    merge_idx = 0
                    j = 0
                    
                    while j < len(non_zero):
                        if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                            # Merge tiles
                            merged_row[merge_idx] = non_zero[j] * 2
                            scores[action_indices[env_idx]] += merged_row[merge_idx].item()
                            j += 2
                        else:
                            # Just move the tile
                            merged_row[merge_idx] = non_zero[j]
                            j += 1
                        merge_idx += 1
                    
                    # Check if row changed
                    if not torch.equal(merged_row, row):
                        changed[action_indices[env_idx]] = True
                    
                    # Update the rotated board
                    rotated[env_idx, i] = merged_row
            
            # Rotate back to original orientation and update boards
            rotated_back = torch.rot90(rotated, k=-action)
            new_boards[action_mask] = rotated_back
        
        return new_boards, scores, changed
    
    def step(self, actions):
        """Step all environments using the provided actions"""
        # Convert numpy actions to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device)
            
        # Skip already done environments
        active_mask = ~self.done
        if not torch.any(active_mask):
            return self.boards.clone().cpu().numpy(), torch.zeros(self.num_envs, device=self.device), self.done.clone(), {}
            
        # Execute moves for active environments
        active_boards = self.boards[active_mask]
        active_actions = actions[active_mask]
        
        new_boards, rewards, changed = self._move_batch(active_boards, active_actions)
        
        # Update active boards and add random tiles
        self.boards[active_mask] = new_boards
        self.scores[active_mask] += rewards
        
        # Add new random tiles to boards that changed
        changed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        changed_mask[active_mask.nonzero().flatten()[changed]] = True
        self.add_random_tiles_mask(changed_mask)
        
        # Check for game over
        self.update_done()
        
        # Return observation, reward, done, info
        return self.boards.clone().cpu().numpy(), rewards, self.done.clone(), {"scores": self.scores.clone()}
    
    def add_random_tiles_mask(self, mask):
        """Add random tiles only to specified environments"""
        for env_idx in torch.where(mask)[0]:
            empty_cells = torch.where(self.boards[env_idx] == 0)
            if len(empty_cells[0]) == 0:
                continue
                
            # Randomly select an empty cell
            idx = torch.randint(0, len(empty_cells[0]), (1,), device=self.device)
            row, col = empty_cells[0][idx], empty_cells[1][idx]
            
            # Place a 2 (90%) or 4 (10%)
            self.boards[env_idx, row, col] = 2.0 if torch.rand(1, device=self.device) < 0.9 else 4.0
    
    def update_done(self):
        """Update done status for all environments"""
        for env_idx in range(self.num_envs):
            if self.done[env_idx]:
                continue
                
            # Check if no empty cells
            if not torch.any(self.boards[env_idx] == 0):
                # Check if no possible merges
                can_merge = False
                board = self.boards[env_idx]
                
                # Check horizontal merges
                for i in range(4):
                    for j in range(3):
                        if board[i, j] == board[i, j+1]:
                            can_merge = True
                            break
                
                # Check vertical merges
                for i in range(3):
                    for j in range(4):
                        if board[i, j] == board[i+1, j]:
                            can_merge = True
                            break
                
                # If no merges possible, game is over
                if not can_merge:
                    self.done[env_idx] = True
    
    def get_valid_moves(self, env_idx=0):
        """Get valid moves for a specific environment"""
        valid_moves = []
        board = self.boards[env_idx].clone()
        
        for action in range(4):
            temp_board = board.clone()
            rotated = torch.rot90(temp_board, k=action)
            changed = False
            
            # Process each row
            for i in range(4):
                row = rotated[i].clone()
                non_zero = row[row > 0]
                
                if len(non_zero) <= 1:
                    continue
                
                merged_row = torch.zeros_like(row)
                merge_idx = 0
                j = 0
                
                while j < len(non_zero):
                    if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                        merged_row[merge_idx] = non_zero[j] * 2
                        j += 2
                    else:
                        merged_row[merge_idx] = non_zero[j]
                        j += 1
                    merge_idx += 1
                
                if not torch.equal(merged_row, row):
                    changed = True
                    break
            
            if changed:
                valid_moves.append(action)
        
        return valid_moves

    def _move(self, state, action):
        """Execute a single move on a state"""
        # Convert state to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, device=self.device)
        
        # Rotate board based on action
        rotated = torch.rot90(state, k=action)
        score = 0
        changed = False
        
        # Process each row
        for i in range(4):
            row = rotated[i].clone()
            non_zero = row[row > 0]
            
            if len(non_zero) <= 1:
                continue
            
            merged_row = torch.zeros_like(row)
            merge_idx = 0
            j = 0
            
            while j < len(non_zero):
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged_row[merge_idx] = non_zero[j] * 2
                    score += merged_row[merge_idx].item()
                    j += 2
                else:
                    merged_row[merge_idx] = non_zero[j]
                    j += 1
                merge_idx += 1
            
            if not torch.equal(merged_row, row):
                changed = True
            
            rotated[i] = merged_row
        
        # Rotate back
        new_board = torch.rot90(rotated, k=-action)
        
        return new_board.cpu().numpy(), score, changed