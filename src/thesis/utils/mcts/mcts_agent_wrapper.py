"""
MCTS Agent Wrapper that combines existing neural network agents with MCTS search.
This allows using MCTS with any existing agent model without modifying the agent code.
"""

import torch
import numpy as np
import logging
import math
import time
from ...environment.game2048 import preprocess_state_onehot
from ...config import device
from .mcts import MCTS, mcts_action, analyze_position

class MCTSAgentWrapper:
    """
    Wrapper that adds MCTS capabilities to any existing agent.
    This allows using MCTS for inference without changing the agent training code.
    """
    def __init__(self, agent, num_simulations=50, temperature=1.0, adaptive_simulations=True):
        """
        Initialize the MCTS wrapper.
        
        Args:
            agent: The neural network agent to wrap (any existing agent)
            num_simulations: Number of MCTS simulations to run
            temperature: Temperature for action selection (higher = more exploration)
            adaptive_simulations: Whether to adaptively adjust simulation count based on game state
        """
        self.agent = agent
        self.num_simulations = num_simulations
        self.base_simulations = num_simulations  # Store the base simulation count
        self.temperature = temperature
        self.adaptive_simulations = adaptive_simulations
        self.mcts = MCTS(agent, num_simulations=num_simulations, temperature=temperature)
        self.inference_count = 0
        self.average_time = 0
        self.temperature_schedule = {
            2: 1.0,     # Early game - more exploration
            128: 0.8,   # Mid game - moderate exploration
            256: 0.5,   # Late early game - less exploration
            512: 0.3,   # Mid-late game - minimal exploration
            1024: 0.1   # Late game - almost deterministic
        }
        
    def set_num_simulations(self, num_simulations):
        """Change the number of MCTS simulations."""
        self.num_simulations = num_simulations
        self.base_simulations = num_simulations
        self.mcts = MCTS(self.agent, num_simulations=num_simulations, temperature=self.temperature)
        
    def set_temperature(self, temperature):
        """Change the temperature for action selection."""
        self.temperature = temperature
        self.mcts = MCTS(self.agent, num_simulations=self.num_simulations, temperature=temperature)
        
    def _get_adaptive_simulations(self, board):
        """
        Adaptively determine the number of simulations based on game state.
        Uses more simulations for critical states and late-game positions.
        
        Args:
            board: Current game board
            
        Returns:
            Number of simulations to use
        """
        if not self.adaptive_simulations:
            return self.num_simulations
            
        max_tile = np.max(board)
        
        # Scale simulations based on max tile value
        # More simulations for higher-value states
        if max_tile >= 1024:
            # Late game - use 2x simulations
            return self.base_simulations * 2
        elif max_tile >= 512:
            # Mid-late game - use 1.5x simulations
            return int(self.base_simulations * 1.5)
        elif max_tile >= 256:
            # Early-mid game - use 1.2x simulations
            return int(self.base_simulations * 1.2)
        else:
            # Early game - use base simulations
            return self.base_simulations
            
    def _get_adaptive_temperature(self, board):
        """
        Adaptively determine the temperature based on game state.
        Uses lower temperature (more deterministic) for late-game positions.
        
        Args:
            board: Current game board
            
        Returns:
            Temperature to use
        """
        max_tile = np.max(board)
        
        # Find the highest threshold that's less than or equal to max_tile
        applicable_thresholds = [t for t in self.temperature_schedule.keys() if t <= max_tile]
        if applicable_thresholds:
            threshold = max(applicable_thresholds)
            return self.temperature_schedule[threshold]
        
        # Default to base temperature if no threshold applies
        return self.temperature
        
    def __call__(self, state_tensor, training=False):
        """
        Forward pass with MCTS search when not in training mode.
        During training, defers to the wrapped agent.
        
        Args:
            state_tensor: Preprocessed state tensor
            training: Whether in training mode
            
        Returns:
            Policy logits and value estimate
        """
        # During training, just use the base agent (no MCTS)
        if training:
            return self.agent(state_tensor, training=True)
        
        # For a batch of states, we need to process them individually with MCTS
        batch_size = state_tensor.shape[0]
        
        if batch_size > 1:
            # Process each sample in the batch separately
            all_policies = []
            all_values = []
            
            for i in range(batch_size):
                # Convert tensor to numpy array for game state
                sample = state_tensor[i].cpu().numpy()
                # For one-hot encoded states, convert back to board
                board = self._tensor_to_board(sample)
                
                # Get adaptive parameters
                sim_count = self._get_adaptive_simulations(board)
                temp = self._get_adaptive_temperature(board)
                
                # Create MCTS with adaptive parameters
                adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
                
                # Get MCTS policy
                start_time = time.time()
                policy, root = adaptive_mcts.search(board)
                search_time = time.time() - start_time
                
                # Log search statistics occasionally
                self.inference_count += 1
                self.average_time = (self.average_time * (self.inference_count - 1) + search_time) / self.inference_count
                if self.inference_count % 10 == 0:
                    max_tile = np.max(board)
                    logging.debug(f"MCTS search: {sim_count} simulations, max tile: {max_tile}, time: {search_time:.2f}s, avg: {self.average_time:.2f}s")
                
                if policy is None:
                    # No valid moves - return zeros
                    policy = np.zeros(4, dtype=np.float32)
                
                # Get value from search
                value = root.value() if root.visit_count > 0 else 0
                
                all_policies.append(policy)
                all_values.append(value)
            
            # Convert lists to tensors
            policy_tensor = torch.tensor(all_policies, dtype=torch.float, device=device)
            value_tensor = torch.tensor(all_values, dtype=torch.float, device=device).unsqueeze(1)
            
            # Convert policy probabilities to logits
            epsilon = 1e-8  # Small value to avoid log(0)
            policy_logits = torch.log(policy_tensor + epsilon)
            
            return policy_logits, value_tensor
        
        else:
            # Handle a single state
            # Convert tensor to numpy array for game state
            state_np = state_tensor[0].cpu().numpy()
            board = self._tensor_to_board(state_np)
            
            # Get adaptive parameters
            sim_count = self._get_adaptive_simulations(board)
            temp = self._get_adaptive_temperature(board)
            
            # Create MCTS with adaptive parameters
            adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
            
            # Get MCTS policy
            start_time = time.time()
            policy, root = adaptive_mcts.search(board)
            search_time = time.time() - start_time
            
            # Log search statistics occasionally
            self.inference_count += 1
            self.average_time = (self.average_time * (self.inference_count - 1) + search_time) / self.inference_count
            if self.inference_count % 10 == 0:
                max_tile = np.max(board)
                logging.debug(f"MCTS search: {sim_count} simulations, max tile: {max_tile}, time: {search_time:.2f}s, avg: {self.average_time:.2f}s")
            
            if policy is None:
                # No valid moves - return zeros
                policy = np.zeros(4, dtype=np.float32)
            
            # Get value from search
            value = root.value() if root.visit_count > 0 else 0
            
            # Convert to tensors
            policy_tensor = torch.tensor(policy, dtype=torch.float, device=device).unsqueeze(0)
            value_tensor = torch.tensor(value, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
            
            # Convert policy probabilities to logits
            epsilon = 1e-8  # Small value to avoid log(0)
            policy_logits = torch.log(policy_tensor + epsilon)
            
            return policy_logits, value_tensor
    
    def _tensor_to_board(self, tensor):
        """
        Convert a preprocessed state tensor back to a game board.
        Works with one-hot encoded states.
        
        Args:
            tensor: Preprocessed state tensor
            
        Returns:
            Game board as a numpy array
        """
        # Handle one-hot encoded states (shape: [channels, height, width])
        if tensor.ndim == 3 and tensor.shape[0] > 4:
            # Convert from one-hot back to board
            board = np.zeros((4, 4), dtype=np.int32)
            for i in range(1, min(16, tensor.shape[0])):  # Skip channel 0 (empty)
                mask = tensor[i] > 0.5
                board[mask] = 2 ** i
            return board
        # If already in board form
        elif tensor.shape == (4, 4):
            return tensor.astype(np.int32)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    def select_action(self, state, valid_moves=None, deterministic=False):
        """
        Select an action for the given state using MCTS.
        
        Args:
            state: Game state (either board or preprocessed tensor)
            valid_moves: Optional list of valid moves
            deterministic: Whether to select deterministically
            
        Returns:
            Selected action
        """
        # Convert state to board if it's a tensor
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            if state_np.ndim > 2:
                # One-hot encoded state
                board = self._tensor_to_board(state_np)
            else:
                board = state_np
        else:
            board = state
            
        # Get adaptive parameters
        sim_count = self._get_adaptive_simulations(board)
        temp = self._get_adaptive_temperature(board)
        
        # Create MCTS with adaptive parameters
        adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
        
        # Use MCTS to select action
        action, policy = adaptive_mcts.get_action(board, deterministic=deterministic)
        
        # If no valid moves from MCTS, but valid_moves was provided
        if action is None and valid_moves:
            # Fallback to random valid move
            action = np.random.choice(valid_moves)
            
        return action
    
    def analyze(self, state, num_simulations=None):
        """
        Analyze a position using MCTS and return detailed statistics.
        
        Args:
            state: Game state to analyze
            num_simulations: Optional override for simulation count
            
        Returns:
            Dictionary with analysis results
        """
        # Convert state to board if it's a tensor
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            if state_np.ndim > 2:
                # One-hot encoded state
                board = self._tensor_to_board(state_np)
            else:
                board = state_np
        else:
            board = state
            
        # Use analyze_position with specified or default simulation count
        if num_simulations is None:
            # Use adaptive simulation count
            sims = self._get_adaptive_simulations(board)
        else:
            sims = num_simulations
            
        return analyze_position(self.agent, board, num_simulations=sims)


def wrap_agent_with_mcts(agent, num_simulations=50, temperature=1.0, adaptive_simulations=True):
    """
    Utility function to wrap an existing agent with MCTS capabilities.
    
    Args:
        agent: The neural network agent to wrap
        num_simulations: Number of MCTS simulations to run
        temperature: Temperature for action selection
        adaptive_simulations: Whether to use adaptive simulation counts
        
    Returns:
        Wrapped agent with MCTS capabilities
    """
    return MCTSAgentWrapper(agent, num_simulations=num_simulations, temperature=temperature, adaptive_simulations=adaptive_simulations) 