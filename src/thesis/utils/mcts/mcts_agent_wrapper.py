"""
MCTS Agent Wrapper that combines existing neural network agents with MCTS search.
This allows using MCTS with any existing agent model without modifying the agent code.
"""

import torch
import numpy as np
import logging
import math
import time
from .parallel_mcts import ParallelMCTS
from ...environment.game2048 import preprocess_state_onehot
from ...config import device
from .mcts import MCTS, mcts_action, analyze_position, TILE_BONUSES, MIN_EMPTY_CELLS

class MCTSAgentWrapper:
    """
    Wrapper that adds MCTS capabilities to any existing agent.
    This allows using MCTS for inference without changing the agent training code.
    """
    def __init__(self, agent, num_simulations=50, temperature=1.0, adaptive_simulations=True, use_parallel=True, num_workers=4, batch_size=16):
        """
        Initialize the MCTS wrapper.
        
        Args:
            agent: The neural network agent to wrap (any existing agent)
            num_simulations: Number of MCTS simulations to run
            temperature: Temperature for action selection (higher = more exploration)
            adaptive_simulations: Whether to adaptively adjust simulation count based on game state
            use_parallel: Whether to use parallel MCTS implementation
            num_workers: Number of parallel worker threads (for parallel MCTS)
            batch_size: Batch size for network inference (for parallel MCTS)
        """
        self.agent = agent
        self.num_simulations = num_simulations
        self.base_simulations = num_simulations  # Store the base simulation count
        self.temperature = temperature
        self.adaptive_simulations = adaptive_simulations
        self.use_parallel = use_parallel
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Create appropriate MCTS implementation
        if use_parallel:
            self.mcts = ParallelMCTS(agent, num_simulations=num_simulations, temperature=temperature, 
                                   num_workers=num_workers, batch_size=batch_size)
        else:
            self.mcts = MCTS(agent, num_simulations=num_simulations, temperature=temperature)
            
        self.inference_count = 0
        self.average_time = 0
        
        # Enhanced temperature schedule for more strategic play
        self.temperature_schedule = {
            2: 1.0,     # Early game - balanced exploration
            64: 0.8,    # Very early mid-game
            128: 0.6,   # Mid game - less exploration
            256: 0.4,   # Late early game - more focused
            512: 0.2,   # Mid-late game - highly focused
            1024: 0.1,  # Late game - almost deterministic
            2048: 0.05  # End game - highly deterministic
        }
        
        # Enhanced simulation count schedule - more aggressive scaling
        self.simulation_schedule = {
            2: 1.0,      # Base simulations for early game
            64: 1.3,     # More for early mid-game
            128: 1.6,    # More for mid-game
            256: 2.0,    # Double for late early game
            512: 2.5,    # More for mid-late game
            1024: 3.0,   # Triple for late game
            2048: 4.0    # Even more for end game
        }
        
        # Cache for previously computed actions to save time in similar states
        self.action_cache = {}
        self.cache_hits = 0
        self.cache_size_limit = 20000  # Increased cache size limit (was 10000)
        
        # Shared transposition table for all MCTS instances
        self.shared_transposition_table = {}
        
        # Performance tracking
        self.total_search_time = 0
        self.search_count = 0
        
    def set_num_simulations(self, num_simulations):
        """Change the number of MCTS simulations."""
        self.num_simulations = num_simulations
        self.base_simulations = num_simulations
        
        if self.use_parallel:
            self.mcts = ParallelMCTS(self.agent, num_simulations=num_simulations, temperature=self.temperature,
                                   num_workers=self.num_workers, batch_size=self.batch_size)
        else:
            self.mcts = MCTS(self.agent, num_simulations=num_simulations, temperature=self.temperature)
            
        # Share the transposition table
        self.mcts.transposition_table = self.shared_transposition_table
        
    def set_temperature(self, temperature):
        """Change the temperature for action selection."""
        self.temperature = temperature
        
        if self.use_parallel:
            self.mcts = ParallelMCTS(self.agent, num_simulations=self.num_simulations, temperature=temperature,
                                   num_workers=self.num_workers, batch_size=self.batch_size)
        else:
            self.mcts = MCTS(self.agent, num_simulations=self.num_simulations, temperature=temperature)
            
        # Share the transposition table
        self.mcts.transposition_table = self.shared_transposition_table
        
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
        empty_cells = np.sum(board == 0)
        
        # Find the highest threshold that's less than or equal to max_tile
        applicable_thresholds = [t for t in self.simulation_schedule.keys() if t <= max_tile]
        if applicable_thresholds:
            threshold = max(applicable_thresholds)
            multiplier = self.simulation_schedule[threshold]
        else:
            multiplier = 1.0
            
        # Additional boost for critical states (few empty cells)
        if empty_cells <= 2:
            multiplier *= 2.0  # Critical state - double simulations
        elif empty_cells <= 4:
            multiplier *= 1.5  # Near-critical state
            
        # Additional boost for high-value boards
        if max_tile >= 512:
            multiplier *= 1.2  # High-value board - 20% more simulations
            
        # Cap the maximum number of simulations to prevent excessive computation
        max_simulations = 800  # Increased from 500
        return min(int(self.base_simulations * multiplier), max_simulations)
            
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
        empty_cells = np.sum(board == 0)
        
        # Find the highest threshold that's less than or equal to max_tile
        applicable_thresholds = [t for t in self.temperature_schedule.keys() if t <= max_tile]
        if applicable_thresholds:
            threshold = max(applicable_thresholds)
            temp = self.temperature_schedule[threshold]
        else:
            temp = self.temperature
            
        # Adjust temperature based on empty cells
        # More deterministic when few empty cells (critical decisions)
        if empty_cells <= 2:
            temp *= 0.3  # Much more deterministic for very critical states
        elif empty_cells <= 4:
            temp *= 0.5  # More deterministic for critical states
        elif empty_cells >= 10:
            temp *= 1.2  # More exploration when many options available
            
        # Ensure temperature doesn't go too low or too high
        return max(0.05, min(temp, 1.5))
    
    def _get_board_hash(self, board):
        """
        Create a hash of the board state for caching.
        
        Args:
            board: Game board
            
        Returns:
            String hash of the board
        """
        return str(board.tobytes())
        
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
                
                # Check cache for previously computed actions
                board_hash = self._get_board_hash(board)
                if board_hash in self.action_cache:
                    self.cache_hits += 1
                    cached_result = self.action_cache[board_hash]
                    all_policies.append(cached_result['policy'])
                    all_values.append(cached_result['value'])
                    continue
                
                # Get adaptive parameters
                sim_count = self._get_adaptive_simulations(board)
                temp = self._get_adaptive_temperature(board)
                
                # Create MCTS with adaptive parameters
                if self.use_parallel:
                    adaptive_mcts = ParallelMCTS(self.agent, num_simulations=sim_count, temperature=temp,
                                               num_workers=self.num_workers, batch_size=self.batch_size)
                else:
                    adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
                    
                # Share the transposition table
                adaptive_mcts.transposition_table = self.shared_transposition_table
                
                # Get MCTS policy
                start_time = time.time()
                policy, root = adaptive_mcts.search(board)
                search_time = time.time() - start_time
                
                # Update performance tracking
                self.total_search_time += search_time
                self.search_count += 1
                
                # Log search statistics occasionally
                self.inference_count += 1
                self.average_time = self.total_search_time / self.search_count
                if self.inference_count % 10 == 0:
                    max_tile = np.max(board)
                    empty_cells = np.sum(board == 0)
                    logging.debug(f"MCTS search: {sim_count} simulations, max tile: {max_tile}, empty cells: {empty_cells}, time: {search_time:.2f}s, avg: {self.average_time:.2f}s, cache hits: {self.cache_hits}")
                
                if policy is None:
                    # No valid moves - return zeros
                    policy = np.zeros(4, dtype=np.float32)
                
                # Get value from search
                value = root.value() if root.visit_count > 0 else 0
                
                # Cache the result
                if len(self.action_cache) < self.cache_size_limit:
                    self.action_cache[board_hash] = {
                        'policy': policy,
                        'value': value
                    }
                
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
            
            # Check cache for previously computed actions
            board_hash = self._get_board_hash(board)
            if board_hash in self.action_cache:
                self.cache_hits += 1
                cached_result = self.action_cache[board_hash]
                policy = cached_result['policy']
                value = cached_result['value']
                
                # Convert to tensors
                policy_tensor = torch.tensor(policy, dtype=torch.float, device=device).unsqueeze(0)
                value_tensor = torch.tensor(value, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                
                # Convert policy probabilities to logits
                epsilon = 1e-8  # Small value to avoid log(0)
                policy_logits = torch.log(policy_tensor + epsilon)
                
                return policy_logits, value_tensor
            
            # Get adaptive parameters
            sim_count = self._get_adaptive_simulations(board)
            temp = self._get_adaptive_temperature(board)
            
            # Create MCTS with adaptive parameters
            if self.use_parallel:
                adaptive_mcts = ParallelMCTS(self.agent, num_simulations=sim_count, temperature=temp,
                                           num_workers=self.num_workers, batch_size=self.batch_size)
            else:
                adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
                
            # Share the transposition table
            adaptive_mcts.transposition_table = self.shared_transposition_table
            
            # Get MCTS policy
            start_time = time.time()
            policy, root = adaptive_mcts.search(board)
            search_time = time.time() - start_time
            
            # Update performance tracking
            self.total_search_time += search_time
            self.search_count += 1
            
            # Log search statistics occasionally
            self.inference_count += 1
            self.average_time = self.total_search_time / self.search_count
            if self.inference_count % 10 == 0:
                max_tile = np.max(board)
                empty_cells = np.sum(board == 0)
                logging.debug(f"MCTS search: {sim_count} simulations, max tile: {max_tile}, empty cells: {empty_cells}, time: {search_time:.2f}s, avg: {self.average_time:.2f}s, cache hits: {self.cache_hits}")
            
            if policy is None:
                # No valid moves - return zeros
                policy = np.zeros(4, dtype=np.float32)
            
            # Get value from search
            value = root.value() if root.visit_count > 0 else 0
            
            # Cache the result
            if len(self.action_cache) < self.cache_size_limit:
                self.action_cache[board_hash] = {
                    'policy': policy,
                    'value': value
                }
            
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
            
        # Check cache for previously computed actions
        board_hash = self._get_board_hash(board)
        if board_hash in self.action_cache and not deterministic:
            self.cache_hits += 1
            cached_result = self.action_cache[board_hash]
            policy = cached_result['policy']
            
            # Select action based on policy
            if deterministic:
                action = np.argmax(policy)
            else:
                # Sample from policy
                valid_policy = np.zeros_like(policy)
                if valid_moves:
                    valid_policy[valid_moves] = policy[valid_moves]
                else:
                    valid_policy = policy
                    
                if np.sum(valid_policy) > 0:
                    valid_policy /= np.sum(valid_policy)
                    action = np.random.choice(4, p=valid_policy)
                else:
                    action = np.random.choice(valid_moves) if valid_moves else np.random.randint(0, 4)
                    
            return action
            
        # Get adaptive parameters
        sim_count = self._get_adaptive_simulations(board)
        temp = self._get_adaptive_temperature(board)
        
        # For high-value states (with tiles >= 512), use more simulations
        max_tile = np.max(board)
        if max_tile >= 512:
            sim_count = int(sim_count * 1.5)  # 50% more simulations for high-value states
        elif max_tile >= 1024:
            sim_count = int(sim_count * 2.0)  # Double simulations for very high-value states
            
        # For critical states (few empty cells), use more deterministic selection
        empty_cells = np.sum(board == 0)
        if empty_cells <= MIN_EMPTY_CELLS:
            deterministic = True  # Force deterministic selection for critical states
            
        # Create MCTS with adaptive parameters
        if self.use_parallel:
            adaptive_mcts = ParallelMCTS(self.agent, num_simulations=sim_count, temperature=temp,
                                       num_workers=self.num_workers, batch_size=self.batch_size)
        else:
            adaptive_mcts = MCTS(self.agent, num_simulations=sim_count, temperature=temp)
            
        # Share the transposition table
        adaptive_mcts.transposition_table = self.shared_transposition_table
        
        # Use MCTS to select action
        start_time = time.time()
        action, policy = adaptive_mcts.get_action(board, deterministic=deterministic)
        search_time = time.time() - start_time
        
        # Update performance tracking
        self.total_search_time += search_time
        self.search_count += 1
        
        # Cache the result
        if policy is not None and len(self.action_cache) < self.cache_size_limit:
            self.action_cache[board_hash] = {
                'policy': policy,
                'value': 0.0  # We don't have the value here, but it's not used for action selection
            }
        
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
            
        # Create MCTS with shared transposition table
        if self.use_parallel:
            mcts = ParallelMCTS(self.agent, num_simulations=sims, temperature=self.temperature,
                              num_workers=self.num_workers, batch_size=self.batch_size)
        else:
            mcts = MCTS(self.agent, num_simulations=sims, temperature=self.temperature)
            
        mcts.transposition_table = self.shared_transposition_table
        
        # Analyze the position
        return analyze_position(self.agent, board, num_simulations=sims)
    
    def clear_caches(self):
        """Clear the action cache and transposition table to free memory."""
        self.action_cache = {}
        self.shared_transposition_table = {}
        self.cache_hits = 0
        logging.info("Cleared action cache and transposition table")


def wrap_agent_with_mcts(agent, num_simulations=50, temperature=1.0, adaptive_simulations=True, use_parallel=True, num_workers=4, batch_size=16):
    """
    Utility function to wrap an existing agent with MCTS capabilities.
    
    Args:
        agent: The neural network agent to wrap
        num_simulations: Number of MCTS simulations to run
        temperature: Temperature for action selection
        adaptive_simulations: Whether to use adaptive simulation counts
        use_parallel: Whether to use parallel MCTS implementation
        num_workers: Number of parallel worker threads (for parallel MCTS)
        batch_size: Batch size for network inference (for parallel MCTS)
        
    Returns:
        Wrapped agent with MCTS capabilities
    """
    return MCTSAgentWrapper(agent, num_simulations=num_simulations, temperature=temperature, 
                           adaptive_simulations=adaptive_simulations, use_parallel=use_parallel,
                           num_workers=num_workers, batch_size=batch_size) 