# Path: /src/thesis/utils/mcts/parallel_mcts.py
"""
Parallel Monte Carlo Tree Search (MCTS) implementation for 2048 game.
Optimized for H100 GPUs with batch inference.
"""

import math
import numpy as np
import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from ...environment.game2048 import preprocess_state_onehot, Game2048
from ...config import device
from .mcts import MCTSNode, MCTS, C_PUCT, DIRICHLET_ALPHA, MAX_DEPTH, TILE_BONUSES, MIN_EMPTY_CELLS

class ParallelMCTS(MCTS):
    """
    Parallel Monte Carlo Tree Search implementation optimized for H100 GPUs.
    Uses batch inference to amortize GPU computation costs.
    """
    def __init__(self, agent, num_simulations=100, temperature=1.0, num_workers=4, batch_size=16):
        """
        Initialize Parallel MCTS with the given neural network agent.
        
        Args:
            agent: Neural network agent that provides policy and value predictions
            num_simulations: Number of simulations to run for each search
            temperature: Temperature for action selection
            num_workers: Number of parallel worker threads
            batch_size: Batch size for network inference
        """
        super(ParallelMCTS, self).__init__(agent, num_simulations, temperature)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.inference_queue = Queue()
        self.inference_results = {}
        self.worker_locks = {}
        self.global_lock = threading.Lock()
    
    def _batch_inference(self, states):
        """
        Run batch inference on a list of states.
        
        Args:
            states: List of state tensors
            
        Returns:
            Tuple of (policies, values)
        """
        if not states:
            return [], []
        
        # Stack states into a batch
        batch = torch.stack(states)
        
        # Run inference on the batch
        with torch.no_grad():
            policy_logits, values = self.agent(batch)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()
        
        return policies, values
    
    def _inference_worker(self):
        """Worker thread for batch inference"""
        while True:
            # Collect states for batch inference
            states = []
            state_ids = []
            
            # Get first item
            state_id, state = self.inference_queue.get()
            if state_id is None:  # Signal to stop
                self.inference_queue.task_done()
                break
                
            states.append(state)
            state_ids.append(state_id)
            self.inference_queue.task_done()
            
            # Collect more items up to batch size
            while len(states) < self.batch_size:
                try:
                    state_id, state = self.inference_queue.get_nowait()
                    if state_id is None:  # Signal to stop
                        self.inference_queue.task_done()
                        break
                        
                    states.append(state)
                    state_ids.append(state_id)
                    self.inference_queue.task_done()
                except:
                    # Queue is empty
                    break
            
            # Run batch inference
            policies, values = self._batch_inference(states)
            
            # Store results
            with self.global_lock:
                for i, state_id in enumerate(state_ids):
                    self.inference_results[state_id] = (policies[i], values[i][0])
                    if state_id in self.worker_locks:
                        self.worker_locks[state_id].release()
    
    def _simulation_worker(self, root, root_state, worker_id, num_simulations):
        """
        Worker thread for running MCTS simulations.
        
        Args:
            root: Root node of the search tree
            root_state: Root game state
            worker_id: ID of the worker thread
            num_simulations: Number of simulations to run
        """
        # Create a local copy of the environment
        env = Game2048()
        
        for _ in range(num_simulations):
            # Clone the environment for this simulation
            env.board = np.copy(root_state)
            
            # Selection phase - traverse tree to a leaf node
            node = root
            search_path = [node]
            current_depth = 0
            
            # Add virtual loss to the root to encourage diversity
            with self.global_lock:
                node.add_virtual_loss()
            
            # Select child nodes until we reach an unexpanded node
            while node.expanded() and current_depth < MAX_DEPTH:
                with self.global_lock:
                    action, node = node.select_child(depth=current_depth)
                
                # Apply the action to the environment
                if action != -1:
                    _, reward, done, info = env.step(action)
                    
                    # Store the immediate reward
                    node.reward = reward
                    
                    # Update max tile information
                    current_max_tile = np.max(env.board)
                    node.max_tile = max(node.max_tile, current_max_tile)
                    
                    # Update empty cells information
                    node.empty_cells = np.sum(env.board == 0)
                    
                    # Update board hash
                    node.board_hash = self._get_board_hash(env.board)
                    
                    if done:
                        break
                    
                # Add virtual loss to encourage thread diversity
                with self.global_lock:
                    node.add_virtual_loss()
                
                search_path.append(node)
                current_depth += 1
            
            # Check for end of game
            if env.is_game_over():
                # Game over - use a more sophisticated terminal value
                max_tile = np.max(env.board)
                
                # Use the tile bonuses for terminal state evaluation
                if max_tile in TILE_BONUSES:
                    value = TILE_BONUSES[max_tile] - 10  # Base penalty for game over
                else:
                    # Logarithmic reward based on max tile with reduced penalty
                    value = math.log2(max_tile) - 10
            elif not node.expanded():
                # Check if this state is in the transposition table
                board_hash = self._get_board_hash(env.board)
                
                with self.global_lock:
                    if board_hash in self.transposition_table:
                        # Reuse existing node from transposition table
                        existing_node = self.transposition_table[board_hash]
                        if existing_node.visit_count > 0:
                            value = existing_node.value()
                            
                            # Add a small bonus for finding a known good state
                            if existing_node.max_tile >= 256:
                                value += 0.5
                                
                            # Backup the value and continue to next simulation
                            for node in reversed(search_path):
                                with self.global_lock:
                                    node.backup(value)
                            continue
                
                # Expansion phase - expand the node if it's not expanded
                valid_actions = env.get_possible_moves()
                
                if valid_actions:
                    # Get network predictions for this state
                    state_tensor = torch.tensor(
                        preprocess_state_onehot(env.board), 
                        dtype=torch.float, 
                        device=device
                    )
                    
                    # Create a unique ID for this inference request
                    state_id = f"worker_{worker_id}_{time.time()}_{np.random.randint(1000000)}"
                    
                    # Create a lock for this request
                    self.worker_locks[state_id] = threading.Lock()
                    self.worker_locks[state_id].acquire()
                    
                    # Queue the state for batch inference
                    self.inference_queue.put((state_id, state_tensor))
                    
                    # Wait for inference result
                    self.worker_locks[state_id].acquire()
                    
                    # Get result
                    policy, value_scalar = self.inference_results[state_id]
                    value = value_scalar * 15.0  # Scale value
                    
                    # Clean up
                    del self.inference_results[state_id]
                    del self.worker_locks[state_id]
                    
                    # Get maximum tile and empty cells in the current state
                    current_max_tile = np.max(env.board)
                    current_empty_cells = np.sum(env.board == 0)
                    
                    # Add a bonus based on the maximum tile
                    if current_max_tile in TILE_BONUSES:
                        tile_bonus = TILE_BONUSES[current_max_tile]
                    else:
                        # For tiles not in the bonus table, use logarithmic scale
                        tile_bonus = 0.3 * math.log2(current_max_tile) if current_max_tile > 0 else 0
                    value += tile_bonus
                    
                    # Add a bonus for empty cells
                    empty_cells_bonus = 0.2 * current_empty_cells
                    if current_empty_cells >= MIN_EMPTY_CELLS:
                        empty_cells_bonus *= 1.5
                    value += empty_cells_bonus
                    
                    # Add a bonus for monotonicity
                    monotonicity_bonus = self._calculate_monotonicity(env.board)
                    value += monotonicity_bonus
                    
                    # Expand the node with network priors
                    priors = {a: policy[a] for a in range(4)}
                    
                    with self.global_lock:
                        node.expand(valid_actions, priors, max_tile=current_max_tile, 
                                  empty_cells=current_empty_cells, board_hash=board_hash)
                        # Add to transposition table
                        self.transposition_table[board_hash] = node
                else:
                    # No valid moves - use a negative value
                    value = -10
            
            # Backup phase - propagate the value up the tree
            for node in reversed(search_path):
                with self.global_lock:
                    node.backup(value)
    
    def search(self, root_state):
        """
        Perform parallel MCTS search from the given root state.
        
        Args:
            root_state: The game state to search from
            
        Returns:
            Policy for each action, represented as visit counts
        """
        # Start the inference worker thread
        inference_thread = threading.Thread(target=self._inference_worker)
        inference_thread.daemon = True
        inference_thread.start()
        
        # Create a fresh game for simulations
        env = Game2048()
        env.board = np.copy(root_state)
        
        # Initialize root node
        root = MCTSNode()
        
        # Get valid actions and initial policy from agent
        valid_actions = env.get_possible_moves()
        if not valid_actions:
            # Signal inference worker to stop
            self.inference_queue.put((None, None))
            return None, None  # No valid moves
        
        # Get policy and value from neural network
        state_tensor = torch.tensor(
            preprocess_state_onehot(root_state), 
            dtype=torch.float, 
            device=device
        )
        
        with torch.no_grad():
            policy_logits, value = self.agent(state_tensor.unsqueeze(0))
            # Apply softmax to get probabilities
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Get maximum tile and empty cells in the current state
        max_tile = np.max(root_state)
        empty_cells = np.sum(root_state == 0)
        board_hash = self._get_board_hash(root_state)
        
        # Expand root with valid actions and policy priors
        priors = {a: policy[a] for a in range(4)}
        root.expand(valid_actions, priors, max_tile=max_tile, empty_cells=empty_cells, board_hash=board_hash)
        
        # Add root to transposition table
        self.transposition_table[board_hash] = root
        
        # Calculate simulations per worker
        simulations_per_worker = [self.num_simulations // self.num_workers] * self.num_workers
        # Distribute any remainder
        for i in range(self.num_simulations % self.num_workers):
            simulations_per_worker[i] += 1
        
        # Start worker threads
        workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._simulation_worker,
                args=(root, root_state, i, simulations_per_worker[i])
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        # Signal inference worker to stop
        self.inference_queue.put((None, None))
        inference_thread.join()
        
        # Calculate improved policy based on visit counts
        improved_policy = np.zeros(4, dtype=np.float32)
        
        if self.temperature == 0:
            # Deterministic policy - choose the most visited action
            visits = [child.visit_count for action, child in root.children.items()]
            best_action = list(root.children.keys())[int(np.argmax(visits))]
            improved_policy[best_action] = 1.0
        else:
            # Stochastic policy based on visit counts and temperature
            visits = np.array([child.visit_count for action, child in sorted(root.children.items())])
            actions = np.array(sorted(root.children.keys()))
            
            if sum(visits) > 0:
                # Apply temperature and normalize
                if self.temperature == 1.0:
                    probs = visits / sum(visits)
                else:
                    # Sharpen the distribution with temperature
                    visits = visits ** (1.0 / self.temperature)
                    probs = visits / sum(visits)
                
                # Set policy for each action
                for action, prob in zip(actions, probs):
                    improved_policy[action] = prob
        
        return improved_policy, root