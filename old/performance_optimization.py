import os
import numpy as np
import torch
import random
import numba
import concurrent.futures
from config import device
from game2048 import Game2048, preprocess_state_onehot

# Performance Optimization Recommendations

## 1. Environment Optimization

class OptimizedGame2048:
    def __init__(self, size=4, seed=None):
        self.size = size
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        
        # Precompute corner weights as a tensor for faster calculation
        self.corner_weights = torch.tensor([
            [3.0, 2.0, 2.0, 3.0],
            [2.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0, 2.0],
            [3.0, 2.0, 2.0, 3.0]
        ], dtype=torch.float, device=device)
        
        # Cache for snake pattern indices
        self.snake_path = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 3), (3, 2), (3, 1), (3, 0)
        ]
        
        # Pre-allocate buffers for faster processing
        self.empty_cells_buffer = np.zeros((self.size, self.size), dtype=bool)
        
    def _merge_row_vectorized(self, board):
        """
        Vectorized implementation of merging all rows at once
        Returns new board, score gained, and whether any row changed
        """
        # Implementation here...
        pass


## 2. Batch Processing

def collect_trajectories_batched(agent, envs, min_steps=500, num_envs=4):
    """
    Collect trajectories from multiple environments in parallel
    """
    # Initialize states, rewards, actions, etc. for all environments
    states = [env.reset() for env in envs]
    states_proc = [preprocess_state_onehot(state) for state in states]
    dones = [False] * num_envs
    
    # Collection structures
    trajectories = [[] for _ in range(num_envs)]
    total_steps = 0
    
    while total_steps < min_steps:
        # Process all states in a batch
        states_batch = torch.tensor(np.array(states_proc), 
                                   dtype=torch.float, device=device)
        
        # Get actions for all environments at once
        with torch.no_grad():
            logits_batch, values_batch = agent(states_batch)
            
            # Apply action masks for valid moves
            action_masks = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    valid_moves = env.get_possible_moves()
                    mask = torch.full((1, 4), float('-inf'), device=device)
                    mask[0, valid_moves] = 0
                    action_masks.append(mask)
                else:
                    action_masks.append(torch.zeros((1, 4), device=device))
            
            action_masks_batch = torch.cat(action_masks, dim=0)
            logits_batch = logits_batch + action_masks_batch
            
            # Sample actions
            dists = torch.distributions.Categorical(logits=logits_batch)
            actions_batch = dists.sample()
            log_probs_batch = dists.log_prob(actions_batch)
        
        # Execute actions in all environments
        next_states = []
        rewards = []
        new_dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(envs, actions_batch)):
            if not dones[i]:
                next_state, reward, done, info = env.step(action.item())
                next_states.append(next_state)
                rewards.append(reward)
                new_dones.append(done)
                infos.append(info)
                
                # Store transition
                trajectories[i].append((
                    states_proc[i],
                    action.item(),
                    reward,
                    preprocess_state_onehot(next_state),
                    done,
                    log_probs_batch[i].item(),
                    values_batch[i].item()
                ))
                
                total_steps += 1
            else:
                next_states.append(states[i])
                rewards.append(0)
                new_dones.append(True)
                infos.append({})
        
        # Update states
        states = next_states
        states_proc = [preprocess_state_onehot(state) for state in states]
        dones = new_dones
        
        # Reset environments that are done
        for i, done in enumerate(dones):
            if done:
                states[i] = envs[i].reset()
                states_proc[i] = preprocess_state_onehot(states[i])
                dones[i] = False
    
    # Flatten trajectories and return
    flat_trajectories = []
    for traj in trajectories:
        flat_trajectories.extend(traj)
    
    # Process into the expected format
    # ...
    
    return flat_trajectories


## 3. Mixed Precision Training

def apply_mixed_precision_training(agent, states, rewards, advantages, optimizer, max_grad_norm):
    """
    Example of using mixed precision training for faster computation
    """
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Forward pass with autocast
    with torch.cuda.amp.autocast():
        policy_logits, value_preds = agent(states)
        loss = compute_loss(policy_logits, value_preds, rewards, advantages)
    
    # Backward pass with gradient scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    
    # Gradient clipping and optimizer step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()


## 4. State Preprocessing Optimization

def fast_preprocess_state_onehot(state, output_buffer=None):
    """
    Optimized function to convert state to one-hot encoding
    Reuses output buffer to avoid repeated memory allocations
    """
    if output_buffer is None:
        output_buffer = np.zeros((16, 4, 4), dtype=np.float32)
    else:
        output_buffer.fill(0)
    
    # Vectorized operations
    board = state.astype(np.int32)
    mask = board > 0
    if np.any(mask):
        powers = np.log2(board[mask]).astype(int)
        
        # Using advanced indexing for faster assignment
        valid_indices = np.where(mask)
        for idx, power in zip(zip(*valid_indices), powers):
            if power < 16:  # Make sure the power fits in our channels
                output_buffer[power, idx[0], idx[1]] = 1.0
    
    # Mark empty tiles in channel 0
    output_buffer[0, board == 0] = 1.0
    
    return output_buffer


## 5. Optimized Replay Buffer Implementation

class FastPrioritizedReplayBuffer:
    """
    Optimized implementation of Prioritized Experience Replay
    Uses NumPy arrays and vectorized operations for speed
    """
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        
        # Pre-allocate arrays for all components
        self.states = np.zeros((capacity, 16, 4, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 16, 4, 4), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.ones(capacity, dtype=np.float32)
        
        # For high-value tiles
        self.high_value_buffer_size = capacity // 5
        self.high_value_states = np.zeros((self.high_value_buffer_size, 16, 4, 4), dtype=np.float32)
        self.high_value_actions = np.zeros(self.high_value_buffer_size, dtype=np.int32)
        self.high_value_rewards = np.zeros(self.high_value_buffer_size, dtype=np.float32)
        self.high_value_next_states = np.zeros((self.high_value_buffer_size, 16, 4, 4), dtype=np.float32)
        self.high_value_dones = np.zeros(self.high_value_buffer_size, dtype=np.bool_)
        self.high_value_count = 0
        self.high_value_position = 0
        
        self.position = 0
        self.size = 0
        
    def __len__(self):
        """Return current buffer size"""
        return self.size
        
    def add(self, state, action, reward, next_state, done, tile=0):
        """Add experience to buffer with max priority"""
        # Copy data to pre-allocated arrays
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Set max priority for new experience
        max_priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        self.priorities[self.position] = max_priority
        
        # If this state has a high-value tile, also add to high-value buffer
        if tile >= 256:  # Consider 256+ as high value
            self.high_value_states[self.high_value_position] = state
            self.high_value_actions[self.high_value_position] = action
            self.high_value_rewards[self.high_value_position] = reward
            self.high_value_next_states[self.high_value_position] = next_state
            self.high_value_dones[self.high_value_position] = done
            
            self.high_value_position = (self.high_value_position + 1) % self.high_value_buffer_size
            self.high_value_count = min(self.high_value_count + 1, self.high_value_buffer_size)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size, beta=0.4, high_value_ratio=0.3):
        """Sample batch with priority"""
        if self.size == 0:
            return [], np.array([]), []
            
        high_value_count = min(int(batch_size * high_value_ratio), self.high_value_count)
        regular_count = batch_size - high_value_count
        
        if self.size < regular_count:
            indices = np.random.randint(0, self.size, size=regular_count)
            weights = np.ones(regular_count, dtype=np.float32)
        else:
            # Get sampling probabilities from priorities
            priorities = self.priorities[:self.size]
            # Ensure all priorities are positive
            priorities = np.maximum(priorities, 1e-8)
            
            probs = priorities ** self.alpha
            probs_sum = probs.sum()
            
            if probs_sum <= 1e-10:
                # Fall back to uniform if all priorities are effectively zero
                probs = np.ones_like(priorities) / len(priorities)
            else:
                probs = probs / probs_sum
                
            # Sample indices
            indices = np.random.choice(self.size, regular_count, p=probs)
            
            # Calculate importance sampling weights
            weights = (self.size * probs[indices]) ** (-beta)
            # Avoid division by zero
            max_weight = weights.max()
            if max_weight > 0:
                weights = weights / max_weight
            
        # Get regular samples
        states = self.states[indices].copy()
        actions = self.actions[indices].copy()
        rewards = self.rewards[indices].copy()
        next_states = self.next_states[indices].copy()
        dones = self.dones[indices].copy()
        
        # Add high-value samples if available
        if high_value_count > 0 and self.high_value_count > 0:
            # Sample from high value buffer
            hv_count = min(high_value_count, self.high_value_count)
            if self.high_value_count <= hv_count:
                hv_indices = np.arange(self.high_value_count)
            else:
                hv_indices = np.random.choice(self.high_value_count, hv_count, replace=False)
                
            # Concatenate with regular samples
            states = np.concatenate([states, self.high_value_states[hv_indices]])
            actions = np.concatenate([actions, self.high_value_actions[hv_indices]])
            rewards = np.concatenate([rewards, self.high_value_rewards[hv_indices]])
            next_states = np.concatenate([next_states, self.high_value_next_states[hv_indices]])
            dones = np.concatenate([dones, self.high_value_dones[hv_indices]])
            
            # Add weights for high-value samples (all 1.0)
            weights = np.concatenate([weights, np.ones(hv_count, dtype=np.float32)])
        
        # Return batch as lists (for compatibility with existing code)
        return [
            states.tolist(), 
            actions.tolist(), 
            rewards.tolist(), 
            next_states.tolist(), 
            dones.tolist()
        ], weights, indices
        
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled indices"""
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = max(priority, 1e-8)  # Ensure positive priority


## 6. JIT Compilation with Numba

@numba.jit(nopython=True)
def compute_monotonicity_fast(board):
    """
    JIT-compiled version of monotonicity calculation for speed
    """
    board = board.astype(np.float32)
    # Avoid log(0) by setting zeros to 1
    safe_board = np.where(board > 0, board, 1)
    log_board = np.log2(safe_board)
    
    mono_score = 0.0
    # Rows
    for i in range(4):
        for j in range(3):
            mono_score -= abs(log_board[i, j] - log_board[i, j+1])
    
    # Columns
    for j in range(4):
        for i in range(3):
            mono_score -= abs(log_board[i, j] - log_board[i+1, j])
            
    return mono_score


## 7. Multithreading for Data Collection

def collect_trajectories_threaded(agent, num_threads=4, min_steps_per_thread=125):
    """
    Collect trajectories using multiple threads
    """
    # Create environments for each thread
    envs = [Game2048() for _ in range(num_threads)]
    
    # Function for trajectory collection in a single thread
    def collect_in_thread(env_id):
        env = envs[env_id]
        trajectories = []
        steps = 0
        
        while steps < min_steps_per_thread:
            # Collect one episode
            state = env.reset()
            done = False
            episode_traj = []
            
            while not done and steps < min_steps_per_thread:
                # Process state, get action, etc.
                # ... (code omitted for brevity)
                
                # Add to trajectory
                episode_traj.append((state, action, reward, next_state, done))
                steps += 1
                
                # Update state
                state = next_state
            
            trajectories.extend(episode_traj)
        
        return trajectories
    
    # Use ThreadPoolExecutor to run collection in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks
        futures = [executor.submit(collect_in_thread, i) for i in range(num_threads)]
        
        # Gather results
        all_trajectories = []
        for future in concurrent.futures.as_completed(futures):
            all_trajectories.extend(future.result())
    
    return all_trajectories


## 8. Memory Optimization

def memory_efficient_training(agent, env, optimizer, epochs, batch_size):
    """
    Memory-efficient training loop that processes batches incrementally
    """
    # Use a generator for experience collection
    def experience_generator():
        state = env.reset()
        while True:
            # Get action
            action = agent.get_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Yield experience
            yield (state, action, reward, next_state, done)
            
            # Update state
            state = next_state if not done else env.reset()
    
    # Create generator
    experiences = experience_generator()
    
    # Training loop
    for epoch in range(epochs):
        # Collect mini-batch
        batch = [next(experiences) for _ in range(batch_size)]
        
        # Process batch and update agent
        update_agent(agent, optimizer, batch)
        
        # Free memory explicitly
        del batch
        if epoch % 10 == 0:
            torch.cuda.empty_cache()  # Clear GPU cache periodically


## 9. Profile-Guided Optimization

def profile_training():
    """
    Profile the training function to identify bottlenecks
    """
    import cProfile
    import pstats

    # Profile training function
    cProfile.run('train(agent, env, optimizer, epochs=100)', 'train_stats')

    # Print sorted statistics
    p = pstats.Stats('train_stats')
    p.sort_stats('cumulative').print_stats(20)


## 10. GPU Utilization Optimization

def optimize_gpu_utilization():
    """
    Optimize GPU utilization settings
    """
    # Set CUDA device to most efficient settings
    if torch.cuda.is_available():
        # Enable TF32 precision on Ampere GPUs (faster than FP32, almost as accurate)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set up cudnn for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Set CUDA stream priorities for important operations
        stream = torch.cuda.Stream(priority=-1)  # High priority stream
        with torch.cuda.stream(stream):
            # Critical forward/backward pass
            pass

# Function to compute loss for mixed precision training
def compute_loss(policy_logits, value_preds, rewards, advantages):
    """
    Compute the combined policy and value loss
    (dummy implementation for the mixed precision example)
    """
    # Import necessary modules
    import torch.nn.functional as F
    
    # Calculate value loss
    value_loss = F.mse_loss(value_preds, rewards)
    
    # Calculate policy loss
    policy_loss = -torch.mean(advantages * policy_logits)
    
    # Combined loss
    loss = value_loss + policy_loss
    
    return loss

# Function to update agent for memory efficient training
def update_agent(agent, optimizer, batch):
    """
    Update agent parameters based on batch of experiences
    (dummy implementation for the memory efficient example)
    """
    # Process batch
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    
    # Forward pass
    logits, values = agent(states)
    
    # Compute loss and update parameters
    # ...
    
    return
