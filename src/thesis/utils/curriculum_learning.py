import numpy as np
import torch
import random
import os
import logging
from torch.utils.data import Dataset, DataLoader
from ..environment.game2048 import Game2048, preprocess_state_onehot
from ..agents.base_agent import PPOAgent
from ..training.training import compute_advantages_vectorized
from .visualizations import generate_board_with_high_tile, augment_board

# Add safe globals for model loading
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])

class HighTileDataset(Dataset):
    """
    Dataset for curriculum learning that focuses on board states with high-value tiles
    like 256 and 512, helping the agent learn how to progress beyond these tiles.
    """
    def __init__(self, target_tiles=[256, 512], num_samples=1000, augment=True):
        """
        Args:
            target_tiles: List of tile values to focus on
            num_samples: Number of samples to generate
            augment: Whether to use data augmentation (rotations/reflections)
        """
        self.target_tiles = target_tiles
        self.num_samples = num_samples
        self.augment = augment
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        """Generate board samples with high-value tiles."""
        samples = []
        for _ in range(self.num_samples):
            tile = np.random.choice(self.target_tiles)
            board = generate_board_with_high_tile(tile)
            
            if self.augment:
                board = augment_board(board)
                
            # Convert to one-hot representation
            state = preprocess_state_onehot(board)
            samples.append(state)
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def curriculum_fine_tune(agent, optimizer, 
                        target_tiles=[256, 512], 
                        num_samples=1000,
                        batch_size=32,
                        fine_tune_epochs=100,
                        entropy_coef=0.2,
                        learning_rate=0.0005,
                        clip_param=0.2,
                        checkpoint_dir="checkpoints"):
    """
    Fine-tune the agent specifically on situations with high-value tiles
    to help it learn how to progress beyond these tiles.
    
    Args:
        agent: The PPO agent to fine-tune
        optimizer: The optimizer to use
        target_tiles: The tile values to focus on
        num_samples: Number of samples to generate
        batch_size: Batch size for training
        fine_tune_epochs: Number of epochs to fine-tune
        entropy_coef: Entropy coefficient to encourage exploration
        learning_rate: Learning rate for the optimizer
        clip_param: PPO clipping parameter
        checkpoint_dir: Directory to save checkpoints
    """
    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Create dataset and dataloader
    dataset = HighTileDataset(target_tiles=target_tiles, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create environment for evaluation
    env = Game2048()
    device = next(agent.parameters()).device
    
    # Set agent to training mode
    agent.train()
    
    # Temporarily set exploration noise higher
    original_exploration = agent.exploration_noise
    original_min_exploration = agent.min_exploration_noise
    agent.exploration_noise = 1.5  # Higher exploration noise
    agent.min_exploration_noise = 0.2  # Higher minimum exploration
    
    print(f"Starting curriculum fine-tuning on tiles: {target_tiles}")
    print(f"Using higher exploration noise: {agent.exploration_noise}")
    
    # Track statistics
    best_max_tile = 0
    best_score = 0
    
    # Training loop
    for epoch in range(fine_tune_epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Training phase
        for batch_states in dataloader:
            batch_states = batch_states.float().to(device)
            
            # Get actions from the current policy
            with torch.no_grad():
                policy_logits, values = agent(batch_states, training=False)
                dist = torch.distributions.Categorical(logits=policy_logits)
                actions = dist.sample()
                old_log_probs = dist.log_prob(actions)
            
            # Simulate taking these actions to get rewards and next states
            rewards = []
            next_states = []
            for i in range(batch_states.shape[0]):
                state = batch_states[i].cpu().numpy()
                action = actions[i].item()
                
                # Convert from one-hot back to regular board
                regular_board = np.zeros((4, 4), dtype=np.int32)
                for c in range(state.shape[0]):
                    if c == 0:  # Skip empty channel
                        continue
                    regular_board += (state[c] * (2 ** c)).astype(np.int32)
                
                # Set up environment and take action
                env.board = regular_board.copy()
                next_state, reward, done, _ = env.step(action)
                
                # Get one-hot representation of next state
                next_state_onehot = preprocess_state_onehot(next_state)
                
                rewards.append(reward)
                next_states.append(next_state_onehot)
            
            # Convert to tensors
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            
            # Get value estimates for the next states
            next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
            with torch.no_grad():
                _, next_values = agent(next_states)
                next_values = next_values.squeeze()
            
            # Compute rewards plus value estimates for bootstrapping
            returns = rewards + 0.99 * next_values
            
            # Compute advantages
            advantages = returns - values.squeeze()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(2):  # Fewer PPO epochs for curriculum learning
                # Get current policy and value estimates
                new_policy_logits, new_values = agent(batch_states, training=True)
                new_dist = torch.distributions.Categorical(logits=new_policy_logits)
                new_log_probs = new_dist.log_prob(actions)
                entropy = new_dist.entropy().mean()
                
                # Compute policy loss with PPO clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = torch.nn.functional.mse_loss(new_values.squeeze(), returns)
                
                # Combined loss with entropy bonus
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                
                # Update the model
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Evaluation phase (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            max_tiles = []
            total_scores = []
            
            # Run several evaluation episodes
            for _ in range(5):
                state = env.reset()
                done = False
                game_score = 0
                
                while not done:
                    state_proc = preprocess_state_onehot(state)
                    state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits, _ = agent(state_tensor)
                        action = torch.argmax(logits).item()
                    
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    game_score += info.get('merge_score', 0)
                
                max_tiles.append(int(np.max(state)))
                total_scores.append(game_score)
            
            # Print statistics
            avg_max_tile = np.mean(max_tiles)
            max_max_tile = np.max(max_tiles)
            avg_score = np.mean(total_scores)
            max_score = np.max(total_scores)
            
            # Track best performance
            if max_max_tile > best_max_tile:
                best_max_tile = max_max_tile
            if max_score > best_score:
                best_score = max_score
                
            # Log progress
            print(f"Epoch {epoch+1}/{fine_tune_epochs}, " 
                  f"Avg Tile: {avg_max_tile}, Max Tile: {max_max_tile}, "
                  f"Avg Score: {avg_score:.1f}, Best Score: {best_score:.1f}")
            print(f"Losses - Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f}, Entropy: {total_entropy:.4f}")
    
    # Restore original exploration noise
    agent.exploration_noise = original_exploration
    agent.min_exploration_noise = original_min_exploration
    
    print(f"Curriculum fine-tuning complete!")
    print(f"Best Max Tile: {best_max_tile}, Best Score: {best_score:.1f}")
    
    # Save specialized model
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"model_curriculum_{target_tiles[-1]}.pt")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_max_tile': best_max_tile,
        'best_score': best_score,
    }, save_path)
    
    print(f"Saved curriculum-tuned model to {save_path}")
    return best_max_tile, best_score 

# Add curriculum_learning function at the end of the file
def curriculum_learning(agent, optimizer, epochs=500, target_tiles=[256, 512, 1024], output_dir="checkpoints/curriculum"):
    """Alias for curriculum_fine_tune for backward compatibility"""
    return curriculum_fine_tune(
        agent=agent, 
        optimizer=optimizer, 
        target_tiles=target_tiles,
        fine_tune_epochs=epochs,
        checkpoint_dir=output_dir
    ) 