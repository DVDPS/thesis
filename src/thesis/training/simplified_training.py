import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import random
from collections import deque
from config import device
from game2048 import preprocess_state_onehot, Game2048

def collect_experiences(agent, env, num_episodes=5):
    """
    Collect gameplay experiences for training.
    Simple and reliable approach without complicated optimizations.
    """
    experiences = []
    episode_rewards = []
    max_tiles = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        max_tile = 0
        
        while not done:
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:  # No valid moves means game over
                break
                
            # Process state
            state_proc = preprocess_state_onehot(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get action from policy with epsilon-greedy exploration
            with torch.no_grad():
                # Create action mask
                action_mask = torch.full((1, 4), float('-inf'), device=device)
                action_mask[0, valid_moves] = 0
                
                # Get action probabilities and value
                logits, value = agent(state_tensor)
                logits = logits + action_mask
                
                # Handle potential NaN values
                if torch.isnan(logits).any():
                    logits = action_mask.clone()
                    logits[action_mask == 0] = 1.0
                
                # Sample action - epsilon greedy approach
                if random.random() < agent.exploration_noise:
                    # Random valid action
                    action = random.choice(valid_moves)
                else:
                    # Action from policy
                    probs = F.softmax(logits, dim=1)
                    action = torch.multinomial(probs, 1).item()
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            next_state_proc = preprocess_state_onehot(next_state)
            
            # Store experience
            experiences.append((
                state_proc,
                action,
                reward,
                next_state_proc,
                done,
                value.item()
            ))
            
            # Update state and tracking variables
            state = next_state
            episode_reward += reward
            max_tile = max(max_tile, info.get('max_tile', 0))
        
        episode_rewards.append(episode_reward)
        max_tiles.append(max_tile)
    
    return experiences, np.mean(episode_rewards), max(max_tiles)

def train_agent(agent, env, optimizer, num_epochs=1000, batch_size=64, gamma=0.99,
                save_dir="checkpoints", log_interval=20):
    """
    Simplified training process that focuses on stability and reliability.
    Uses a basic DQN-style approach with experience replay.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create experience buffer
    buffer = deque(maxlen=10000)
    
    # Training metrics
    running_reward = 0
    best_reward = float('-inf')
    highest_tile = 0
    
    logging.info("Starting training with simplified approach")
    logging.info(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    for epoch in range(num_epochs):
        # Update exploration noise
        progress = epoch / num_epochs
        agent.update_exploration(progress)
        
        # Collect new experiences
        agent.eval()  # Disable dropout for experience collection
        experiences, avg_reward, max_tile = collect_experiences(agent, env, num_episodes=3)
        agent.train()  # Re-enable dropout for training
        
        # Add experiences to buffer
        buffer.extend(experiences)
        
        # Skip training if not enough experiences
        if len(buffer) < batch_size:
            logging.info(f"Epoch {epoch+1}: Not enough experiences yet, continuing collection")
            continue
        
        # Update running reward with simple moving average
        running_reward = 0.05 * avg_reward + (1 - 0.05) * (running_reward if epoch > 0 else avg_reward)
        
        # Track highest tile achieved
        highest_tile = max(highest_tile, max_tile)
        
        # Training loop - simple and stable approach
        for _ in range(4):  # Multiple updates per collection
            # Sample mini-batch
            batch = random.sample(buffer, batch_size)
            states, actions, rewards, next_states, dones, _ = zip(*batch)
            
            # Convert to tensors
            states_tensor = torch.tensor(np.array(states), dtype=torch.float, device=device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
            dones_tensor = torch.tensor(dones, dtype=torch.float, device=device)
            
            # Get current Q values
            q_logits, state_values = agent(states_tensor, training=True)
            action_qs = q_logits.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Get next state values
            with torch.no_grad():
                _, next_state_values = agent(next_states_tensor)
                next_state_values = next_state_values.squeeze(1)
                
            # Compute targets (simple value-based approach)
            targets = rewards_tensor + gamma * next_state_values * (1 - dones_tensor)
            
            # Clamp targets to reasonable values for stability
            targets = torch.clamp(targets, min=-100, max=1000)
            
            # Compute value loss (MSE)
            value_loss = F.mse_loss(action_qs, targets)
            
            # Compute policy loss (simple policy gradient)
            probs = F.softmax(q_logits, dim=1)
            log_probs = F.log_softmax(q_logits, dim=1)
            policy_loss = -(log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) * 
                           (targets.detach() - state_values.squeeze(1)))
            policy_loss = policy_loss.mean()
            
            # Add entropy loss for exploration
            entropy = -(probs * log_probs).sum(dim=1).mean()
            entropy_bonus = 0.01 * entropy
            
            # Combined loss
            loss = value_loss + policy_loss - entropy_bonus
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in agent.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    has_nan_grad = True
            
            if has_nan_grad:
                logging.warning("NaN gradients detected and zeroed")
            
            optimizer.step()
        
        # Log progress
        if (epoch + 1) % log_interval == 0:
            logging.info(f"Epoch {epoch+1}/{num_epochs} | " 
                       f"Reward: {avg_reward:.1f} | "
                       f"Running reward: {running_reward:.1f} | "
                       f"Max tile: {max_tile} | "
                       f"Exploration: {agent.exploration_noise:.2f}")
        
        # Save best model
        if running_reward > best_reward:
            best_reward = running_reward
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_reward': running_reward,
                'max_tile': max_tile,
            }, os.path.join(save_dir, "best_model.pt"))
            logging.info(f"New best model saved with reward {best_reward:.1f}")
        
        # Save milestone models for high tiles
        if max_tile > 0 and max_tile in [256, 512, 1024, 2048] and max_tile > highest_tile:
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_reward': running_reward,
                'max_tile': max_tile,
            }, os.path.join(save_dir, f"model_tile_{max_tile}.pt"))
            logging.info(f"New milestone model saved for tile {max_tile}")
    
    # Final logging
    logging.info("Training complete!")
    logging.info(f"Best reward achieved: {best_reward:.1f}")
    logging.info(f"Highest tile achieved: {highest_tile}")
    
    return {
        'best_reward': best_reward,
        'highest_tile': highest_tile
    }