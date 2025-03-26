#!/usr/bin/env python
"""
Optimized training script for Custom DQN agent on 2048 game.
Includes mixed precision training and H100 GPU optimizations.
"""

import torch
import numpy as np
import logging
import os
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.thesis.environment.game2048 import Game2048, preprocess_state_onehot
from src.thesis.agents.custom_dqn_agent import CustomDQNAgent, DQN
from src.thesis.config import device, set_seeds

# Enable mixed precision training if CUDA is available
USE_MIXED_PRECISION = False
if torch.cuda.is_available():
    try:
        from torch.cuda.amp import autocast, GradScaler
        USE_MIXED_PRECISION = True
        print("Mixed precision training enabled")
    except ImportError:
        print("Mixed precision training not available")

# Enable JIT compilation for faster forward passes
def jit_compile_model(model):
    """JIT compile a model for faster inference if possible"""
    try:
        # Try to JIT compile the model
        example_input = torch.zeros((1, 16, 4, 4), device=device)
        return torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return model

class OptimizedDQNAgent(CustomDQNAgent):
    """
    Optimized version of the CustomDQNAgent with mixed precision training
    and other performance enhancements for H100 GPUs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Use gradient scaler for mixed precision training
        self.scaler = GradScaler() if USE_MIXED_PRECISION else None
        
        # Try to JIT compile the policy and target networks
        if torch.cuda.is_available():
            try:
                self.policy_net = jit_compile_model(self.policy_net)
                print("JIT compilation successful for policy network")
            except:
                print("JIT compilation failed for policy network")
        
        # Use half precision if mixed precision is enabled
        if USE_MIXED_PRECISION:
            # Keep optimizer in full precision
            for param in self.policy_net.parameters():
                # Ensure params are in FP32 for optimizer
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)
    
    def update(self):
        """Update the policy network using a batch of experiences with mixed precision"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float, device=device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float, device=device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float, device=device)
        done_batch = torch.tensor(batch[4], dtype=torch.float, device=device).unsqueeze(1)
        
        # Mixed precision training
        if USE_MIXED_PRECISION:
            with autocast():
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                
                # Compute V(s_{t+1}) for all next states
                with torch.no_grad():
                    next_state_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
                    
                # Compute the expected Q values
                expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
                
                # Compute loss
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # Optimize with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # Clip gradients to stabilize training
            self.scaler.unscale_(self.optimizer)
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard full precision training
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # Compute V(s_{t+1}) for all next states
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
                
            # Compute the expected Q values
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
            
            # Compute loss
            loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to stabilize training
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        
        return loss.item()

def train_custom_dqn_optimized(args):
    """
    Train a Custom DQN agent on the 2048 game with optimizations for H100 GPUs.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set random seeds for reproducibility
    set_seeds(args.seed)
    
    # Log device and arguments
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {args}")
    logging.info(f"Mixed precision training: {USE_MIXED_PRECISION}")
    
    # Pin memory for faster data transfer (if CUDA is available)
    pin_memory = torch.cuda.is_available()
    if pin_memory:
        torch.cuda.empty_cache()
        logging.info("CUDA memory pinning enabled")
    
    # Create environment
    env = Game2048()
    
    # Create optimized agent
    agent = OptimizedDQNAgent(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Training metrics
    episode_rewards = []
    episode_max_tiles = []
    episode_lengths = []
    evaluation_scores = []
    evaluation_max_tiles = []
    losses = []
    
    # Main training loop
    logging.info(f"Starting training for {args.episodes} episodes")
    
    total_steps = 0
    start_time = time.time()
    speed_benchmark_time = time.time()
    speed_benchmark_episodes = 0
    episode_times = []
    
    # Pre-allocate tensors to reduce memory allocation overhead
    if torch.cuda.is_available():
        # Pre-warm CUDA for more consistent timing
        dummy_tensor = torch.zeros(1, device=device)
        del dummy_tensor
        torch.cuda.empty_cache()
    
    for episode in tqdm(range(1, args.episodes + 1)):
        episode_start_time = time.time()
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_max_tile = 0
        episode_loss = 0
        done = False
        
        while not done and episode_length < args.max_steps:
            # Process state
            state_proc = preprocess_state_onehot(state)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
            
            # Select action
            action = agent.select_action(state_proc, valid_moves)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            episode_max_tile = max(episode_max_tile, np.max(next_state))
            
            # Process next state
            next_state_proc = preprocess_state_onehot(next_state)
            
            # Store transition
            agent.store_transition(state_proc, action, reward, next_state_proc, done)
            
            # Update state
            state = next_state
            
            # Update network
            if len(agent.memory) >= args.batch_size:
                loss = agent.update()
                episode_loss += loss
                
            total_steps += 1
        
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_max_tiles.append(episode_max_tile)
        episode_lengths.append(episode_length)
        
        # Calculate average loss
        avg_loss = episode_loss / episode_length if episode_length > 0 else 0
        losses.append(avg_loss)
        
        # Track episode time
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Log progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_max_tile = np.mean(episode_max_tiles[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            avg_loss = np.mean(losses[-args.log_interval:])
            avg_episode_time = np.mean(episode_times[-args.log_interval:])
            
            logging.info(f"Episode {episode}/{args.episodes} | "
                         f"Avg Reward: {avg_reward:.1f} | "
                         f"Avg Max Tile: {avg_max_tile:.1f} | "
                         f"Avg Length: {avg_length:.1f} | "
                         f"Avg Loss: {avg_loss:.6f} | "
                         f"Avg Episode Time: {avg_episode_time:.3f}s | "
                         f"Epsilon: {agent.epsilon:.4f}")
            
            # Calculate and log training speed
            speed_benchmark_episodes += args.log_interval
            current_time = time.time()
            elapsed = current_time - speed_benchmark_time
            if elapsed > 0:
                episodes_per_second = speed_benchmark_episodes / elapsed
                steps_per_second = total_steps / (current_time - start_time)
                estimated_total_time = (args.episodes - episode) / episodes_per_second / 60
                logging.info(f"Training Speed: {episodes_per_second:.2f} episodes/s, "
                             f"{steps_per_second:.2f} steps/s, "
                             f"Est. remaining time: {estimated_total_time:.1f} minutes")
                
                # Reset benchmark
                speed_benchmark_episodes = 0
                speed_benchmark_time = current_time
        
        # Evaluate agent
        if episode % args.eval_interval == 0:
            eval_results = evaluate_agent(agent, num_games=args.eval_episodes)
            avg_eval_score = eval_results['avg_score']
            avg_eval_max_tile = eval_results['avg_max_tile']
            
            evaluation_scores.append(avg_eval_score)
            evaluation_max_tiles.append(avg_eval_max_tile)
            
            logging.info(f"Evaluation | "
                         f"Avg Score: {avg_eval_score:.1f} | "
                         f"Avg Max Tile: {avg_eval_max_tile:.1f} | "
                         f"Best Max Tile: {eval_results['max_tile_reached']}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_eval_{len(evaluation_scores)}.pt")
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model if this is the best performance
            if len(evaluation_max_tiles) == 1 or avg_eval_max_tile > max(evaluation_max_tiles[:-1]):
                best_path = os.path.join(args.output_dir, "best_model.pt")
                agent.save(best_path)
                logging.info(f"Saved best model to {best_path}")
    
    # Calculate training statistics
    total_time = time.time() - start_time
    steps_per_sec = total_steps / total_time
    
    logging.info(f"Training completed in {total_time:.2f} seconds")
    logging.info(f"Average steps per second: {steps_per_sec:.1f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    agent.save(final_path)
    logging.info(f"Saved final model to {final_path}")
    
    # Plot training curves
    plot_training_curves(
        episode_rewards, episode_max_tiles, 
        evaluation_scores, evaluation_max_tiles,
        losses, args.output_dir
    )
    
    return agent

def evaluate_agent(agent, env=None, num_games=10, render=False, max_steps=1000):
    """
    Evaluate the agent's performance.
    
    Args:
        agent: Agent to evaluate
        env: Environment to use (creates a new one if None)
        num_games: Number of games to play
        render: Whether to render the games
        max_steps: Maximum steps per game
        
    Returns:
        Dictionary with evaluation results
    """
    if env is None:
        env = Game2048()
        
    max_tiles = []
    scores = []
    steps = []
    
    for game_idx in range(num_games):
        state = env.reset()
        done = False
        step_count = 0
        max_tile_seen = 0
        total_reward = 0
        
        while not done and step_count < max_steps:
            # Process state
            state_proc = preprocess_state_onehot(state)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Select action (deterministic for evaluation)
            with torch.no_grad():
                state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
                q_values = agent.policy_net(state_tensor)
                
                # Mask invalid actions
                mask = torch.ones(4, device=device) * float('-inf')
                mask[valid_moves] = 0
                
                masked_q_values = q_values + mask
                action = torch.argmax(masked_q_values).item()
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Update state and metrics
            state = next_state
            step_count += 1
            max_tile_seen = max(max_tile_seen, np.max(next_state))
        
        # Record results
        max_tiles.append(max_tile_seen)
        scores.append(total_reward)
        steps.append(step_count)
    
    # Calculate statistics
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    max_tile_reached = np.max(max_tiles)
    avg_steps = np.mean(steps)
    
    return {
        'avg_score': avg_score,
        'avg_max_tile': avg_max_tile,
        'max_tile_reached': max_tile_reached,
        'avg_steps': avg_steps
    }

def plot_training_curves(rewards, max_tiles, eval_scores, eval_max_tiles, losses, output_dir):
    """
    Plot training curves.
    
    Args:
        rewards: List of episode rewards
        max_tiles: List of maximum tiles achieved in each episode
        eval_scores: List of evaluation scores
        eval_max_tiles: List of evaluation maximum tiles
        losses: List of losses
        output_dir: Output directory
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    # Plot max tiles
    plt.subplot(2, 2, 2)
    plt.plot(max_tiles)
    plt.xlabel('Episode')
    plt.ylabel('Max Tile')
    plt.title('Max Tiles per Episode')
    
    # Plot evaluation scores
    plt.subplot(2, 2, 3)
    x = np.arange(len(eval_scores)) + 1
    plt.plot(x, eval_scores)
    plt.xlabel('Evaluation')
    plt.ylabel('Average Score')
    plt.title('Evaluation Scores')
    
    # Plot evaluation max tiles
    plt.subplot(2, 2, 4)
    plt.plot(x, eval_max_tiles)
    plt.xlabel('Evaluation')
    plt.ylabel('Average Max Tile')
    plt.title('Evaluation Max Tiles')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Plot losses separately
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    plt.close('all')  # Close figures to save memory

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train Custom DQN agent for 2048 game with H100 optimizations")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Exploration decay rate")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    
    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=5, help="Episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=100, help="Episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", type=str, default="h100_dqn_optimized_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Train agent
    train_custom_dqn_optimized(args)

if __name__ == "__main__":
    main() 