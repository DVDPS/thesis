#!/usr/bin/env python
"""
Training script for DQN agent on 2048 game.
"""

import torch
import numpy as np
import logging
import os
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from .environment.game2048 import Game2048, preprocess_state_onehot
from .agents.dqn_agent import DQNAgent
from .config import device, set_seeds

def train_dqn(args):
    """
    Train a DQN agent on the 2048 game.
    
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
    
    # Create environment
    env = Game2048()
    
    # Create agent
    agent = DQNAgent(
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        update_freq=args.update_freq,
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
    losses = []
    evaluation_scores = []
    evaluation_max_tiles = []
    
    # Fill replay buffer with random experiences before training
    if len(agent.replay_buffer) < args.min_buffer_size and not args.checkpoint:
        logging.info(f"Filling replay buffer with {args.min_buffer_size} experiences...")
        state = env.reset()
        for _ in tqdm(range(args.min_buffer_size)):
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                state = env.reset()
                valid_moves = env.get_possible_moves()
                
            action = np.random.choice(valid_moves)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            state_proc = preprocess_state_onehot(state)
            next_state_proc = preprocess_state_onehot(next_state)
            next_valid_moves = env.get_possible_moves() if not done else []
            
            agent.store_transition(state_proc, action, reward, next_state_proc, done, next_valid_moves)
            
            state = next_state if not done else env.reset()
    
    # Main training loop
    logging.info(f"Starting training for {args.episodes} episodes...")
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_max_tile = 0
        episode_loss = 0
        episode_steps = 0
        
        # Play one episode
        done = False
        while not done:
            # Process state
            state_proc = preprocess_state_onehot(state)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Select action
            action = agent.get_action(state_proc, valid_moves)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_max_tile = max(episode_max_tile, np.max(next_state))
            episode_steps += 1
            
            # Process next state
            next_state_proc = preprocess_state_onehot(next_state)
            
            # Get valid moves for next state
            next_valid_moves = env.get_possible_moves() if not done else []
            
            # Store transition
            agent.store_transition(state_proc, action, reward, next_state_proc, done, next_valid_moves)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
                
            # Update state
            state = next_state
            
            # Break if episode is too long
            if episode_steps >= args.max_steps:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_max_tiles.append(episode_max_tile)
        episode_lengths.append(episode_steps)
        losses.append(episode_loss / max(1, episode_steps))
        
        # Print max tile every 100 episodes
        if episode % 100 == 0:
            recent_max_tiles = episode_max_tiles[-100:]  # Get last 100 episodes
            avg_max_tile = np.mean(recent_max_tiles)
            best_max_tile = np.max(recent_max_tiles)
            print(f"\nEpisode {episode} Max Tile Stats:")
            print(f"  Average Max Tile (last 100): {avg_max_tile:.1f}")
            print(f"  Best Max Tile (last 100): {best_max_tile}")
            print(f"  Best Max Tile (all time): {max(episode_max_tiles)}")
            
            # Print tile distribution for last 100 episodes
            tile_counts = {}
            for tile in recent_max_tiles:
                tile_counts[tile] = tile_counts.get(tile, 0) + 1
            print("\nTile Distribution (last 100 episodes):")
            for tile in sorted(tile_counts.keys()):
                count = tile_counts[tile]
                print(f"  {tile}: {count} times ({count}%)")
            print()
        
        # Log progress at regular intervals
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_max_tile = np.mean(episode_max_tiles[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            avg_loss = np.mean(losses[-args.log_interval:])
            
            logging.info(f"Episode {episode}/{args.episodes} | "
                         f"Avg Reward: {avg_reward:.1f} | "
                         f"Avg Max Tile: {avg_max_tile:.1f} | "
                         f"Avg Length: {avg_length:.1f} | "
                         f"Avg Loss: {avg_loss:.4f} | "
                         f"Epsilon: {agent.epsilon:.4f}")
        
        # Evaluate agent
        if episode % args.eval_interval == 0:
            eval_scores, eval_max_tiles = evaluate_agent(agent, args.eval_episodes)
            avg_eval_score = np.mean(eval_scores)
            avg_eval_max_tile = np.mean(eval_max_tiles)
            max_eval_tile = np.max(eval_max_tiles)
            
            evaluation_scores.append(avg_eval_score)
            evaluation_max_tiles.append(avg_eval_max_tile)
            
            logging.info(f"Evaluation | "
                         f"Avg Score: {avg_eval_score:.1f} | "
                         f"Avg Max Tile: {avg_eval_max_tile:.1f} | "
                         f"Best Max Tile: {max_eval_tile}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_episode_{episode}.pt")
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model if this is the best performance
            if len(evaluation_max_tiles) == 1 or avg_eval_max_tile > max(evaluation_max_tiles[:-1]):
                best_path = os.path.join(args.output_dir, "best_model.pt")
                agent.save(best_path)
                logging.info(f"Saved best model to {best_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    agent.save(final_path)
    logging.info(f"Saved final model to {final_path}")
    
    # Plot training curves
    plot_training_curves(
        episode_rewards, episode_max_tiles, losses, 
        evaluation_scores, evaluation_max_tiles,
        args.output_dir
    )
    
    return agent

def evaluate_agent(agent, num_episodes=10, max_steps=1000):
    """
    Evaluate the agent without exploration.
    
    Args:
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (scores, max_tiles)
    """
    env = Game2048()
    scores = []
    max_tiles = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_score = 0
        episode_max_tile = 0
        
        for _ in range(max_steps):
            # Process state
            state_proc = preprocess_state_onehot(state)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Select action (no exploration)
            action = agent.get_action(state_proc, valid_moves, epsilon=0.0)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            episode_max_tile = max(episode_max_tile, np.max(next_state))
            
            # Update state
            state = next_state
            
            if done:
                break
        
        scores.append(episode_score)
        max_tiles.append(episode_max_tile)
    
    return scores, max_tiles

def plot_training_curves(rewards, max_tiles, losses, eval_scores, eval_max_tiles, output_dir):
    """
    Plot training curves.
    
    Args:
        rewards: List of episode rewards
        max_tiles: List of episode max tiles
        losses: List of episode losses
        eval_scores: List of evaluation scores
        eval_max_tiles: List of evaluation max tiles
        output_dir: Directory to save plots
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
    plt.title('Episode Max Tiles')
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Episode Losses')
    
    # Plot evaluation metrics
    plt.subplot(2, 2, 4)
    eval_episodes = np.arange(len(eval_scores)) * 100  # Assuming eval_interval=100
    plt.plot(eval_episodes, eval_scores, label='Eval Score')
    plt.plot(eval_episodes, eval_max_tiles, label='Eval Max Tile')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Plot tile distribution
    plt.figure(figsize=(10, 6))
    tile_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    tile_counts = [sum(1 for t in max_tiles if t == v) for v in tile_values]
    plt.bar([str(v) for v in tile_values], tile_counts)
    plt.xlabel('Tile Value')
    plt.ylabel('Count')
    plt.title('Max Tile Distribution')
    plt.savefig(os.path.join(output_dir, 'tile_distribution.png'))
    plt.close()

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train DQN agent for 2048 game")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for updates")
    parser.add_argument("--min-buffer-size", type=int, default=10000, help="Minimum buffer size before training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Exploration decay rate")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--update-freq", type=int, default=4, help="How often to update the network (in steps)")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    
    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=250, help="Episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", type=str, default="dqn_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_dqn(args)

if __name__ == "__main__":
    main() 