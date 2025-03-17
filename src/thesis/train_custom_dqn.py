#!/usr/bin/env python
"""
Training script for Custom DQN agent on 2048 game.
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
from .agents.custom_dqn_agent import CustomDQNAgent
from .config import device, set_seeds

def train_custom_dqn(args):
    """
    Train a Custom DQN agent on the 2048 game.
    
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
    agent = CustomDQNAgent(
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
    
    for episode in tqdm(range(1, args.episodes + 1)):
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
        
        # Log progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_max_tile = np.mean(episode_max_tiles[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            avg_loss = np.mean(losses[-args.log_interval:])
            
            logging.info(f"Episode {episode}/{args.episodes} | "
                         f"Avg Reward: {avg_reward:.1f} | "
                         f"Avg Max Tile: {avg_max_tile:.1f} | "
                         f"Avg Length: {avg_length:.1f} | "
                         f"Avg Loss: {avg_loss:.6f} | "
                         f"Epsilon: {agent.epsilon:.4f}")
        
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
            current_max_tile = np.max(next_state)
            max_tile_seen = max(max_tile_seen, current_max_tile)
            
            # Render if requested
            if render:
                env.render()
        
        # Record game metrics
        max_tiles.append(max_tile_seen)
        scores.append(total_reward)
        steps.append(step_count)
    
    # Calculate statistics
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    max_tile_reached = np.max(max_tiles)
    
    return {
        'avg_score': avg_score,
        'avg_max_tile': avg_max_tile,
        'max_tile_reached': max_tile_reached,
        'scores': scores,
        'max_tiles': max_tiles,
        'steps': steps
    }

def plot_training_curves(rewards, max_tiles, eval_scores, eval_max_tiles, losses, output_dir):
    """
    Plot training curves.
    
    Args:
        rewards: List of episode rewards
        max_tiles: List of episode max tiles
        eval_scores: List of evaluation scores
        eval_max_tiles: List of evaluation max tiles
        losses: List of training losses
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    # Plot max tiles
    plt.subplot(3, 2, 2)
    plt.plot(max_tiles)
    plt.xlabel('Episode')
    plt.ylabel('Max Tile')
    plt.title('Episode Max Tiles')
    
    # Plot evaluation metrics
    plt.subplot(3, 2, 3)
    eval_episodes = np.arange(len(eval_scores)) + 1
    plt.plot(eval_episodes, eval_scores, label='Eval Score')
    plt.xlabel('Evaluation')
    plt.ylabel('Score')
    plt.title('Evaluation Scores')
    plt.legend()
    
    # Plot evaluation max tiles
    plt.subplot(3, 2, 4)
    plt.plot(eval_episodes, eval_max_tiles, label='Eval Max Tile')
    plt.xlabel('Evaluation')
    plt.ylabel('Max Tile')
    plt.title('Evaluation Max Tiles')
    plt.legend()
    
    # Plot losses
    plt.subplot(3, 2, 5)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    # Plot max tile distribution
    plt.subplot(3, 2, 6)
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    plt.bar([str(v) for v in unique_tiles], counts)
    plt.xlabel('Tile Value')
    plt.ylabel('Count')
    plt.title('Max Tile Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train Custom DQN agent for 2048 game")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=0.9, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.9999, help="Exploration decay rate")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=100, help="Episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=500, help="Episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", type=str, default="custom_dqn_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_custom_dqn(args)

if __name__ == "__main__":
    main() 