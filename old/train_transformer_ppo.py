#!/usr/bin/env python
"""
Training script for Transformer-based PPO agent on 2048 game.
Uses self-attention mechanisms to better capture spatial patterns and long-term dependencies.
Optimized for H100 GPUs with gradient accumulation and mixed precision.
"""

import torch
import numpy as np
import logging
import os
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .environment.game2048 import Game2048, preprocess_state_onehot
from .agents.transformer_ppo_agent import TransformerPPOAgent
from .config import device, set_seeds
from .utils.evaluation.evaluation import evaluate_agent
import torch.nn.functional as F
import torch.optim as optim



def debug_policy_distribution(policy_logits, action_mask=None):
    """
    Debug the policy distribution to identify potential issues.
    
    Args:
        policy_logits: Policy logits from the agent
        action_mask: Optional mask of valid actions
        
    Returns:
        Dict with debug info
    """
    with torch.no_grad():
        # Convert to probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        # If masked, apply the mask
        if action_mask is not None:
            # Create mask tensor (1 for valid actions, 0 for invalid)
            mask_tensor = torch.zeros_like(policy)
            mask_tensor[:, action_mask] = 1.0
            
            # Apply mask
            masked_policy = policy * mask_tensor
            # Renormalize
            masked_policy = masked_policy / (masked_policy.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Check if masked policy contains NaNs
            if torch.isnan(masked_policy).any():
                return {"error": "NaN in masked policy", "policy": policy.cpu().numpy()}
            
            policy = masked_policy
        
        # Compute entropy
        log_policy = torch.log(policy + 1e-8)
        entropy = -(policy * log_policy).sum(dim=-1).mean().item()
        
        # Get stats on the distribution
        max_prob = policy.max(dim=-1)[0].mean().item()
        min_prob = policy.min(dim=-1)[0].mean().item()
        std_prob = policy.std(dim=-1).mean().item()
        
        return {
            "entropy": entropy,
            "max_prob": max_prob,
            "min_prob": min_prob,
            "std": std_prob,
            "policy": policy.cpu().numpy()
        }

def curriculum_setup(episode_count, max_episodes=1000):
    """
    Set up curriculum learning parameters based on training progress.
    Implements a three-phase curriculum with smooth transitions.
    
    Args:
        episode_count: Current episode count
        max_episodes: Maximum number of episodes for scaling
        
    Returns:
        Dictionary of curriculum parameters
    """
    progress = min(1.0, episode_count / max_episodes)
    
    # Early training (0-20%): Focus on exploration and basic patterns
    if progress < 0.2:
        phase = "early"
        # High exploration, high learning rate, large policy updates
        entropy_coef = 0.02
        learning_rate = 0.0001
        target_kl = 0.03
        # Smooth transition to mid phase
        transition = progress / 0.2
        entropy_coef = entropy_coef * (1 - transition) + 0.01 * transition
        learning_rate = learning_rate * (1 - transition) + 0.00005 * transition
        target_kl = target_kl * (1 - transition) + 0.02 * transition
    
    # Mid training (20-60%): Balance exploration and exploitation
    elif progress < 0.6:
        phase = "mid"
        # Moderate exploration, moderate learning rate, balanced updates
        entropy_coef = 0.01
        learning_rate = 0.00005
        target_kl = 0.02
        # Smooth transition to late phase
        transition = (progress - 0.2) / 0.4
        entropy_coef = entropy_coef * (1 - transition) + 0.005 * transition
        learning_rate = learning_rate * (1 - transition) + 0.00002 * transition
        target_kl = target_kl * (1 - transition) + 0.01 * transition
    
    # Late training (60-100%): Focus on exploitation and fine-tuning
    else:
        phase = "late"
        # Low exploration, low learning rate, small policy updates
        entropy_coef = 0.005
        learning_rate = 0.00002
        target_kl = 0.01
    
    return {
        "phase": phase,
        "entropy_coef": entropy_coef,
        "learning_rate": learning_rate,
        "target_kl": target_kl
    }

def train_transformer_ppo(
    output_dir: str,
    total_timesteps: int = 1_000_000,
    batch_size: int = 64,
    update_epochs: int = 10,
    timesteps_per_update: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.015,
    learning_rate: float = 3e-4,
    embed_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    mixed_precision: bool = False,
    use_data_parallel: bool = False,
    seed: int = 42
):
    """
    Train a Transformer-based PPO agent on the 2048 game with curriculum learning.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    set_seeds(seed)
    
    # Create environment
    env = Game2048()
    
    # Create agent with increased capacity
    agent = TransformerPPOAgent(
        board_size=4,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        mixed_precision=mixed_precision
    )
    
    # Handle data parallel if requested
    if use_data_parallel and torch.cuda.device_count() > 1:
        agent.network = torch.nn.DataParallel(agent.network)
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    episode_max_tiles = []
    total_steps = 0
    episode_count = 0
    
    # Create TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Training loop
    while total_steps < total_timesteps:
        # Get curriculum parameters based on training progress
        curriculum_params = curriculum_setup(episode_count, total_timesteps // timesteps_per_update)
        
        # Update agent parameters based on curriculum
        agent.ent_coef = curriculum_params['entropy_coef']
        agent.target_kl = curriculum_params['target_kl']
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = curriculum_params['learning_rate']
        
        # Collect experience
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_reward = 0
        episode_length = 0
        episode_max_tile = 0
        done = False
        
        # Reset environment
        state = env.reset()
        
        # Collect timesteps_per_update steps
        while len(states) < timesteps_per_update:
            # Get valid moves from environment
            valid_moves = env.get_possible_moves()
            
            # Get action from policy with temperature-based exploration
            action, _, _ = agent.get_action(state, valid_moves)  # Extract just the action
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_max_tile = max(episode_max_tile, np.max(next_state))
            
            # Update state
            state = next_state
            
            # If episode is done, reset environment
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_max_tiles.append(episode_max_tile)
                episode_count += 1
                
                # Reset metrics for next episode
                episode_reward = 0
                episode_length = 0
                episode_max_tile = 0
                state = env.reset()
        
        # Update agent with curriculum parameters
        agent.states = states
        agent.actions = actions
        agent.rewards = rewards
        agent.next_states = next_states
        agent.dones = dones
        
        if not dones[-1]:
            final_state = next_states[-1]
            final_state_proc = preprocess_state_onehot(final_state)
            # Pass the raw state to get_action, which will handle preprocessing internally
            _,_,final_value = agent.get_action(final_state, env.get_possible_moves())
            agent.update(final_value)
        else:
            agent.update()
        
        # Log metrics
        total_steps += len(states)
        if episode_count % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            avg_max_tile = np.mean(episode_max_tiles[-100:]) if episode_max_tiles else 0
            max_tile_reached = max(episode_max_tiles) if episode_max_tiles else 0
            
            # Log curriculum parameters
            logging.info(f"Episode {episode_count}")
            logging.info(f"Curriculum Phase: {curriculum_params['phase']}")
            logging.info(f"Entropy Coefficient: {curriculum_params['entropy_coef']:.4f}")
            logging.info(f"Learning Rate: {curriculum_params['learning_rate']:.6f}")
            logging.info(f"Target KL: {curriculum_params['target_kl']:.4f}")
            logging.info(f"Average reward: {avg_reward:.2f}")
            logging.info(f"Average length: {avg_length:.2f}")
            logging.info(f"Average max tile: {avg_max_tile:.2f}")
            logging.info(f"Max tile reached: {max_tile_reached}")
            
            # Save metrics to TensorBoard
            writer.add_scalar('train/reward', avg_reward, episode_count)
            writer.add_scalar('train/length', avg_length, episode_count)
            writer.add_scalar('train/max_tile', avg_max_tile, episode_count)
            writer.add_scalar('train/learning_rate', curriculum_params['learning_rate'], episode_count)
            writer.add_scalar('train/entropy_coef', curriculum_params['entropy_coef'], episode_count)
            writer.add_scalar('train/target_kl', curriculum_params['target_kl'], episode_count)
            
            # Save model checkpoint
            if episode_count % 100 == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{episode_count}.pt')
                agent.save(checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    return agent, episode_rewards, episode_lengths, episode_max_tiles

def plot_training_curves(rewards, max_tiles, eval_scores, eval_max_tiles, output_dir):
    """
    Plot training curves.
    
    Args:
        rewards: List of episode rewards
        max_tiles: List of episode max tiles
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
    
    # Plot evaluation metrics
    plt.subplot(2, 2, 3)
    eval_episodes = np.arange(len(eval_scores)) + 1
    plt.plot(eval_episodes, eval_scores, label='Eval Score')
    plt.xlabel('Evaluation')
    plt.ylabel('Score')
    plt.title('Evaluation Scores')
    plt.legend()
    
    # Plot evaluation max tiles
    plt.subplot(2, 2, 4)
    plt.plot(eval_episodes, eval_max_tiles, label='Eval Max Tile')
    plt.xlabel('Evaluation')
    plt.ylabel('Max Tile')
    plt.title('Evaluation Max Tiles')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Plot max tile distribution
    plt.figure(figsize=(10, 6))
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    plt.bar([str(v) for v in unique_tiles], counts)
    plt.xlabel('Tile Value')
    plt.ylabel('Count')
    plt.title('Max Tile Distribution')
    plt.savefig(os.path.join(output_dir, 'tile_distribution.png'))
    plt.close()

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train Transformer-based PPO agent for 2048 game (H100 optimized)")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--timesteps-per-update", type=int, default=1024, help="Timesteps per PPO update")
    parser.add_argument("--max-episode-length", type=int, default=2000, help="Maximum steps per episode")
    
    # Transformer architecture parameters
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension for transformer")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    
    # PPO parameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of PPO update epochs")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for updates")
    
    # H100 optimization parameters
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use-data-parallel", action="store_true", help="Use data parallelism for multiple GPUs")
    
    # Logging and evaluation
    parser.add_argument("--eval-interval", type=int, default=50, help="Episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", type=str, default="transformer_ppo_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_transformer_ppo(args.output_dir, args.total_timesteps, args.batch_size, args.update_epochs, args.timesteps_per_update, args.gamma, args.gae_lambda, args.clip_ratio, args.vf_coef, args.ent_coef, args.max_grad_norm, args.target_kl, args.learning_rate, args.embed_dim, args.num_heads, args.num_layers, args.mixed_precision, args.use_data_parallel, args.seed)

if __name__ == "__main__":
    main() 