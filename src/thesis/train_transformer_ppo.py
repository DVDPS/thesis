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
    Set up curriculum learning parameters based on training progress
    
    Args:
        episode_count: Current episode count
        max_episodes: Maximum number of episodes for scaling
        
    Returns:
        Dictionary of curriculum parameters
    """
    progress = min(1.0, episode_count / max_episodes)
    
    # Early training: focus on exploration and simple patterns
    if progress < 0.2:
        return {
            "entropy_coef": 0.02,
            "learning_rate": 0.0001,
            "target_kl": 0.03
        }
    # Mid training: balance exploration and exploitation
    elif progress < 0.6:
        return {
            "entropy_coef": 0.01, 
            "learning_rate": 0.00005,
            "target_kl": 0.02
        }
    # Late training: focus on exploitation and fine-tuning
    else:
        return {
            "entropy_coef": 0.005,
            "learning_rate": 0.00002,
            "target_kl": 0.01
        }

def train_transformer_ppo(
    output_dir: str,
    total_timesteps: int = 1_000_000,
    batch_size: int = 128,
    n_epochs: int = 10,
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.1,
    clip_range_vf: float = 0.1,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    learning_rate: float = 0.0001,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    mixed_precision: bool = True,
    data_parallel: bool = False,
    seed: int = 42
):
    """
    Train a Transformer-based PPO agent on 2048 game with two-stage learning.
    """
    # Set random seed for reproducibility
    set_seeds(seed)
    
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
    
    # Stage 1 parameters (early training)
    stage1_params = {
        'learning_rate': learning_rate * 2,  # Higher learning rate for exploration
        'ent_coef': ent_coef * 2,  # Higher entropy for exploration
        'target_kl': target_kl * 2,  # Larger policy updates
        'clip_range': clip_range * 1.5,  # More aggressive clipping
        'batch_size': batch_size // 2  # Smaller batches for faster updates
    }
    
    # Stage 2 parameters (fine-tuning)
    stage2_params = {
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'target_kl': target_kl,
        'clip_range': clip_range,
        'batch_size': batch_size
    }
    
    # Create environment and agent
    env = Game2048()
    agent = TransformerPPOAgent(
        board_size=4,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        input_channels=16
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(agent.network.parameters(), lr=stage1_params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    episode_count = 0
    total_steps = 0
    best_score = float('-inf')
    best_model_path = None
    
    while total_steps < total_timesteps:
        # Determine current stage based on total steps
        is_stage1 = total_steps < total_timesteps * 0.5
        current_params = stage1_params if is_stage1 else stage2_params
        
        # Update hyperparameters based on stage
        agent.ent_coef = current_params['ent_coef']
        agent.target_kl = current_params['target_kl']
        agent.clip_range = current_params['clip_range']
        
        # Training episode
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_max_tile = 0
        
        while not done and episode_length < n_steps:
            # Get action with tile-downgrading search
            action, log_prob, value = agent.get_action(state, env.get_possible_moves())
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            episode_max_tile = max(episode_max_tile, np.max(next_state))
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            # Update state
            state = next_state
        
        # Update agent with current stage parameters
        update_info = agent.update(
            batch_size=current_params['batch_size'],
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=current_params['clip_range'],
            clip_range_vf=clip_range_vf,
            ent_coef=current_params['ent_coef'],
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=current_params['target_kl']
        )
        
        # Update total steps and episode count
        total_steps += episode_length
        episode_count += 1
        
        # Log metrics
        logging.info(f"Episode {episode_count} | "
                    f"Score: {episode_reward:.2f} | "
                    f"Max Tile: {episode_max_tile} | "
                    f"Length: {episode_length} | "
                    f"Stage: {'1' if is_stage1 else '2'}")
        
        # Save best model
        if episode_reward > best_score:
            best_score = episode_reward
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            agent.save(best_model_path)
            logging.info(f"New best model saved with score: {best_score:.2f}")
        
        # Save periodic checkpoints
        if episode_count % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_eval_{episode_count}.pt')
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint at episode {episode_count}")
        
        # Update learning rate based on performance
        scheduler.step(episode_reward)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    agent.save(final_model_path)
    logging.info("Training completed. Final model saved.")
    
    return best_model_path

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