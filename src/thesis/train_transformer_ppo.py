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

def train_transformer_ppo(args):
    """
    Train a Transformer-based PPO agent on the 2048 game.
    
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
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Log device and arguments
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {args}")
    
    # Initialize performance monitoring
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        # Start GPU monitoring if on Linux
        try:
            os.system('nvidia-smi -l 5 > ' + os.path.join(args.output_dir, 'gpu_usage.log') + ' &')
            gpu_monitor_pid = int(os.popen('echo $!').read().strip())
            logging.info(f"Started GPU monitoring with PID {gpu_monitor_pid}")
        except:
            logging.warning("Could not start GPU monitoring")
    
    # Create environment
    env = Game2048()
    
    # Create Transformer-based PPO agent
    agent = TransformerPPOAgent(
        board_size=4,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        input_channels=16,  # This matches the onehot encoding size
        lr=args.learning_rate,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        target_kl=args.target_kl,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision
    )
    
    # Print actual batch size for confirmation
    logging.info(f"Using batch size: {args.batch_size} (agent internal batch size: {agent.batch_size})")
    
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
    training_times = []
    
    # Configure data parallel if multiple GPUs available
    if torch.cuda.device_count() > 1 and args.use_data_parallel:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for data parallel")
        agent.network = torch.nn.DataParallel(agent.network)
    
    # Main training loop
    logging.info(f"Starting training for {args.total_timesteps} timesteps with Transformer-based PPO")
    
    timesteps_per_update = args.timesteps_per_update
    total_timesteps = 0
    update_count = 0
    start_time = time.time()
    
    with tqdm(total=args.total_timesteps) as pbar:
        while total_timesteps < args.total_timesteps:
            # Collect trajectories
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_max_tile = 0
            done = False
            
            while not done and episode_length < args.max_episode_length:
                # Process state
                state_proc = preprocess_state_onehot(state)
                
                # Get valid moves
                valid_moves = env.get_possible_moves()
                if not valid_moves:
                    break
                
                # Select action
                action, log_prob, value = agent.get_action(state_proc, valid_moves)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_max_tile = max(episode_max_tile, np.max(next_state))
                
                # Process next state
                next_state_proc = preprocess_state_onehot(next_state)
                
                # Get valid moves for next state
                next_valid_moves = env.get_possible_moves() if not done else []
                
                # Store transition
                agent.store_transition(
                    state_proc, action, reward, next_state_proc, 
                    done, log_prob, value, next_valid_moves
                )
                
                # Update state
                state = next_state
                
                # Update total timesteps
                total_timesteps += 1
                pbar.update(1)
                
                # Update agent if enough timesteps collected
                if len(agent.states) >= timesteps_per_update or done:
                    # Get value estimate for final state
                    if not done:
                        _, _, next_value = agent.get_action(next_state_proc, next_valid_moves)
                    else:
                        next_value = 0.0
                    
                    # Log transition count before update
                    transition_count = len(agent.states)
                    logging.info(f"Updating with {transition_count} transitions")
                    
                    # Update policy
                    update_start = time.time()
                    stats = agent.update(next_value)
                    update_time = time.time() - update_start
                    training_times.append(update_time)
                    
                    # Log update stats
                    update_count += 1
                    
                    # Enhanced debugging for loss values
                    if stats['policy_loss'] == 0.0 and stats['value_loss'] == 0.0 and stats['entropy'] == 0.0:
                        logging.warning(f"Update {update_count} produced zero losses!")
                        # Log the number of stored transitions for debugging
                        logging.warning(f"Number of transitions before update: {transition_count}")
                        logging.warning(f"Approx KL value: {stats['approx_kl']}")
                        # Try to check agent internals for debugging
                        try:
                            if hasattr(agent, 'training_stats') and len(agent.training_stats) > 0:
                                logging.warning(f"Last training stats record: {agent.training_stats[-1]}")
                        except Exception as e:
                            logging.error(f"Error accessing agent internals: {e}")
                    
                    logging.info(f"Update {update_count} | "
                                 f"Policy Loss: {stats['policy_loss']:.4f} | "
                                 f"Value Loss: {stats['value_loss']:.4f} | "
                                 f"Entropy: {stats['entropy']:.4f} | "
                                 f"Approx KL: {stats['approx_kl']:.4f} | "
                                 f"LR: {stats['learning_rate']:.6f} | "
                                 f"Time: {update_time:.2f}s")
                    
                    # Record stats in tensorboard
                    writer.add_scalar('losses/policy_loss', stats['policy_loss'], update_count)
                    writer.add_scalar('losses/value_loss', stats['value_loss'], update_count)
                    writer.add_scalar('losses/entropy', stats['entropy'], update_count)
                    writer.add_scalar('losses/approx_kl', stats['approx_kl'], update_count)
                    writer.add_scalar('training/update_time', update_time, update_count)
                    writer.add_scalar('training/learning_rate', stats['learning_rate'], update_count)
                
                # Break if reached total timesteps
                if total_timesteps >= args.total_timesteps:
                    break
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_max_tiles.append(episode_max_tile)
            episode_lengths.append(episode_length)
            
            # Apply curriculum learning to adjust hyperparameters
            if len(episode_rewards) % 10 == 0:  # Adjust parameters every 10 episodes
                curriculum_params = curriculum_setup(len(episode_rewards), max_episodes=args.total_timesteps/100)
                
                # Update agent hyperparameters
                agent.ent_coef = curriculum_params["entropy_coef"]
                agent.target_kl = curriculum_params["target_kl"]
                
                # Update learning rate in optimizer
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = curriculum_params["learning_rate"]
                
                # Log the curriculum update
                logging.info(f"Curriculum update: entropy_coef={agent.ent_coef}, "
                            f"target_kl={agent.target_kl}, "
                            f"learning_rate={curriculum_params['learning_rate']}")
                
                # Record in tensorboard
                writer.add_scalar('curriculum/entropy_coef', agent.ent_coef, len(episode_rewards))
                writer.add_scalar('curriculum/target_kl', agent.target_kl, len(episode_rewards))
                writer.add_scalar('curriculum/learning_rate', curriculum_params["learning_rate"], len(episode_rewards))
            
            # Log progress
            logging.info(f"Episode finished: Steps={episode_length}, "
                         f"Reward={episode_reward:.1f}, Max Tile={episode_max_tile}")
            
            # Record in tensorboard
            writer.add_scalar('episode/reward', episode_reward, len(episode_rewards))
            writer.add_scalar('episode/max_tile', episode_max_tile, len(episode_rewards))
            writer.add_scalar('episode/length', episode_length, len(episode_lengths))
            
            # Evaluate agent
            if len(episode_rewards) % args.eval_interval == 0:
                eval_results = evaluate_agent(agent, num_games=args.eval_episodes)
                avg_eval_score = eval_results['avg_score']
                avg_eval_max_tile = eval_results['avg_max_tile']
                
                evaluation_scores.append(avg_eval_score)
                evaluation_max_tiles.append(avg_eval_max_tile)
                
                logging.info(f"Evaluation | "
                             f"Avg Score: {avg_eval_score:.1f} | "
                             f"Avg Max Tile: {avg_eval_max_tile:.1f} | "
                             f"Best Max Tile: {eval_results['max_tile_reached']}")
                
                # Record in tensorboard
                writer.add_scalar('evaluation/avg_score', avg_eval_score, len(evaluation_scores))
                writer.add_scalar('evaluation/avg_max_tile', avg_eval_max_tile, len(evaluation_max_tiles))
                writer.add_scalar('evaluation/max_tile_reached', eval_results['max_tile_reached'], len(evaluation_scores))
                
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
    fps = total_timesteps / total_time
    
    logging.info(f"Training completed in {total_time:.2f} seconds")
    logging.info(f"Average FPS: {fps:.1f}")
    logging.info(f"Average update time: {np.mean(training_times):.4f}s")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    agent.save(final_path)
    logging.info(f"Saved final model to {final_path}")
    
    # Plot training curves
    plot_training_curves(
        episode_rewards, episode_max_tiles, 
        evaluation_scores, evaluation_max_tiles,
        args.output_dir
    )
    
    # Clean up GPU monitoring
    if torch.cuda.is_available() and 'gpu_monitor_pid' in locals():
        try:
            os.system(f'kill {gpu_monitor_pid}')
            logging.info("Stopped GPU monitoring")
        except:
            pass
    
    writer.close()
    
    return agent

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
    train_transformer_ppo(args)

if __name__ == "__main__":
    main() 