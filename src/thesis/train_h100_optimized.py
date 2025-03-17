#!/usr/bin/env python
"""
Training script for PPO agent on 2048 game.
Optimized for multiple NVIDIA H100 GPUs with distributed training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
import time
import argparse
import matplotlib.pyplot as plt
import json
import datetime
from tqdm import tqdm
from src.thesis.environment.game2048 import Game2048, preprocess_state_onehot
from src.thesis.agents.ppo_agent import PPOAgent
from src.thesis.config import set_seeds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    try:
        # Use a random port to avoid conflicts
        port = 29500 + rank
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        
        # Set NCCL environment variables for better debugging and performance
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
        
        # Initialize the process group with timeout and a different backend as fallback
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
        except Exception as e:
            logging.warning(f"NCCL initialization failed: {e}, falling back to gloo backend")
            dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
        
        # Set device for this process
        torch.cuda.set_device(rank)
        logging.info(f"Process {rank} initialized successfully on port {port}")
    except Exception as e:
        logging.error(f"Error in setup for rank {rank}: {e}")
        raise

def cleanup():
    """
    Clean up the distributed environment.
    """
    try:
        dist.destroy_process_group()
    except Exception as e:
        logging.error(f"Error in cleanup: {e}")

def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    # Append a zero value for bootstrapping
    values = torch.cat([values, torch.zeros(1, device=values.device)])
    
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    
    # Iterate backwards over the rewards to compute GAE
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values[:-1]
    return advantages, returns

def gather_metrics(metrics, world_size, rank):
    """Gather metrics from all processes with error handling"""
    if not metrics:
        return []
    
    try:
        # Convert to tensor
        metrics_tensor = torch.tensor(metrics, device=f"cuda:{rank}")
        
        # Create output tensor
        gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
        
        # Gather metrics from all processes
        dist.all_gather(gathered_metrics, metrics_tensor)
        
        # Convert back to list
        return [item.cpu().numpy() for gathered in gathered_metrics for item in gathered]
    except Exception as e:
        logging.error(f"Error in gather_metrics for rank {rank}: {e}")
        # Return local metrics as fallback
        return metrics

def train_distributed_ppo(rank, world_size, args):
    """
    Train a PPO agent on the 2048 game using distributed training.
    
    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        args: Command-line arguments
    """
    try:
        # Setup the distributed environment
        setup(rank, world_size)
        
        # Create output directory
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            # Configure logging for rank 0 only (to avoid file conflicts)
            file_handler = logging.FileHandler(os.path.join(args.output_dir, 'training.log'), encoding='utf-8')
            logging.getLogger().addHandler(file_handler)
            
            # Create a separate log file just for max tile tracking
            max_tile_log_path = os.path.join(args.output_dir, 'max_tile_log.txt')
            with open(max_tile_log_path, 'w') as f:
                f.write("episode,max_tile,time\n")
        
        # Set random seeds for reproducibility
        set_seeds(args.seed + rank)  # Different seed per process
        
        # Log device and arguments
        device = torch.device(f"cuda:{rank}")
        logging.info(f"Rank {rank} using device: {device}")
        if rank == 0:
            logging.info(f"Using {world_size} GPUs")
            logging.info(f"Arguments: {args}")
            logging.info(f"Training for {args.episodes} episodes")
        
        # Create environment (each process has its own)
        env = Game2048()
        
        # Create agent with smaller batch size for stability
        actual_batch_size = args.batch_size // world_size  # Scale batch size per GPU
        logging.info(f"Rank {rank} using batch size: {actual_batch_size}")
        
        agent = PPOAgent(
            board_size=4,
            hidden_dim=256,  # Make sure this is a multiple of 8 for Tensor Cores
            input_channels=16,
            lr=args.learning_rate,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            gae_lambda=args.gae_lambda,
            update_epochs=args.update_epochs,
            target_kl=args.target_kl,
            batch_size=actual_batch_size,  # Scaled batch size
            mixed_precision=True
        )
        
        # Move agent's network to the correct device
        agent.network.to(device)
        
        # Wrap the model with DDP
        ddp_model = DDP(agent.network, device_ids=[rank], find_unused_parameters=True)
        agent.network = ddp_model
        
        # Set gradient accumulation steps
        grad_accumulation_steps = args.grad_accumulation_steps
        
        # Load checkpoint if provided
        if args.checkpoint:
            if rank == 0:
                logging.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            agent.network.module.load_state_dict(checkpoint['network_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Training metrics
        episode_rewards = []
        episode_max_tiles = []
        episode_lengths = []
        evaluation_scores = []
        evaluation_max_tiles = []
        losses = []
        
        # Max tile tracking
        max_tile_reached = 0
        max_tile_counts = {}
        max_tile_episodes = {}  # Track which episodes reached each max tile
        
        # Main training loop
        if rank == 0:
            logging.info(f"Starting training for {args.episodes} episodes")
        
        total_steps = 0
        start_time = time.time()
        
        # Determine episodes per process
        episodes_per_process = args.episodes // world_size
        start_episode = rank * episodes_per_process + 1
        end_episode = start_episode + episodes_per_process if rank < world_size - 1 else args.episodes + 1
        
        logging.info(f"Rank {rank} will process episodes {start_episode} to {end_episode}")
        
        for episode in tqdm(range(start_episode, end_episode), disable=rank != 0):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_max_tile = 0
            done = False
            
            # Initialize trajectory storage
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            valid_masks = []
            
            # Collect trajectory
            while not done and episode_length < args.max_steps:
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
                current_max_tile = np.max(next_state)
                episode_max_tile = max(episode_max_tile, current_max_tile)
                
                # Store transition
                states.append(state_proc)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                
                # Create and store valid actions mask
                valid_mask = torch.zeros(4, device=device)
                valid_mask[valid_moves] = 1.0
                valid_masks.append(valid_mask)
                
                # Update state
                state = next_state
                
                total_steps += 1
            
            # Update max tile tracking
            if episode_max_tile > max_tile_reached:
                max_tile_reached = episode_max_tile
                elapsed_time = time.time() - start_time
                if rank == 0:
                    logging.info(f"New max tile reached: {max_tile_reached} at episode {episode} (time: {elapsed_time:.2f}s)")
                    # Log to the dedicated max tile log file
                    with open(os.path.join(args.output_dir, 'max_tile_log.txt'), 'a') as f:
                        f.write(f"{episode},{max_tile_reached},{elapsed_time:.2f}\n")
            
            # Count max tiles
            if episode_max_tile in max_tile_counts:
                max_tile_counts[episode_max_tile] += 1
                max_tile_episodes[episode_max_tile].append(episode)
            else:
                max_tile_counts[episode_max_tile] = 1
                max_tile_episodes[episode_max_tile] = [episode]
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_max_tiles.append(episode_max_tile)
            episode_lengths.append(episode_length)
            
            # Skip update if trajectory is too short
            if len(states) < 10:
                logging.warning(f"Rank {rank}, Episode {episode}: Trajectory too short ({len(states)} steps), skipping update")
                continue
                
            # Prepare for PPO update
            try:
                # Convert all data to tensors
                state_tensors = [torch.tensor(s, dtype=torch.float, device=device) for s in states]
                action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
                reward_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
                value_tensor = torch.tensor(values, dtype=torch.float, device=device)
                log_prob_tensor = torch.tensor(log_probs, dtype=torch.float, device=device)
                done_tensor = torch.tensor(dones, dtype=torch.float, device=device)
                valid_mask_tensor = torch.stack(valid_masks)
                
                # Compute advantages and returns
                advantages, returns = compute_advantages(
                    rewards=reward_tensor,
                    values=value_tensor,
                    dones=done_tensor,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda
                )
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Create dataset and dataloader for this episode's data
                dataset = TensorDataset(
                    torch.stack(state_tensors),
                    action_tensor,
                    log_prob_tensor,
                    returns,
                    advantages,
                    valid_mask_tensor
                )
                
                # Use a simple random sampler instead of distributed sampler for stability
                dataloader = DataLoader(
                    dataset,
                    batch_size=agent.batch_size,
                    shuffle=True,
                    drop_last=False
                )
                
                # Perform PPO updates with gradient accumulation
                policy_losses = []
                value_losses = []
                entropy_losses = []
                total_losses = []
                
                # Set model to training mode
                agent.network.train()
                
                # Perform multiple epochs of updates
                for epoch in range(args.update_epochs):
                    # Track accumulated gradients
                    accumulated_loss = 0
                    
                    for batch_idx, (
                        mb_states,
                        mb_actions,
                        mb_old_log_probs,
                        mb_returns,
                        mb_advantages,
                        mb_valid_masks
                    ) in enumerate(dataloader):
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda'):  # Updated to new API
                            policy_logits, values = agent.network(mb_states)
                            
                            # Apply valid action mask
                            policy_logits = policy_logits + (1.0 - mb_valid_masks) * -1e10
                            
                            # Calculate new log probabilities
                            policy = torch.nn.functional.softmax(policy_logits, dim=1)
                            dist = torch.distributions.Categorical(policy)
                            new_log_probs = dist.log_prob(mb_actions)
                            entropy = dist.entropy().mean()
                            
                            # Calculate ratio and clipped loss
                            ratio = torch.exp(new_log_probs - mb_old_log_probs)
                            surr1 = ratio * mb_advantages
                            surr2 = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * mb_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()
                            
                            # Value loss
                            value_pred = values.squeeze()
                            value_loss = torch.nn.functional.mse_loss(value_pred, mb_returns)
                            
                            # Total loss
                            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                            
                            # Scale loss for gradient accumulation
                            loss = loss / grad_accumulation_steps
                        
                        # Backward pass with gradient scaling
                        agent.scaler.scale(loss).backward()
                        accumulated_loss += loss.item() * grad_accumulation_steps
                        
                        # Update weights if we've accumulated enough gradients
                        if (batch_idx + 1) % grad_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                            # Unscale gradients for clipping
                            agent.scaler.unscale_(agent.optimizer)
                            
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), args.max_grad_norm)
                            
                            # Update weights
                            agent.scaler.step(agent.optimizer)
                            agent.scaler.update()
                            agent.optimizer.zero_grad()
                            
                            # Track metrics
                            policy_losses.append(policy_loss.item())
                            value_losses.append(value_loss.item())
                            entropy_losses.append(entropy.item())
                            total_losses.append(accumulated_loss)
                            accumulated_loss = 0
                
                # Average losses
                avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
                avg_value_loss = np.mean(value_losses) if value_losses else 0
                avg_entropy_loss = np.mean(entropy_losses) if entropy_losses else 0
                avg_total_loss = np.mean(total_losses) if total_losses else 0
                
                losses.append(avg_total_loss)
                
                if episode % 10 == 0:
                    logging.info(f"Rank {rank}, Episode {episode}: Loss={avg_total_loss:.4f}, Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")
                
            except Exception as e:
                logging.error(f"Error in PPO update for rank {rank}, episode {episode}: {e}")
                continue
            
            # Log progress
            if rank == 0 and episode % args.log_interval == 0:
                try:
                    # Gather metrics from all processes
                    all_rewards = gather_metrics(episode_rewards[-args.log_interval:], world_size, rank)
                    all_max_tiles = gather_metrics(episode_max_tiles[-args.log_interval:], world_size, rank)
                    all_lengths = gather_metrics(episode_lengths[-args.log_interval:], world_size, rank)
                    all_losses = gather_metrics(losses[-args.log_interval:] if losses else [0], world_size, rank)
                    
                    avg_reward = np.mean(all_rewards)
                    avg_max_tile = np.mean(all_max_tiles)
                    avg_length = np.mean(all_lengths)
                    avg_loss = np.mean(all_losses)
                    
                    elapsed_time = time.time() - start_time
                    logging.info(f"Episode {episode}/{args.episodes} | "
                                f"Avg Reward: {avg_reward:.1f} | "
                                f"Avg Max Tile: {avg_max_tile:.1f} | "
                                f"Avg Length: {avg_length:.1f} | "
                                f"Avg Loss: {avg_loss:.6f} | "
                                f"Max Tile Reached: {max_tile_reached} | "
                                f"Time: {elapsed_time:.2f}s")
                    
                    # Log max tile distribution
                    if max_tile_counts:
                        logging.info(f"Max Tile Distribution: {json.dumps(max_tile_counts)}")
                except Exception as e:
                    logging.error(f"Error in logging for rank {rank}, episode {episode}: {e}")
            
            # Evaluate agent
            if rank == 0 and episode % args.eval_interval == 0:
                try:
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
                    save_checkpoint(agent, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Save best model if this is the best performance
                    if len(evaluation_max_tiles) == 1 or avg_eval_max_tile > max(evaluation_max_tiles[:-1]):
                        best_path = os.path.join(args.output_dir, "best_model.pt")
                        save_checkpoint(agent, best_path)
                        logging.info(f"Saved best model to {best_path}")
                except Exception as e:
                    logging.error(f"Error in evaluation for rank {rank}, episode {episode}: {e}")
        
        # Calculate training statistics
        if rank == 0:
            total_time = time.time() - start_time
            steps_per_sec = total_steps / total_time
            
            logging.info(f"Training completed in {total_time:.2f} seconds")
            logging.info(f"Average steps per second: {steps_per_sec:.1f}")
            
            # Log final max tile statistics
            logging.info(f"Final Max Tile Reached: {max_tile_reached}")
            logging.info(f"Max Tile Distribution: {json.dumps(max_tile_counts)}")
            logging.info(f"Max Tile Episodes: {json.dumps({str(k): v for k, v in max_tile_episodes.items()})}")
            
            # Save final model
            final_path = os.path.join(args.output_dir, "final_model.pt")
            save_checkpoint(agent, final_path)
            logging.info(f"Saved final model to {final_path}")
            
            # Plot training curves
            plot_training_curves(
                episode_rewards, episode_max_tiles, 
                evaluation_scores, evaluation_max_tiles,
                losses, max_tile_counts, args.output_dir
            )
        
        # Clean up distributed environment
        cleanup()
        
    except Exception as e:
        logging.error(f"Error in train_distributed_ppo for rank {rank}: {e}")
        # Try to clean up even if there was an error
        try:
            cleanup()
        except:
            pass

def save_checkpoint(agent, path):
    """Save model checkpoint"""
    try:
        # Get state dict from DDP model
        model_state_dict = agent.network.module.state_dict()
        
        torch.save({
            'network_state_dict': model_state_dict,
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'update_count': agent.update_count,
            'training_stats': agent.training_stats
        }, path)
    except Exception as e:
        logging.error(f"Error saving checkpoint to {path}: {e}")

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
            action, _, _ = agent.get_action(state_proc, valid_moves, deterministic=True)
            
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

def plot_training_curves(rewards, max_tiles, eval_scores, eval_max_tiles, losses, max_tile_counts, output_dir):
    """
    Plot training curves.
    
    Args:
        rewards: List of episode rewards
        max_tiles: List of episode max tiles
        eval_scores: List of evaluation scores
        eval_max_tiles: List of evaluation max tiles
        losses: List of training losses
        max_tile_counts: Dictionary of max tile counts
        output_dir: Directory to save plots
    """
    try:
        plt.figure(figsize=(15, 15))
        
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
        if max_tile_counts:
            tiles = sorted(max_tile_counts.keys())
            counts = [max_tile_counts[tile] for tile in tiles]
            plt.bar([str(v) for v in tiles], counts)
            plt.xlabel('Tile Value')
            plt.ylabel('Count')
            plt.title('Max Tile Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting training curves: {e}")

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train PPO agent for 2048 game with distributed training")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of update epochs")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--grad-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--single-gpu", action="store_true", help="Force single GPU training")
    
    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=10, help="Episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=100, help="Episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", type=str, default="h100_ppo_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available. Running on CPU.")
        world_size = 1
    
    # Check if we should use distributed training
    use_distributed = world_size > 1 and not args.single_gpu
    
    if use_distributed:
        print(f"Using {world_size} GPUs for distributed training")
        
        # Try distributed training
        try:
            mp.spawn(
                train_distributed_ppo,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            logging.error(f"Distributed training failed: {e}")
            logging.info("Falling back to single GPU training")
            
            # Fall back to single GPU training
            args.single_gpu = True
            use_distributed = False
    
    # If not using distributed training or if distributed training failed
    if not use_distributed:
        print("Using single GPU training")
        # Run on a single GPU
        train_distributed_ppo(0, 1, args)

if __name__ == "__main__":
    main() 