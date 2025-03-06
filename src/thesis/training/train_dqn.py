import argparse
import torch
import numpy as np
import logging
import os
from datetime import datetime
from ..environment.game2048 import Game2048, preprocess_state_onehot
from ..agents.dqn_agent import DQNAgent
from ..config import device

def setup_logging(checkpoint_dir):
    """Setup logging configuration"""
    log_file = os.path.join(checkpoint_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_dqn(args):
    """Train the DQN agent"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(args.checkpoint_dir)
    
    # Initialize environment and agent
    env = Game2048()
    agent = DQNAgent(
        board_size=4,
        hidden_dim=args.hidden_dim,
        input_channels=16,
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

    # Initialize training statistics
    start_episode = 0
    episode_rewards = []
    episode_max_tiles = []
    best_score = 0
    highest_tile = 0
    beta_start = 0.4
    beta_frames = 100000
    MIN_TILE_TO_LOG = 128  # Minimum tile value to log

    # Load checkpoint if specified
    if args.resume_from:
        try:
            logging.info(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Load model and training state
                if 'model_state_dict' in checkpoint:
                    agent.load_state_dict(checkpoint['model_state_dict'])
                if 'target_state_dict' in checkpoint:
                    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
                agent.update_count = checkpoint.get('update_count', 0)
                
                # Restore replay buffer state if available
                if all(k in checkpoint for k in ['replay_buffer', 'replay_priorities', 'replay_pos']):
                    agent.replay_buffer.buffer = checkpoint['replay_buffer']
                    agent.replay_buffer.priorities = checkpoint['replay_priorities']
                    agent.replay_buffer.pos = checkpoint['replay_pos']
                else:
                    logging.warning("Replay buffer state not found in checkpoint, starting with empty buffer")
                
                # Restore GradScaler state if available
                if 'scaler_state' in checkpoint:
                    agent.scaler.load_state_dict(checkpoint['scaler_state'])
                else:
                    logging.warning("GradScaler state not found in checkpoint, using default initialization")
                
                # Load training statistics
                start_episode = checkpoint.get('episode', 0) + 1
                episode_rewards = checkpoint.get('episode_rewards', [])
                episode_max_tiles = checkpoint.get('episode_max_tiles', [])
                best_score = checkpoint.get('best_score', 0)
                highest_tile = checkpoint.get('highest_tile', 0)
                
                logging.info(f"Resuming from episode {start_episode}")
                if best_score > 0:
                    logging.info(f"Previous best score: {best_score}")
                if highest_tile > 0:
                    logging.info(f"Previous highest tile: {highest_tile}")
                logging.info(f"Current epsilon: {agent.epsilon:.3f}")
            else:
                logging.warning("Checkpoint format not recognized, starting with fresh statistics")
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            return
    
    logging.info(f"Starting training with device: {device}")
    logging.info(f"Model parameters: {sum(p.numel() for p in agent.parameters())}")
    
    for episode in range(start_episode, args.episodes):
        state = env.reset()
        state = preprocess_state_onehot(state)
        done = False
        episode_reward = 0
        max_tile = 0
        new_max_achieved = False  # Flag to track if a new max tile was achieved in this episode
        
        # Calculate current beta for PER
        beta = min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)
        
        while not done:
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Select and perform action
            action = agent.get_action(state, valid_moves)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state_onehot(next_state)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update network
            loss = agent.update(beta=beta)
            
            # Update statistics
            episode_reward += reward
            current_max_tile = np.max(env.board)
            max_tile = max(max_tile, current_max_tile)
            
            # Check for new highest tile but don't log it yet
            if current_max_tile > highest_tile:
                highest_tile = current_max_tile
                new_max_achieved = True
            
            state = next_state
        
        # Log max tile achievement at the end of the episode
        if new_max_achieved and highest_tile >= MIN_TILE_TO_LOG:
            logging.info(f"New highest tile achieved: {highest_tile} (Episode {episode + 1})")
            logging.info(f"Final board state for new max tile {highest_tile}:\n{np.array2string(env.board, separator=', ')}")
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_max_tiles.append(max_tile)
        
        # Save best model with additional statistics
        if episode_reward > best_score:
            best_score = episode_reward
            best_model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pt")
            checkpoint = {
                'model_state_dict': agent.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'update_count': agent.update_count,
                'replay_buffer': agent.replay_buffer.buffer,
                'replay_priorities': agent.replay_buffer.priorities,
                'replay_pos': agent.replay_buffer.pos,
                'scaler_state': agent.scaler.state_dict(),
                # Add training statistics
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_max_tiles': episode_max_tiles,
                'best_score': best_score,
                'highest_tile': highest_tile
            }
            torch.save(checkpoint, best_model_path)
            logging.info(f"New best score: {best_score} - Model saved to {best_model_path}")
            logging.info(f"Final board state for new best score {best_score}:\n{np.array2string(env.board, separator=', ')}")
        
        # Save periodic checkpoint with statistics
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_{episode+1}.pt")
            checkpoint = {
                'model_state_dict': agent.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'update_count': agent.update_count,
                'replay_buffer': agent.replay_buffer.buffer,
                'replay_priorities': agent.replay_buffer.priorities,
                'replay_pos': agent.replay_buffer.pos,
                'scaler_state': agent.scaler.state_dict(),
                # Add training statistics
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_max_tiles': episode_max_tiles,
                'best_score': best_score,
                'highest_tile': highest_tile
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_max_tile = np.mean(episode_max_tiles[-100:])
            logging.info(f"Episode {episode+1}/{args.episodes} - "
                        f"Avg Reward: {avg_reward:.2f} - "
                        f"Avg Max Tile: {avg_max_tile:.2f} - "
                        f"Epsilon: {agent.epsilon:.3f} - "
                        f"Beta: {beta:.3f}")
    
    # Save final model with statistics
    final_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_final.pt")
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'update_count': agent.update_count,
        'replay_buffer': agent.replay_buffer.buffer,
        'replay_priorities': agent.replay_buffer.priorities,
        'replay_pos': agent.replay_buffer.pos,
        'scaler_state': agent.scaler.state_dict(),
        # Add training statistics
        'episode': args.episodes - 1,
        'episode_rewards': episode_rewards,
        'episode_max_tiles': episode_max_tiles,
        'best_score': best_score,
        'highest_tile': highest_tile
    }
    torch.save(checkpoint, final_path)
    logging.info(f"Training completed - Final model saved to {final_path}")
    logging.info(f"Highest tile achieved during training: {highest_tile}")

def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for 2048")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of episodes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--update_freq", type=int, default=4, help="Network update frequency")
    parser.add_argument("--save_freq", type=int, default=1000, help="Model save frequency")
    parser.add_argument("--checkpoint_dir", type=str, default="models/dueling_dqn", help="Directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="dueling_dqn_per", help="Model name prefix")
    parser.add_argument("--use_per", action="store_true", help="Use Prioritized Experience Replay")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling DQN architecture")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    train_dqn(args)

if __name__ == "__main__":
    main() 