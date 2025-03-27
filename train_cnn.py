import numpy as np
import torch
from agents.cnn_agent import CNNAgent
from src.thesis.environment.game2048 import Game2048
import math
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging():
    """Setup logging to both file and console"""
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return logging.getLogger()

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    return 0

def train_cnn_agent(num_episodes: int = 10000, epsilon: float = 0.5, batch_size: int = 512, update_frequency: int = 1):
    # Setup logging
    logger = setup_logging()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize agent with smaller batch size
    agent = CNNAgent(device, buffer_size=100000, batch_size=batch_size)
    game = Game2048(seed=42)
    
    # Print initial GPU memory usage and model info
    logger.info(f"Initial GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
    logger.info(f"Initial replay buffer size: {len(agent.replay_buffer)}")
    
    best_score = 0
    best_board_score = 0
    best_max_tile = 0
    best_board_state = None
    
    # Training statistics
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    episode_losses = []
    
    # Track maximum tile statistics
    max_tile_history = []  # Track max tile for each episode
    max_tile_reached = 0   # Track highest tile ever reached
    max_tile_count = {}    # Count occurrences of each max tile
    
    # Progressive training parameters
    initial_update_freq = 50   # More frequent updates initially
    final_update_freq = 500    # Less frequent updates later
    update_freq_decay = 0.98   # Slower decay for update frequency
    
    # Experience collection parameters
    min_experiences = 500     # Lower minimum experiences to start learning earlier
    update_interval = 25      # Update more frequently
    max_batches = 16         # Process more batches per update
    
    # Slower epsilon decay for better exploration
    epsilon_decay_episodes = 100000  # Decay over full training period
    min_epsilon = 0.3  # Higher minimum epsilon for continued exploration
    
    # Enhanced milestone rewards for better high-tile achievement
    milestone_rewards = {
        512: 50,     # Increased from 20
        1024: 150,   # Increased from 50
        2048: 400,   # Increased from 120
        4096: 1000,  # Increased from 300
        8192: 2500   # Increased from 800
    }
    
    total_steps = 0
    max_steps_per_episode = 10000  # Prevent infinite episodes
    
    # Training loop with progress bar
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Slower epsilon decay
        current_epsilon = max(min_epsilon, epsilon - (episode / epsilon_decay_episodes))
        
        # Smooth transition between update frequencies
        if episode < 10:
            agent.target_update_frequency = initial_update_freq
        else:
            # Smooth transition between frequencies
            progress = min(1.0, (episode - 10) / 90)  # 0 to 1 over 100 episodes
            agent.target_update_frequency = int(initial_update_freq + progress * (final_update_freq - initial_update_freq))
        
        state = game.reset()
        done = False
        episode_reward = 0
        previous_max_tile = 0
        steps = 0
        episode_loss = 0
        num_updates = 0
        consecutive_invalid_moves = 0  # Track stuck states
        
        while not done and steps < max_steps_per_episode:
            steps += 1
            # Store previous board state (convert to numpy for storage)
            prev_board = state.copy() if isinstance(state, np.ndarray) else state.cpu().numpy()
            
            # Epsilon-greedy policy with dynamic exploration
            if np.random.rand() < current_epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                # Use the new evaluate_all_actions method for better efficiency
                action_values = agent.evaluate_all_actions(state, game)
                if action_values:
                    # Find best action based on value + reward
                    best_action, best_score, value = max(
                        action_values, 
                        key=lambda x: x[1] + 0.95 * x[2]  # score + discount * value
                    )[:3]
                    action = best_action
                else:
                    # No valid moves
                    action = np.random.choice([0, 1, 2, 3])
            
            new_board, score, changed = game._move(game.board, action)
            
            if changed:
                # Board changed - process normally
                consecutive_invalid_moves = 0
                game.board = new_board
                game.score += score
                game.add_random_tile()
                state = game.board.copy() if isinstance(game.board, np.ndarray) else game.board.cpu().numpy()
                
                # Enhanced reward calculation with scaled rewards
                current_max_tile = np.max(game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board)
                
                # Track max tile statistics
                max_tile_history.append(current_max_tile)
                if current_max_tile > max_tile_reached:
                    max_tile_reached = current_max_tile
                # Only update max_tile_count when the episode ends
                if game.is_game_over():
                    max_tile_count[current_max_tile] = max_tile_count.get(current_max_tile, 0) + 1
                
                # Add milestone rewards (already scaled)
                if current_max_tile in milestone_rewards:
                    episode_reward += milestone_rewards[current_max_tile]
                
                # Add bonus for reaching new max tile (with proper zero handling)
                if current_max_tile > previous_max_tile:
                    if previous_max_tile == 0:
                        # Special handling for first non-zero tile
                        level_up = math.log2(current_max_tile)
                    else:
                        level_up = math.log2(current_max_tile) - math.log2(previous_max_tile)
                    episode_reward += 3 * level_up * math.log2(current_max_tile)  # Scaled down from 300
                    previous_max_tile = current_max_tile
                
                # Add bonus for maintaining high-value tiles (increased weight)
                if current_max_tile >= 512:
                    episode_reward += current_max_tile * 0.05  # Increased from 0.01
                
                # Add base score (scaled)
                episode_reward += score / 25.0  # Increased from 50.0 to give more weight to immediate rewards
                
                # Add bonus for keeping high tiles in corners
                if current_max_tile >= 32:
                    corner_bonus = 0
                    if game.board[0, 0] == current_max_tile or game.board[0, 3] == current_max_tile or \
                       game.board[3, 0] == current_max_tile or game.board[3, 3] == current_max_tile:
                        corner_bonus = current_max_tile * 0.1
                    episode_reward += corner_bonus
                
                # Store experience every step
                agent.store_experience(prev_board, score, state, game.is_game_over())
                total_steps += 1
                
                # Update less frequently but with more batches
                if total_steps % update_interval == 0 and len(agent.replay_buffer) >= min_experiences:
                    batch_loss = 0
                    # Calculate number of batches based on buffer size
                    num_batches = min(max_batches, max(1, len(agent.replay_buffer) // agent.batch_size))
                    
                    # Process all batches in one go
                    for _ in range(num_batches):
                        loss = agent.update_batch(num_batches=1)  # Process one batch at a time
                        batch_loss += loss
                    
                    episode_loss += batch_loss / num_batches
                    num_updates += 1
                    
                    # Clean up GPU memory after batch processing
                    agent.optimize_memory()
            else:
                # Track consecutive invalid moves to detect stuck states
                consecutive_invalid_moves += 1
                if consecutive_invalid_moves >= 10:
                    # Game is stuck
                    done = True
                    logger.info(f"Episode {episode+1} terminated due to being stuck")
            
            # Check game termination conditions more aggressively
            if game.is_game_over() or np.count_nonzero(state == 0) == 0:
                done = True
            
            # Calculate info for logging
            info = {
                'score': game.score,
                'max_tile': np.max(game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board)
            }
            
            # Save best model based on board score and max tile
            if info['score'] > best_board_score or (info['score'] == best_board_score and info['max_tile'] > best_max_tile):
                best_board_score = info['score']
                best_max_tile = info['max_tile']
                best_board_state = state.copy() if isinstance(state, np.ndarray) else state.cpu().numpy()
                agent.save("best_cnn_model.pth")
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)
        episode_max_tiles.append(info['max_tile'])
        episode_steps.append(steps)
        episode_losses.append(episode_loss / num_updates if num_updates > 0 else 0)
        
        # Log progress more frequently (every 10 episodes)
        if (episode + 1) % 10 == 0:
            # Calculate statistics for last 10 episodes
            recent_rewards = episode_rewards[-10:]
            recent_scores = episode_scores[-10:]
            recent_steps = episode_steps[-10:]
            recent_losses = episode_losses[-10:]
            
            # Calculate max tile statistics
            recent_max_tiles = max_tile_history[-10:] if len(max_tile_history) >= 10 else max_tile_history
            max_tile_last_10 = max(recent_max_tiles) if recent_max_tiles else 0
            
            # Sort max tile counts for display
            sorted_tile_counts = sorted(max_tile_count.items(), key=lambda x: x[0], reverse=True)
            
            logger.info(f"\nEpisode {episode+1}/{num_episodes}")
            logger.info(f"Last 10 Episodes Statistics:")
            logger.info(f"  Average Reward: {np.mean(recent_rewards):,.0f}")
            logger.info(f"  Average Score: {np.mean(recent_scores):,.0f}")
            logger.info(f"  Average Steps: {np.mean(recent_steps):.1f}")
            logger.info(f"  Average Loss: {np.mean(recent_losses):.4f}")
            logger.info(f"  Max Tile Reached: {max_tile_last_10}")
            logger.info(f"\nOverall Statistics:")
            logger.info(f"  Highest Tile Ever: {max_tile_reached}")
            logger.info(f"  Best Score: {best_board_score:,.0f}")
            logger.info(f"  Best Max Tile: {best_max_tile}")
            logger.info(f"\nMax Tile Distribution:")
            for tile, count in sorted_tile_counts[:5]:  # Show top 5 most common max tiles
                logger.info(f"  {tile}: {count} times")
            logger.info(f"\nTraining Info:")
            logger.info(f"  Buffer Size: {len(agent.replay_buffer)}")
            logger.info(f"  Epsilon: {current_epsilon:.3f}")
            logger.info(f"  GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
            logger.info(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"  Valid Moves: {len(agent.get_valid_moves(state, game))}")
            # Fix empty cells counting
            board_array = game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board
            empty_cells = np.sum(board_array == 0)
            logger.info(f"  Empty Cells: {empty_cells}")
            logger.info(f"  Updates This Episode: {num_updates}")
            logger.info(f"  Target Network Update Frequency: {agent.target_update_frequency}")
            logger.info("-" * 50)
            
            # Clean up GPU memory after logging
            agent.optimize_memory()
    
    # Print final training statistics
    logger.info("\nTraining Complete!")
    logger.info("\nFinal Statistics:")
    logger.info(f"Average Reward: {np.mean(episode_rewards):,.0f}")
    logger.info(f"Average Score: {np.mean(episode_scores):,.0f}")
    logger.info(f"Average Steps: {np.mean(episode_steps):.1f}")
    logger.info(f"Average Loss: {np.mean(episode_losses):.4f}")
    logger.info(f"\nBest Performance:")
    logger.info(f"Best Score: {best_board_score:,.0f}")
    logger.info(f"Best Max Tile: {best_max_tile}")
    logger.info(f"Highest Tile Ever: {max_tile_reached}")
    logger.info(f"\nMax Tile Distribution:")
    for tile, count in sorted(max_tile_count.items(), key=lambda x: x[0], reverse=True)[:5]:
        logger.info(f"  {tile}: {count} times")
    logger.info("\nBest Board State:")
    logger.info(best_board_state)
    
    # Final cleanup
    agent.optimize_memory()
    
    return agent, best_board_state, best_max_tile, best_board_score

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting CNN training...")
    trained_agent, best_board_state, best_max_tile, best_board_score = train_cnn_agent(
        num_episodes=100000,
        epsilon=0.5,
        batch_size=512,  # Reduced batch size
        update_frequency=1  # Update more frequently
    )
    
    logger.info("\nTraining complete and best model saved.")
    logger.info("\nBest Performance Statistics:")
    logger.info(f"Best Max Tile: {best_max_tile}")
    logger.info(f"Best Board Score: {best_board_score:,.0f}")
    logger.info("\nBest Board State:")
    logger.info(best_board_state) 