import numpy as np
import torch
from agents.cnn_agent import CNNAgent
from src.thesis.environment.game2048 import Game2048
import math
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def train_cnn_agent(num_episodes: int = 10000, epsilon: float = 0.5, batch_size: int = 512, update_frequency: int = 1):
    logger = setup_logging()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    agent = CNNAgent(device, buffer_size=100000, batch_size=batch_size)
    game = Game2048(seed=42)
    
    logger.info(f"Initial GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
    logger.info(f"Initial replay buffer size: {len(agent.replay_buffer)}")
    
    best_score = 0
    best_board_score = 0
    best_max_tile = 0
    best_board_state = None
    
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    episode_losses = []
    
    max_tile_history = []
    max_tile_reached = 0
    max_tile_count = {}
    
    initial_update_freq = 50
    final_update_freq = 500
    update_freq_decay = 0.98
    
    min_experiences = 500
    update_interval = 25
    max_batches = 16
    
    epsilon_decay_episodes = 100000
    min_epsilon = 0.3
    
    milestone_rewards = {
        512: 50,
        1024: 150,
        2048: 400,
        4096: 1000,
        8192: 2500
    }
    
    total_steps = 0
    max_steps_per_episode = 10000
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        current_epsilon = max(min_epsilon, epsilon - (episode / epsilon_decay_episodes))
        
        if episode < 10:
            agent.target_update_frequency = initial_update_freq
        else:
            progress = min(1.0, (episode - 10) / 90)
            agent.target_update_frequency = int(initial_update_freq + progress * (final_update_freq - initial_update_freq))
        
        state = game.reset()
        done = False
        episode_reward = 0
        previous_max_tile = 0
        steps = 0
        episode_loss = 0
        num_updates = 0
        consecutive_invalid_moves = 0
        
        while not done and steps < max_steps_per_episode:
            steps += 1
            prev_board = state.copy() if isinstance(state, np.ndarray) else state.cpu().numpy()
            
            if np.random.rand() < current_epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action_values = agent.evaluate_all_actions(state, game)
                if action_values:
                    best_action, best_score, value = max(
                        action_values, 
                        key=lambda x: x[1] + 0.95 * x[2]
                    )[:3]
                    action = best_action
                else:
                    action = np.random.choice([0, 1, 2, 3])
            
            new_board, score, changed = game._move(game.board, action)
            
            if changed:
                consecutive_invalid_moves = 0
                game.board = new_board
                game.score += score
                game.add_random_tile()
                state = game.board.copy() if isinstance(game.board, np.ndarray) else game.board.cpu().numpy()
                
                current_max_tile = np.max(game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board)
                
                max_tile_history.append(current_max_tile)
                if current_max_tile > max_tile_reached:
                    max_tile_reached = current_max_tile
                if game.is_game_over():
                    max_tile_count[current_max_tile] = max_tile_count.get(current_max_tile, 0) + 1
                
                if current_max_tile in milestone_rewards:
                    episode_reward += milestone_rewards[current_max_tile]
                
                if current_max_tile > previous_max_tile:
                    if previous_max_tile == 0:
                        level_up = math.log2(current_max_tile)
                    else:
                        level_up = math.log2(current_max_tile) - math.log2(previous_max_tile)
                    episode_reward += 3 * level_up * math.log2(current_max_tile)
                    previous_max_tile = current_max_tile
                
                if current_max_tile >= 512:
                    episode_reward += current_max_tile * 0.05
                
                episode_reward += score / 25.0
                
                if current_max_tile >= 32:
                    corner_bonus = 0
                    if game.board[0, 0] == current_max_tile or game.board[0, 3] == current_max_tile or \
                       game.board[3, 0] == current_max_tile or game.board[3, 3] == current_max_tile:
                        corner_bonus = current_max_tile * 0.1
                    episode_reward += corner_bonus
                
                agent.store_experience(prev_board, score, state, game.is_game_over())
                total_steps += 1
                
                if total_steps % update_interval == 0 and len(agent.replay_buffer) >= min_experiences:
                    batch_loss = 0
                    num_batches = min(max_batches, max(1, len(agent.replay_buffer) // agent.batch_size))
                    
                    for _ in range(num_batches):
                        loss = agent.update_batch(num_batches=1)
                        batch_loss += loss
                    
                    episode_loss += batch_loss / num_batches
                    num_updates += 1
                    
                    agent.optimize_memory()
            else:
                consecutive_invalid_moves += 1
                if consecutive_invalid_moves >= 10:
                    done = True
                    logger.info(f"Episode {episode+1} terminated due to being stuck")
            
            if game.is_game_over() or np.count_nonzero(state == 0) == 0:
                done = True
            
            info = {
                'score': game.score,
                'max_tile': np.max(game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board)
            }
            
            if info['score'] > best_board_score or (info['score'] == best_board_score and info['max_tile'] > best_max_tile):
                best_board_score = info['score']
                best_max_tile = info['max_tile']
                best_board_state = state.copy() if isinstance(state, np.ndarray) else state.cpu().numpy()
                agent.save("best_cnn_model.pth")
        
        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)
        episode_max_tiles.append(info['max_tile'])
        episode_steps.append(steps)
        episode_losses.append(episode_loss / num_updates if num_updates > 0 else 0)
        
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_scores = episode_scores[-10:]
            recent_steps = episode_steps[-10:]
            recent_losses = episode_losses[-10:]
            
            recent_max_tiles = max_tile_history[-10:] if len(max_tile_history) >= 10 else max_tile_history
            max_tile_last_10 = max(recent_max_tiles) if recent_max_tiles else 0
            
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
            for tile, count in sorted_tile_counts[:5]:
                logger.info(f"  {tile}: {count} times")
            logger.info(f"\nTraining Info:")
            logger.info(f"  Buffer Size: {len(agent.replay_buffer)}")
            logger.info(f"  Epsilon: {current_epsilon:.3f}")
            logger.info(f"  GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
            logger.info(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"  Valid Moves: {len(agent.get_valid_moves(state, game))}")
            board_array = game.board.cpu().numpy() if isinstance(game.board, torch.Tensor) else game.board
            empty_cells = np.sum(board_array == 0)
            logger.info(f"  Empty Cells: {empty_cells}")
            logger.info(f"  Updates This Episode: {num_updates}")
            logger.info(f"  Target Network Update Frequency: {agent.target_update_frequency}")
            logger.info("-" * 50)
            
            agent.optimize_memory()
    
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
    
    agent.optimize_memory()
    
    return agent, best_board_state, best_max_tile, best_board_score

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting CNN training...")
    trained_agent, best_board_state, best_max_tile, best_board_score = train_cnn_agent(
        num_episodes=100000,
        epsilon=0.5,
        batch_size=512,
        update_frequency=1
    )
    
    logger.info("\nTraining complete and best model saved.")
    logger.info("\nBest Performance Statistics:")
    logger.info(f"Best Max Tile: {best_max_tile}")
    logger.info(f"Best Board Score: {best_board_score:,.0f}")
    logger.info("\nBest Board State:")
    logger.info(best_board_state) 