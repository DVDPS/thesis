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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    tqdm_handler = logging.StreamHandler()
    tqdm_handler.terminator = ""
    tqdm.pandas(desc="Processing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]', file=tqdm_handler)

    return logging.getLogger()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def train_cnn_agent(num_episodes: int = 10000, epsilon: float = 0.5, batch_size: int = 512, update_frequency: int = 1):
    logger = setup_logging()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    agent = CNNAgent(device, buffer_size=100000, batch_size=batch_size)
    game = Game2048(seed=42)
    
    mem_alloc, mem_reserved = get_gpu_memory_usage()
    logger.info(f"Initial GPU Memory Usage: Allocated={mem_alloc:.2f} GB, Reserved={mem_reserved:.2f} GB")
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
    
    min_experiences = 500
    update_interval = 25
    max_batches_per_update = 16

    epsilon_start = epsilon
    epsilon_decay_episodes = 100000
    min_epsilon = 0.3
    
    milestone_rewards = {
        512: 50, 1024: 150, 2048: 400, 4096: 1000, 8192: 2500
    }
    
    total_steps_accumulated = 0
    max_steps_per_episode = 10000
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        if episode < 100:
            agent.target_update_frequency = initial_update_freq
        else:
            progress = min(1.0, (episode - 100) / (num_episodes * 0.1))
            agent.target_update_frequency = int(initial_update_freq + progress * (final_update_freq - initial_update_freq))

        state = game.reset()
        if isinstance(state, torch.Tensor):
             state = state.cpu().numpy()
        done = False
        episode_reward_sum = 0
        previous_max_tile_in_episode = 0
        steps_this_episode = 0
        episode_loss_sum = 0.0
        num_updates_this_episode = 0
        consecutive_invalid_moves = 0
        
        while not done and steps_this_episode < max_steps_per_episode:
            steps_this_episode += 1
            prev_board = state.copy()

            current_epsilon = agent.epsilon

            if np.random.rand() < current_epsilon:
                action = np.random.choice([0, 1, 2, 3])
                is_random_move = True
            else:
                is_random_move = False
                action_values = agent.evaluate_all_actions(state, game)
                if action_values:
                    best_action_info = max(action_values, key=lambda x: x[1] + 0.95 * x[2])
                    action = best_action_info[0]
                else:
                    action = np.random.choice([0, 1, 2, 3])
            if isinstance(game.board, np.ndarray):
                 board_tensor = torch.from_numpy(game.board).float().to(device)
            else:
                 board_tensor = game.board.to(device)

            new_board_tensor, score_gain, changed = game._move(board_tensor, action)

            if changed:
                consecutive_invalid_moves = 0
                game.board = new_board_tensor
                game.score += score_gain
                game.add_random_tile()
                
                next_state = game.board.cpu().numpy()
                current_max_tile_value = np.max(next_state)
                
                max_tile_history.append(current_max_tile_value)
                if current_max_tile_value > max_tile_reached:
                    max_tile_reached = current_max_tile_value
                
                game_is_over = game.is_game_over()
                if game_is_over:
                     max_tile_count[current_max_tile_value] = max_tile_count.get(current_max_tile_value, 0) + 1

                step_reward = 0
                if current_max_tile_value in milestone_rewards:
                    step_reward += milestone_rewards[current_max_tile_value]
                
                if current_max_tile_value > previous_max_tile_in_episode:
                    base_log = math.log2(previous_max_tile_in_episode) if previous_max_tile_in_episode > 0 else 0
                    level_up_bonus = math.log2(current_max_tile_value) - base_log
                    step_reward += 3 * level_up_bonus * math.log2(current_max_tile_value)
                    previous_max_tile_in_episode = current_max_tile_value
                
                if current_max_tile_value >= 512:
                    step_reward += current_max_tile_value * 0.05
                
                step_reward += score_gain / 25.0
                
                if current_max_tile_value >= 32:
                    corner_bonus = 0
                    corners = [next_state[0, 0], next_state[0, 3], next_state[3, 0], next_state[3, 3]]
                    if current_max_tile_value in corners:
                         corner_bonus = current_max_tile_value * 0.1
                    step_reward += corner_bonus
                episode_reward_sum += step_reward
                agent.store_experience(prev_board, step_reward, next_state, game_is_over)
                total_steps_accumulated += 1
                if total_steps_accumulated % update_interval == 0 and len(agent.replay_buffer) >= min_experiences:
                    num_batches_to_run = min(max_batches_per_update, max(1, len(agent.replay_buffer) // agent.batch_size))
                    batch_loss = agent.update_batch(num_batches=num_batches_to_run)
                
                    if batch_loss is not None:
                         episode_loss_sum += batch_loss * num_batches_to_run
                         num_updates_this_episode += num_batches_to_run

                state = next_state
            
            else:
                consecutive_invalid_moves += 1
                if consecutive_invalid_moves >= 4 and not is_random_move:
                     logger.debug(f"Episode {episode+1}: Stuck state detected after {consecutive_invalid_moves} invalid non-random moves.")
                     done = True 
                     game_is_over = True

            if not done:
                 game_is_over = game.is_game_over()
                 if game_is_over:
                      done = True
                      final_max_tile = np.max(game.board.cpu().numpy())
                      max_tile_count[final_max_tile] = max_tile_count.get(final_max_tile, 0) + 1

            current_score = game.score
            current_max_tile_on_board = np.max(game.board.cpu().numpy())
            if current_score > best_board_score or \
               (current_score == best_board_score and current_max_tile_on_board > best_max_tile):
                best_board_score = current_score
                best_max_tile = current_max_tile_on_board
                best_board_state = game.board.cpu().numpy().copy()
                agent.save("best_cnn_model.pth")
        
        # End of episode
        agent.episode_count += 1 # Increment agent's episode counter for epsilon decay

        episode_rewards.append(episode_reward_sum)
        episode_scores.append(game.score)
        final_max_tile = np.max(game.board.cpu().numpy())
        episode_max_tiles.append(final_max_tile)
        episode_steps.append(steps_this_episode)
        avg_episode_loss = episode_loss_sum / num_updates_this_episode if num_updates_this_episode > 0 else 0
        episode_losses.append(avg_episode_loss)
        
        if (episode + 1) % 50 == 0:
            last_10_episodes = -50 
            recent_rewards = episode_rewards[last_10_episodes:]
            recent_scores = episode_scores[last_10_episodes:]
            recent_steps = episode_steps[last_10_episodes:]
            recent_losses = episode_losses[last_10_episodes:]
            recent_max_tiles_ep = episode_max_tiles[last_10_episodes:]
            
            max_tile_last_log = max(recent_max_tiles_ep) if recent_max_tiles_ep else 0
            sorted_tile_counts = sorted(max_tile_count.items(), key=lambda x: x[0], reverse=True)
            
            logger.info(f"\n--- Episode {episode+1}/{num_episodes} Report ---")
            logger.info(f"Recent {abs(last_10_episodes)} Episodes Stats:")
            logger.info(f"  Avg Reward: {np.mean(recent_rewards):,.1f}")
            logger.info(f"  Avg Score: {np.mean(recent_scores):,.1f}")
            logger.info(f"  Avg Steps: {np.mean(recent_steps):.1f}")
            logger.info(f"  Avg Loss: {np.mean(recent_losses):.4f}")
            logger.info(f"  Max Tile (Last {abs(last_10_episodes)}): {int(max_tile_last_log)}")
            logger.info(f"Overall Stats:")
            logger.info(f"  Highest Tile Ever: {int(max_tile_reached)}")
            logger.info(f"  Best Score Seen: {best_board_score:,}")
            logger.info(f"  Best Max Tile Seen: {int(best_max_tile)}")
            logger.info(f"Max Tile Distribution (Overall):")
            for tile, count in sorted_tile_counts[:5]:
                logger.info(f"  {int(tile)}: {count} times")
            logger.info(f"Training Info:")
            logger.info(f"  Buffer Size: {len(agent.replay_buffer)}/{agent.buffer_size}")
            logger.info(f"  Epsilon: {agent.epsilon:.4f}")
            mem_alloc, mem_reserved = get_gpu_memory_usage()
            logger.info(f"  GPU Memory: Allocated={mem_alloc:.2f} GB, Reserved={mem_reserved:.2f} GB")
            logger.info(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"  Target Net Update Freq: {agent.target_update_frequency}")
            logger.info("-" * 50)
            
            agent.optimize_memory()
    
    logger.info("\n--- Training Complete ---")
    logger.info("Final Statistics (Averages over all episodes):")
    logger.info(f"Average Reward: {np.mean(episode_rewards):,.1f}")
    logger.info(f"Average Score: {np.mean(episode_scores):,.1f}")
    logger.info(f"Average Steps: {np.mean(episode_steps):.1f}")
    logger.info(f"Average Loss: {np.mean(episode_losses):.4f}")
    logger.info("Best Performance Recorded:")
    logger.info(f"Best Score: {best_board_score:,}")
    logger.info(f"Best Max Tile: {int(best_max_tile)}")
    logger.info(f"Highest Tile Ever Reached During Training: {int(max_tile_reached)}")
    logger.info("Overall Max Tile Distribution:")
    sorted_tile_counts = sorted(max_tile_count.items(), key=lambda x: x[0], reverse=True)
    for tile, count in sorted_tile_counts:
        logger.info(f"  {int(tile)}: {count} times")
    logger.info("Best Board State Saved:")
    logger.info(best_board_state)
    
    agent.optimize_memory()
    
    return agent, best_board_state, best_max_tile, best_board_score

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting CNN training...")
    
    num_train_episodes = 100000
    start_epsilon = 0.5       
    train_batch_size = 512     
    
    trained_agent, best_state, best_tile, best_score_val = train_cnn_agent(
        num_episodes=num_train_episodes,
        epsilon=start_epsilon,
        batch_size=train_batch_size,
        update_frequency=1
    )
    
    logger.info("\n--- Training complete and best model saved (best_cnn_model.pth) ---")
    logger.info("Best Performance Statistics Achieved During Training:")
    logger.info(f"Best Max Tile: {int(best_tile)}")
    logger.info(f"Best Board Score: {best_score_val:,}")
    logger.info("Corresponding Best Board State:")
    logger.info(best_state)