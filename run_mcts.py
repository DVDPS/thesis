import numpy as np
import torch
import time
from agents.cnn_mcts import CNNMCTSAgent
from src.thesis.environment.game2048 import Game2048

def run_mcts(num_episodes: int = 100, num_simulations: int = 200):
    agent = CNNMCTSAgent(num_simulations=num_simulations, use_gpu=True)
    game = Game2048(seed=42)
    
    total_score = 0
    max_tile_overall = 0
    scores = []
    steps_per_episode = []
    max_tiles = []
    times_per_episode = []
    start_time = time.time()
    
    print(f"\nStarting {num_episodes} episodes with MCTS (simulations={num_simulations})")
    print(f"Using device: {agent.device}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = game.reset()
        done = False
        episode_score = 0
        steps = 0
        current_max_tile = 0
        
        while not done:
            if isinstance(state, torch.Tensor):
                state_np = state.cpu().numpy()
            else:
                state_np = state
            
            action = agent.get_move(state_np)
            new_board_tensor, score_gain, changed = game._move(game.board, action)
            
            if changed:
                game.board = new_board_tensor
                game.score += score_gain
                game.add_random_tile()
                episode_score = game.score
                state = game.board
                steps += 1
                
                current_max_tile = max(current_max_tile, torch.max(game.board).item())
                
                if steps % 10 == 0:
                    print(f"\rEpisode {episode+1}/{num_episodes} | Step {steps} | Score: {episode_score:,} | Max Tile: {int(current_max_tile)}", end="")
            
            done = game.is_game_over()
        
        episode_time = time.time() - episode_start_time
        final_max_tile = torch.max(game.board).item()
        
        total_score += episode_score
        scores.append(episode_score)
        steps_per_episode.append(steps)
        max_tiles.append(final_max_tile)
        times_per_episode.append(episode_time)
        max_tile_overall = max(max_tile_overall, final_max_tile)
        
        print(f"\nEpisode {episode+1}/{num_episodes} completed in {episode_time:.1f}s")
        print(f"Final Score: {episode_score:,} | Steps: {steps} | Max Tile: {int(final_max_tile)}")
        print(f"Running Average Score: {total_score/(episode+1):,.1f}")
        print("-" * 50)
        
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_score = total_score/(episode+1)
            avg_steps = sum(steps_per_episode)/len(steps_per_episode)
            avg_max_tile = sum(max_tiles)/len(max_tiles)
            avg_time = sum(times_per_episode)/len(times_per_episode)
            
            print(f"\n--- Progress Report (Episode {episode+1}/{num_episodes}) ---")
            print(f"Time Elapsed: {elapsed_time:.1f}s")
            print(f"Average Score: {avg_score:,.1f}")
            print(f"Average Steps: {avg_steps:.1f}")
            print(f"Average Max Tile: {avg_max_tile:.1f}")
            print(f"Average Time per Episode: {avg_time:.1f}s")
            print(f"Best Score So Far: {max(scores):,}")
            print(f"Highest Max Tile: {int(max_tile_overall)}")
            print("=" * 50)
    
    total_time = time.time() - start_time
    print("\n--- Final Statistics ---")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time per Episode: {total_time/num_episodes:.1f}s")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Average Score: {total_score/num_episodes:,.1f}")
    print(f"Best Score: {max(scores):,}")
    print(f"Highest Max Tile: {int(max_tile_overall)}")
    print(f"Average Max Tile: {sum(max_tiles)/len(max_tiles):.1f}")
    print(f"Average Steps per Episode: {sum(steps_per_episode)/len(steps_per_episode):.1f}")
    print("-" * 50)
    print("Max Tile Distribution:")
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in zip(unique_tiles, counts):
        print(f"  {int(tile)}: {count} times")
    print("=" * 50)

if __name__ == "__main__":
    print("Starting MCTS with CNN evaluator...")
    run_mcts(num_episodes=100, num_simulations=200) 