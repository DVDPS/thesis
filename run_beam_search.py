import numpy as np
import torch
import time
from src.thesis.environment.game2048 import Game2048
from agents.cnn_beam_search import CNNBeamSearchAgent

def run_beam_search_episodes(num_episodes=100, beam_width=10, search_depth=5):
    print(f"Starting {num_episodes} episodes with Beam Search (beam_width={beam_width}, depth={search_depth})")
    
    # Initialize agent and environment
    agent = CNNBeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
    print(f"Using device: {agent.device}")
    print("=" * 50)
    
    # Statistics tracking
    scores = []
    max_tiles = []
    steps_list = []
    total_time = 0
    
    for episode in range(1, num_episodes + 1):
        env = Game2048()
        done = False
        episode_start_time = time.time()
        steps = 0
        
        while not done:
            # Get current state and agent's action
            state = env.board.cpu().numpy()
            action = agent.get_move(state)
            
            # Take action in environment
            new_board, reward, changed = env._move(env.board, action)
            if changed:
                env.board = torch.tensor(new_board, dtype=torch.float32)
                env.score += reward
                env.add_random_tile()
                done = env.is_game_over()
                steps += 1
            
            # Calculate max tile from board
            max_tile = int(torch.max(env.board).item())
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"Episode {episode} | Step {steps} | Score: {env.score:,} | Max Tile: {max_tile}")
        
        # Episode complete - record statistics
        episode_time = time.time() - episode_start_time
        total_time += episode_time
        scores.append(env.score)
        max_tiles.append(max_tile)
        steps_list.append(steps)
        
        # Print episode summary
        print(f"Episode {episode} completed in {episode_time:.1f}s")
        print(f"Final Score: {env.score:,} | Steps: {steps} | Max Tile: {max_tile}")
        print("-" * 50)
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Average Steps: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"Average Max Tile: {np.mean(max_tiles):.1f}")
    print(f"Max Tile Reached: {max(max_tiles)}")
    print(f"Average Time per Episode: {total_time/num_episodes:.1f}s")
    print(f"Total Time: {total_time:.1f}s")

if __name__ == "__main__":
    run_beam_search_episodes(num_episodes=100, beam_width=10, search_depth=5) 