import numpy as np
import pickle
import torch
from agents.ntuple_network import OptimisticTDAgent
from src.thesis.environment.game2048 import Game2048
import math

def train_td_agent(num_episodes: int = 10000, epsilon: float = 0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_tuples = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    agent = OptimisticTDAgent(n_tuples, learning_rate=0.0001, optimistic_value=100)
    agent.to(device)
    game = Game2048(seed=42)
    
    best_score = 0
    best_board_score = 0
    best_max_tile = 0
    best_board_state = None
    best_weights = None
    td_errors = []
    
    milestone_rewards = {
        512: 500,
        1024: 1000,
        2048: 2000,
        4096: 4000,
        8192: 8000
    }
    
    state_tensor = torch.zeros((4, 4), dtype=torch.float32, device=device)
    next_state_tensor = torch.zeros((4, 4), dtype=torch.float32, device=device)
    
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        episode_reward = 0
        episode_td_errors = []
        previous_max_tile = 0
        
        while not done:
            state_tensor.copy_(torch.from_numpy(state))
            
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                best_val = -np.inf
                best_action = 0
                for a in [0, 1, 2, 3]:
                    temp_state = state.copy()
                    new_board, score, changed = game._move(temp_state, a)
                    if changed:
                        next_state_tensor.copy_(torch.from_numpy(new_board))
                        
                        val = score + 0.95 * agent.evaluate(next_state_tensor)
                        current_max_tile = np.max(new_board)
                        
                        if current_max_tile in milestone_rewards:
                            val += milestone_rewards[current_max_tile]
                        
                        if current_max_tile >= 512:
                            val += current_max_tile * 0.1
                        
                        if val > best_val:
                            best_val = val
                            best_action = a
                action = best_action
            
            prev_state = state.copy()
            new_board, score, changed = game._move(game.board, action)
            
            if changed:
                game.board = new_board
                game.score += score
                game.add_random_tile()
                state = game.board.copy()
                
                next_state_tensor.copy_(torch.from_numpy(state))
                
                current_max_tile = np.max(game.board)
                
                if current_max_tile in milestone_rewards:
                    episode_reward += milestone_rewards[current_max_tile]
                
                if current_max_tile > previous_max_tile:
                    if previous_max_tile == 0:
                        level_up = math.log2(current_max_tile)
                    else:
                        level_up = math.log2(current_max_tile) - math.log2(previous_max_tile)
                    episode_reward += 300 * level_up * math.log2(current_max_tile)
                    previous_max_tile = current_max_tile
                
                if current_max_tile >= 512:
                    episode_reward += current_max_tile * 0.1
                
                episode_reward += score
            
            td_error = agent.update(state_tensor, score, next_state_tensor, game.is_game_over())
            episode_td_errors.append(abs(td_error))
            
            info = {
                'score': game.score,
                'max_tile': np.max(game.board)
            }
            
            if info['score'] > best_board_score or (info['score'] == best_board_score and info['max_tile'] > best_max_tile):
                best_board_score = info['score']
                best_max_tile = info['max_tile']
                best_weights = agent.network.weights.copy()
                best_board_state = state.copy()
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"Reward: {episode_reward:,.0f} | Board Score: {info['score']:,.0f}")
                print(f"Max Tile: {info['max_tile']}")
                print(f"Best Board Score: {best_board_score:,.0f}")
                print(f"Best Max Tile: {best_max_tile}")
                print(f"Average TD Error: {np.mean(episode_td_errors):.4f}")
                print(f"Average TD Error (last 100): {np.mean(td_errors[-100:]):.4f}")
                print("-" * 50)
        
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
        td_errors.append(avg_td_error)
    
    return agent, best_weights, best_board_state, best_max_tile, best_board_score

if __name__ == "__main__":
    print("Starting TD learning training...")
    trained_agent, best_weights, best_board_state, best_max_tile, best_board_score = train_td_agent(num_episodes=100000, epsilon=0.1)
    
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(best_weights, f)
    
    print("\nTraining complete and best model saved.")
    print("\nBest Performance Statistics:")
    print(f"Best Max Tile: {best_max_tile}")
    print(f"Best Board Score: {best_board_score:,.0f}")
    print("\nBest Board State:")
    print(best_board_state) 