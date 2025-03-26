import numpy as np
import pickle
from agents.ntuple_network import OptimisticTDAgent
from src.thesis.environment.game2048 import Game2048

def train_td_agent(num_episodes: int = 10000, epsilon: float = 0.1):
    # Define n-tuple features for the board
    n_tuples = [
        [0, 1, 2, 3],  # First row
        [4, 5, 6, 7],  # Second row
        [8, 9, 10, 11],  # Third row
        [12, 13, 14, 15],  # Fourth row
        [0, 4, 8, 12],  # First column
        [1, 5, 9, 13],  # Second column
        [2, 6, 10, 14],  # Third column
        [3, 7, 11, 15],  # Fourth column
        [0, 1, 4, 5],  # Top-left 2x2
        [2, 3, 6, 7],  # Top-right 2x2
        [8, 9, 12, 13],  # Bottom-left 2x2
        [10, 11, 14, 15]  # Bottom-right 2x2
    ]
    
    # Initialize agent with smaller learning rate and optimistic value
    agent = OptimisticTDAgent(n_tuples, learning_rate=0.0001, optimistic_value=100)
    game = Game2048(seed=42)
    
    best_score = 0
    best_board_score = 0
    best_max_tile = 0
    best_board_state = None
    best_weights = None
    td_errors = []
    
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        episode_reward = 0
        episode_td_errors = []
        
        while not done:
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                # Greedy action selection using value estimates
                best_val = -np.inf
                best_action = 0
                for a in [0, 1, 2, 3]:
                    temp_state = state.copy()
                    next_state, reward, _, _ = game.step(a)
                    if not np.array_equal(temp_state, next_state):
                        val = reward + 0.95 * agent.evaluate(next_state)  # Add discount factor
                        if val > best_val:
                            best_val = val
                            best_action = a
                action = best_action
            
            prev_state = state.copy()
            next_state, reward, done, info = game.step(action)
            
            # Update the TD agent with discount factor
            td_error = agent.update(prev_state, reward, next_state, done)
            episode_td_errors.append(abs(td_error))  # Store absolute TD error
            episode_reward += reward
            state = next_state
        
        # Calculate average TD error for this episode
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
        td_errors.append(avg_td_error)
        
        # Save best model based on board score
        if info['score'] > best_board_score:
            best_board_score = info['score']
            best_weights = agent.network.weights.copy()
            best_board_state = state.copy()
        
        # Update best max tile
        if info['max_tile'] > best_max_tile:
            best_max_tile = info['max_tile']
        
        # Log progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"Reward: {episode_reward:,.0f} | Board Score: {info['score']:,.0f}")
            print(f"Max Tile: {info['max_tile']}")
            print(f"Best Board Score: {best_board_score:,.0f}")
            print(f"Best Max Tile: {best_max_tile}")
            print(f"Average TD Error: {avg_td_error:.4f}")
            print(f"Average TD Error (last 100): {np.mean(td_errors[-100:]):.4f}")
            print("-" * 50)
    
    return agent, best_weights, best_board_state, best_max_tile, best_board_score

if __name__ == "__main__":
    print("Starting TD learning training...")
    trained_agent, best_weights, best_board_state, best_max_tile, best_board_score = train_td_agent(num_episodes=100000, epsilon=0.1)
    
    # Save the best model weights
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(best_weights, f)
    
    print("\nTraining complete and best model saved.")
    print("\nBest Performance Statistics:")
    print(f"Best Max Tile: {best_max_tile}")
    print(f"Best Board Score: {best_board_score:,.0f}")
    print("\nBest Board State:")
    print(best_board_state) 