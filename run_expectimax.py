import pickle
import numpy as np
from agents.expectimax import ExpectimaxAgent, apply_tile_downgrading
from agents.ntuple_network import NTupleNetwork
from src.thesis.environment.game2048 import Game2048

class TrainedExpectimaxAgent(ExpectimaxAgent):
    def load_model(self, weights):
        """Load the trained model weights"""
        # Create NTupleNetwork with the same configuration as training
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
        
        # Initialize the value model
        self.value_model = NTupleNetwork(n_tuples)
        
        # Load the weights
        self.value_model.weights = weights.copy()  # Make sure to copy the weights
        
        # Print some statistics about the loaded model
        print(f"Loaded model with {len(weights)} n-tuple weights")
        print(f"Number of n-tuples: {len(n_tuples)}")
        
        # Test the model with a simple state
        test_state = np.zeros((4, 4))
        test_state[0, 0] = 2
        test_value = self.value_model.evaluate(test_state)
        print(f"Test evaluation: {test_value:.4f}")

    def _evaluate_state(self, state: np.ndarray) -> float:
        """Evaluate state using the trained model and enhanced heuristics"""
        # Apply tile downgrading if needed
        state = apply_tile_downgrading(state)
        
        # Get the model's evaluation
        model_value = self.value_model.evaluate(state)
        
        # Add heuristic bonuses for good board properties
        heuristic_value = 0
        
        # 1. Corner strategy (weighted more heavily)
        corners = [state[0,0], state[0,3], state[3,0], state[3,3]]
        max_corner = max(corners)
        if max_corner > 0:
            heuristic_value += max_corner * 0.2  # Increased weight
        
        # 2. Monotonic rows/columns (weighted more heavily)
        for i in range(4):
            row = state[i,:]
            col = state[:,i]
            # Check for strictly decreasing (preferred for snake pattern)
            if all(row[j] >= row[j+1] for j in range(3)):
                heuristic_value += sum(row) * 0.15  # Increased weight
            if all(col[j] >= col[j+1] for j in range(3)):
                heuristic_value += sum(col) * 0.15  # Increased weight
        
        # 3. Smoothness (penalize large differences between adjacent tiles)
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if i < 3:
                    smoothness -= abs(state[i,j] - state[i+1,j])
                if j < 3:
                    smoothness -= abs(state[i,j] - state[i,j+1])
        heuristic_value += smoothness * 0.1
        
        # 4. Empty cells bonus (encourage keeping space for merging)
        empty_cells = np.sum(state == 0)
        heuristic_value += empty_cells * 100
        
        # 5. Merge potential (bonus for adjacent equal tiles)
        merge_potential = 0
        for i in range(4):
            for j in range(4):
                if i < 3 and state[i,j] == state[i+1,j] and state[i,j] > 0:
                    merge_potential += state[i,j] * 2
                if j < 3 and state[i,j] == state[i,j+1] and state[i,j] > 0:
                    merge_potential += state[i,j] * 2
        heuristic_value += merge_potential * 0.1
        
        # Combine model value with heuristic bonuses
        # Scale the model value to be more comparable with heuristics
        scaled_model_value = model_value * 0.1
        return scaled_model_value + heuristic_value

    def _apply_move(self, bitboard, move):
        """Apply a move to the bitboard and add a random tile"""
        # Get the result of the move
        next_bitboard, score = super()._apply_move(bitboard, move)
        
        # Check if the move was valid (board changed)
        if not np.array_equal(bitboard.to_numpy(), next_bitboard.to_numpy()):
            # Add a random tile
            empty_cells = np.transpose(np.where(next_bitboard.to_numpy() == 0))
            if len(empty_cells) > 0:
                i, j = empty_cells[np.random.randint(len(empty_cells))]
                value = 2 if np.random.random() < 0.9 else 4
                next_bitboard.set_tile(i, j, value)
        
        return next_bitboard, score

def run_expectimax(num_episodes: int = 100, depth: int = 4):  # Increased default depth
    # Load the trained model weights
    try:
        with open("trained_model.pkl", "rb") as f:
            trained_weights = pickle.load(f)
        print("Successfully loaded trained model weights.")
    except FileNotFoundError:
        print("Error: trained_model.pkl not found. Please run train_td.py first.")
        return
    
    # Create and configure the agent
    agent = TrainedExpectimaxAgent(depth=depth, use_gpu=True)
    agent.load_model(trained_weights)
    
    # Run episodes
    game = Game2048(seed=42)
    total_score = 0
    max_tile_overall = 0
    scores = []
    
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        episode_score = 0
        
        while not done:
            action = agent.get_move(state)
            state, reward, done, info = game.step(action)
            episode_score += reward
        
        total_score += episode_score
        scores.append(episode_score)
        max_tile_overall = max(max_tile_overall, info['max_tile'])
        
        # Log progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"Score: {episode_score:,} | Max Tile: {info['max_tile']}")
            print(f"Average Score: {total_score/(episode+1):,.0f}")
            print("-" * 50)
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Average Score: {total_score/num_episodes:,.0f}")
    print(f"Best Score: {max(scores):,}")
    print(f"Highest Max Tile: {max_tile_overall}")
    print(f"Average Max Tile: {sum(info['max_tile'] for info in game.info_history)/num_episodes:.1f}")

if __name__ == "__main__":
    print("Starting Expectimax with trained model...")
    run_expectimax(num_episodes=100, depth=4)  # Increased depth to 4 