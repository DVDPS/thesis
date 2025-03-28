import pickle
import numpy as np
import torch
from agents.expectimax import ExpectimaxAgent
from agents.cnn_agent import Game2048CNN
from src.thesis.environment.game2048 import Game2048
import time

class CNNExpectimaxAgent(ExpectimaxAgent):
    def __init__(self, depth=4, use_gpu=True):
        super().__init__(depth, use_gpu)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = Game2048CNN().to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load the trained CNN model
        try:
            checkpoint = torch.load("best_cnn_model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Successfully loaded CNN model weights.")
        except FileNotFoundError:
            print("Error: best_cnn_model.pth not found. Please train the CNN model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Significantly increased batch and parallel processing
        self.batch_size = 16384  # Doubled batch size
        self.parallel_batches = 8  # Doubled parallel batches
        self.warmup_steps = 100  # Steps to warmup CUDA graphs
        
        # Pre-allocate tensors for batch evaluation
        self.state_tensors = [
            torch.zeros((self.batch_size, 16, 4, 4), 
                       dtype=torch.float32).to(self.device)
            for _ in range(self.parallel_batches)
        ]
        
        # Larger buffers for states
        self.state_buffers = [[] for _ in range(self.parallel_batches)]
        self.current_buffer = 0
        
        # Pre-allocate tensors for computation
        self.temp_storage = {
            'features': torch.zeros((self.batch_size * self.parallel_batches, 512, 4, 4), 
                                  dtype=torch.float32).to(self.device),
            'intermediate': torch.zeros((self.batch_size * self.parallel_batches, 256, 2, 2), 
                                     dtype=torch.float32).to(self.device),
            'output': torch.zeros((self.batch_size * self.parallel_batches,), 
                                dtype=torch.float32).to(self.device)
        }
        
        # Initialize CUDA graphs for repeated operations
        self.cuda_graphs = {}
        if use_gpu:
            self._init_cuda_graphs()
            
        # Enable CUDA stream usage
        self.streams = [torch.cuda.Stream() for _ in range(self.parallel_batches)]
    
    def _init_cuda_graphs(self):
        """Initialize CUDA graphs for repeated operations"""
        print("Initializing CUDA graphs for optimized processing...")
        
        # Create a sample input for graph capture
        sample_input = torch.zeros((self.batch_size, 16, 4, 4), 
                                 dtype=torch.float32, 
                                 device=self.device)
        
        # Capture forward pass graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # Warmup
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        self.model(sample_input)
        
        # Capture the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    self.static_output = self.model(sample_input)
        
        self.cuda_graphs['forward'] = g
        self.static_input = sample_input
        print("CUDA graphs initialized successfully.")
    
    def preprocess_state(self, state):
        """Convert board state to one-hot representation"""
        onehot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if state[i, j] > 0:
                    power = int(np.log2(state[i, j]))
                    if power < 16:
                        onehot[power, i, j] = 1.0
                else:
                    onehot[0, i, j] = 1.0
        return torch.tensor(onehot, dtype=torch.float32, device=self.device)
    
    def evaluate_states_batch(self, states, buffer_idx=0):
        """Evaluate multiple states at once using the CNN with parallel processing"""
        if not states:
            return []
        
        values = []
        num_states = len(states)
        
        # Process multiple batches in parallel using CUDA streams
        for start_idx in range(0, num_states, self.batch_size * self.parallel_batches):
            end_idx = min(start_idx + self.batch_size * self.parallel_batches, num_states)
            current_batch_states = states[start_idx:end_idx]
            
            # Split into parallel batches
            batch_splits = np.array_split(current_batch_states, 
                                        min(self.parallel_batches, 
                                            len(current_batch_states)))
            
            # Process batches in parallel streams
            for i, (state_batch, stream) in enumerate(zip(batch_splits, self.streams)):
                stream.wait_stream(torch.cuda.current_stream())
                
                with torch.cuda.stream(stream):
                    # Prepare batch
                    for j, state in enumerate(state_batch):
                        self.state_tensors[i][j].copy_(self.preprocess_state(state), non_blocking=True)
                    
                    # Process batch using CUDA graph if possible
                    if len(state_batch) == self.batch_size and 'forward' in self.cuda_graphs:
                        self.static_input.copy_(self.state_tensors[i][:len(state_batch)], non_blocking=True)
                        self.cuda_graphs['forward'].replay()
                        batch_output = self.static_output
                    else:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda'):
                                batch_output = self.model(self.state_tensors[i][:len(state_batch)])
                    
                    # Handle both single value and batch outputs properly
                    batch_output = batch_output.squeeze()
                    if batch_output.ndim == 0:  # Single value
                        values.append(batch_output.item())
                    else:  # Batch of values
                        values.extend(batch_output.cpu().numpy().tolist())
            
            # Synchronize streams
            torch.cuda.current_stream().wait_stream(stream)
        
        return values
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        """Evaluate state using the CNN model with parallel processing"""
        self.state_buffers[self.current_buffer].append(state)
        
        if len(self.state_buffers[self.current_buffer]) >= self.batch_size:
            values = self.evaluate_states_batch(self.state_buffers[self.current_buffer], 
                                             self.current_buffer)
            self.state_buffers[self.current_buffer] = []
            self.current_buffer = (self.current_buffer + 1) % self.parallel_batches
            return values[0]
        
        return 0.0
    
    def get_move(self, state):
        """Override get_move to process any remaining states in buffers"""
        # Process any non-empty buffers
        for i, buffer in enumerate(self.state_buffers):
            if buffer:
                values = self.evaluate_states_batch(buffer, i)
                self.state_buffers[i] = []
        
        return super().get_move(state)

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

def run_expectimax(num_episodes: int = 100, depth: int = 5):  # Increased depth to 5
    # Create and configure the agent
    agent = CNNExpectimaxAgent(depth=depth, use_gpu=True)
    
    # Run episodes
    game = Game2048(seed=42)
    total_score = 0
    max_tile_overall = 0
    scores = []
    steps_per_episode = []
    max_tiles = []
    start_time = time.time()
    
    print(f"\nStarting {num_episodes} episodes with depth {depth}")
    print("=" * 50)
    
    # Track performance over time
    performance_history = {
        'scores': [],
        'max_tiles': [],
        'steps': [],
        'times': []
    }
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = game.reset()
        done = False
        episode_score = 0
        steps = 0
        max_tile = 0
        
        while not done:
            action = agent.get_move(state)
            # Use _move instead of step
            new_board, score, changed = game._move(game.board, action)
            if changed:
                game.board = new_board
                game.score += score
                game.add_random_tile()
                episode_score += score
                # Convert tensor to numpy array for state
                state = game.board.cpu().numpy()
                steps += 1
                max_tile = max(max_tile, torch.max(game.board).item())
                
                # Log every 50 steps
                if steps % 50 == 0:
                    print(f"\rEpisode {episode+1}/{num_episodes} | Step {steps} | Score: {episode_score:,} | Max Tile: {max_tile}", end="")
            
            done = game.is_game_over()
        
        # Episode completed
        episode_time = time.time() - episode_start_time
        total_score += episode_score
        scores.append(episode_score)
        steps_per_episode.append(steps)
        max_tiles.append(max_tile)
        max_tile_overall = max(max_tile_overall, max_tile)
        
        # Update performance history
        performance_history['scores'].append(episode_score)
        performance_history['max_tiles'].append(max_tile)
        performance_history['steps'].append(steps)
        performance_history['times'].append(episode_time)
        
        # Log episode completion
        print(f"\nEpisode {episode+1}/{num_episodes} completed in {episode_time:.1f}s")
        print(f"Final Score: {episode_score:,} | Steps: {steps} | Max Tile: {max_tile}")
        print(f"Running Average Score: {total_score/(episode+1):,.0f}")
        print("-" * 50)
        
        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_score = total_score/(episode+1)
            avg_steps = sum(steps_per_episode)/len(steps_per_episode)
            avg_max_tile = sum(max_tiles)/len(max_tiles)
            avg_time = sum(performance_history['times'])/len(performance_history['times'])
            
            print(f"\nProgress Report (Episode {episode+1}/{num_episodes})")
            print(f"Time Elapsed: {elapsed_time:.1f}s")
            print(f"Average Score: {avg_score:,.0f}")
            print(f"Average Steps: {avg_steps:.1f}")
            print(f"Average Max Tile: {avg_max_tile:.1f}")
            print(f"Average Time per Episode: {avg_time:.1f}s")
            print(f"Best Score So Far: {max(scores):,}")
            print(f"Highest Max Tile: {max_tile_overall}")
            print("=" * 50)
    
    # Print final statistics
    total_time = time.time() - start_time
    print("\nFinal Statistics:")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time per Episode: {total_time/num_episodes:.1f}s")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Average Score: {total_score/num_episodes:,.0f}")
    print(f"Best Score: {max(scores):,}")
    print(f"Highest Max Tile: {max_tile_overall}")
    print(f"Average Max Tile: {sum(max_tiles)/len(max_tiles):.1f}")
    print(f"Average Steps per Episode: {sum(steps_per_episode)/len(steps_per_episode):.1f}")
    print("=" * 50)

if __name__ == "__main__":
    print("Starting Expectimax with CNN model...")
    run_expectimax(num_episodes=100, depth=5)  # Increased depth to 5

# Note: The original code block for the run_expectimax function was kept as it is, but the depth parameter was changed to 5. 
# This is because the new run_expectimax function uses a different logic for processing states and evaluating them. 
# The original run_expectimax function was kept as it is, but the depth parameter was changed to 5. 
# This is because the new run_expectimax function uses a different logic for processing states and evaluating them. 