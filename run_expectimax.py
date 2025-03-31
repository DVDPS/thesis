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
        self.model.eval()
        
        try:
            checkpoint = torch.load("best_cnn_model.pth", map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Successfully loaded CNN model weights.")
        except FileNotFoundError:
            print("Error: best_cnn_model.pth not found. Please train the CNN model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        self.batch_size = 16384
        self.parallel_batches = 8
        self.warmup_steps = 100
        
        self.state_tensors = [
            torch.zeros((self.batch_size, 16, 4, 4),
                       dtype=torch.float32).to(self.device)
            for _ in range(self.parallel_batches)
        ]
        
        self.state_buffers = [[] for _ in range(self.parallel_batches)]
        self.current_buffer = 0
        
        self.temp_storage = {
            'features': torch.zeros((self.batch_size * self.parallel_batches, 512, 4, 4),
                                  dtype=torch.float32).to(self.device),
            'intermediate': torch.zeros((self.batch_size * self.parallel_batches, 256, 2, 2),
                                     dtype=torch.float32).to(self.device),
            'output': torch.zeros((self.batch_size * self.parallel_batches,),
                                dtype=torch.float32).to(self.device)
        }
        
        self.cuda_graphs = {}
        if use_gpu and torch.cuda.is_available():
             self._init_cuda_graphs()
            
        self.streams = [torch.cuda.Stream() for _ in range(self.parallel_batches)] if use_gpu and torch.cuda.is_available() else []
    
    def _init_cuda_graphs(self):
        if not torch.cuda.is_available():
            print("CUDA not available, skipping CUDA graph initialization.")
            return
            
        print("Initializing CUDA graphs for optimized processing...")
        
        sample_input = torch.zeros((self.batch_size, 16, 4, 4),
                                 dtype=torch.float32,
                                 device=self.device)
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        _ = self.model(sample_input)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    static_output_placeholder = self.model(sample_input)
        
        self.static_output = static_output_placeholder 
        self.cuda_graphs['forward'] = g
        self.static_input = sample_input
        print("CUDA graphs initialized successfully.")
    
    def preprocess_state(self, state):
        onehot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = state[i, j]
                if val > 0:
                    power = int(np.log2(val))
                    if 0 < power < 16: # Ensure power is within valid range (1 to 15 for 2 to 32768)
                        onehot[power, i, j] = 1.0
                    elif val == 1:
                         onehot[0, i, j] = 1.0

                else:
                    onehot[0, i, j] = 1.0 # Channel 0 represents empty tile
        return torch.tensor(onehot, dtype=torch.float32, device=self.device)

    def evaluate_states_batch(self, states, buffer_idx=0):
        if not states:
            return []
        
        values = []
        num_states = len(states)
        use_cuda = self.device.type == 'cuda'

        for start_idx in range(0, num_states, self.batch_size * self.parallel_batches):
            end_idx = min(start_idx + self.batch_size * self.parallel_batches, num_states)
            current_batch_states = states[start_idx:end_idx]
            
            actual_parallel_batches = min(self.parallel_batches, (len(current_batch_states) + self.batch_size -1) // self.batch_size)
            if actual_parallel_batches == 0: continue

            batch_splits = np.array_split(current_batch_states, actual_parallel_batches)
            
            active_streams = self.streams[:actual_parallel_batches] if use_cuda else [None] * actual_parallel_batches

            results_list = [None] * actual_parallel_batches 

            for i, (state_batch, stream) in enumerate(zip(batch_splits, active_streams)):
                if not state_batch.size > 0: continue

                num_in_batch = len(state_batch)
                current_state_tensor = self.state_tensors[i][:num_in_batch]

                if use_cuda:
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for j, state in enumerate(state_batch):
                            current_state_tensor[j].copy_(self.preprocess_state(state), non_blocking=True)

                        can_use_graph = (num_in_batch == self.batch_size and 'forward' in self.cuda_graphs)

                        if can_use_graph:
                            self.static_input.copy_(current_state_tensor, non_blocking=True)
                            self.cuda_graphs['forward'].replay()
                            batch_output = self.static_output.clone()
                        else:
                            with torch.no_grad():
                                with torch.amp.autocast(device_type='cuda'):
                                    batch_output = self.model(current_state_tensor)
                        results_list[i] = batch_output.squeeze().cpu().numpy()

                else: # CPU path
                     for j, state in enumerate(state_batch):
                         current_state_tensor[j].copy_(self.preprocess_state(state))
                     with torch.no_grad():
                         batch_output = self.model(current_state_tensor)
                     results_list[i] = batch_output.squeeze().cpu().numpy()

            if use_cuda:
                for stream in active_streams:
                    torch.cuda.current_stream().wait_stream(stream)

            for batch_result in results_list:
                 if batch_result is not None:
                    if batch_result.ndim == 0:
                        values.append(batch_result.item())
                    else:
                        values.extend(batch_result.tolist())
        
        return values
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        state_tensor = self.preprocess_state(state).unsqueeze(0)
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                     value = self.model(state_tensor).squeeze().float().item()
            else: # CPU path
                 value = self.model(state_tensor).squeeze().float().item()
        return value

    def get_move(self, state):
        for i, buffer in enumerate(self.state_buffers):
            if buffer:
                _ = self.evaluate_states_batch(buffer, i)
                self.state_buffers[i] = []
        
        return super().get_move(state)



def run_expectimax(num_episodes: int = 100, depth: int = 5):
    agent = CNNExpectimaxAgent(depth=depth, use_gpu=True)
    game = Game2048(seed=42)
    total_score = 0
    max_tile_overall = 0
    scores = []
    steps_per_episode = []
    max_tiles = []
    times_per_episode = []
    start_time = time.time()
    
    print(f"\nStarting {num_episodes} episodes with Expectimax depth {depth} using CNN evaluator")
    print(f"Using device: {agent.device}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = game.reset()
        done = False
        episode_score = 0
        steps = 0
        current_max_tile = 0 # Track max tile within the episode more accurately

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
                
                if steps % 50 == 0:
                    print(f"\rEpisode {episode+1}/{num_episodes} | Step {steps} | Score: {episode_score:,} | Max Tile: {int(current_max_tile)}", end="")
            else:
                 if game.is_game_over():
                     done = True
                 else:

                     print(f"\nWarning: Agent chose action {action} which resulted in no change. State:\n{state_np}")
                     valid_moves = game.get_valid_moves()
                     if not valid_moves:
                         done = True
            if not done:
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
    print("Starting Expectimax with CNN evaluator...")
    run_expectimax(num_episodes=100, depth=5)