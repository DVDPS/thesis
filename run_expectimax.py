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


        self.batch_size = 8192  
        self.parallel_batches = 4 
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
        if use_gpu:
            self._init_cuda_graphs()
            
        self.streams = [torch.cuda.Stream() for _ in range(self.parallel_batches)]
    
    def _init_cuda_graphs(self):
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
                        self.model(sample_input)
        

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    self.static_output = self.model(sample_input)
        
        self.cuda_graphs['forward'] = g
        self.static_input = sample_input
        print("CUDA graphs initialized successfully.")
    
    def preprocess_state(self, state):
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
        if not states:
            return []
        
        values = []
        num_states = len(states)
        

        for start_idx in range(0, num_states, self.batch_size * self.parallel_batches):
            end_idx = min(start_idx + self.batch_size * self.parallel_batches, num_states)
            current_batch_states = states[start_idx:end_idx]
            

            batch_splits = np.array_split(current_batch_states, 
                                        min(self.parallel_batches, 
                                            len(current_batch_states)))
            
            for i, (state_batch, stream) in enumerate(zip(batch_splits, self.streams)):
                stream.wait_stream(torch.cuda.current_stream())
                
                with torch.cuda.stream(stream):
                    for j, state in enumerate(state_batch):
                        self.state_tensors[i][j].copy_(self.preprocess_state(state), non_blocking=True)
                    
                    if len(state_batch) == self.batch_size and 'forward' in self.cuda_graphs:
                        self.static_input.copy_(self.state_tensors[i][:len(state_batch)], non_blocking=True)
                        self.cuda_graphs['forward'].replay()
                        batch_output = self.static_output
                    else:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda'):
                                batch_output = self.model(self.state_tensors[i][:len(state_batch)])
                    
                    batch_output = batch_output.squeeze()
                    if batch_output.ndim == 0:
                        values.append(batch_output.item())
                    else:
                        values.extend(batch_output.cpu().numpy().tolist())
            
            torch.cuda.current_stream().wait_stream(stream)
        
        return values
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        self.state_buffers[self.current_buffer].append(state)
        
        if len(self.state_buffers[self.current_buffer]) >= self.batch_size:
            values = self.evaluate_states_batch(self.state_buffers[self.current_buffer], 
                                             self.current_buffer)
            self.state_buffers[self.current_buffer] = []
            self.current_buffer = (self.current_buffer + 1) % self.parallel_batches
            return values[0]
        
        return 0.0
    
    def get_move(self, state):
        for i, buffer in enumerate(self.state_buffers):
            if buffer:
                values = self.evaluate_states_batch(buffer, i)
                self.state_buffers[i] = []
        
        return super().get_move(state)

    def _apply_move(self, bitboard, move):
        next_bitboard, score = super()._apply_move(bitboard, move)
        
        if not np.array_equal(bitboard.to_numpy(), next_bitboard.to_numpy()):
            empty_cells = np.transpose(np.where(next_bitboard.to_numpy() == 0))
            if len(empty_cells) > 0:
                i, j = empty_cells[np.random.randint(len(empty_cells))]
                value = 2 if np.random.random() < 0.9 else 4
                next_bitboard.set_tile(i, j, value)
        
        return next_bitboard, score

def run_expectimax(num_episodes: int = 100, depth: int = 5):
    agent = CNNExpectimaxAgent(depth=depth, use_gpu=True)
    
    game = Game2048(seed=42)
    total_score = 0
    max_tile_overall = 0
    scores = []
    steps_per_episode = []
    max_tiles = []
    start_time = time.time()
    
    print(f"\nStarting {num_episodes} episodes with depth {depth}")
    print("=" * 50)
    
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
            new_board, score, changed = game._move(game.board, action)
            if changed:
                game.board = new_board
                game.score += score
                game.add_random_tile()
                episode_score += score
                state = game.board.cpu().numpy()
                steps += 1
                max_tile = max(max_tile, torch.max(game.board).item())
                
                if steps % 50 == 0:
                    print(f"\rEpisode {episode+1}/{num_episodes} | Step {steps} | Score: {episode_score:,} | Max Tile: {max_tile}", end="")
            
            done = game.is_game_over()
        
        episode_time = time.time() - episode_start_time
        total_score += episode_score
        scores.append(episode_score)
        steps_per_episode.append(steps)
        max_tiles.append(max_tile)
        max_tile_overall = max(max_tile_overall, max_tile)
        
        performance_history['scores'].append(episode_score)
        performance_history['max_tiles'].append(max_tile)
        performance_history['steps'].append(steps)
        performance_history['times'].append(episode_time)
        
        print(f"\nEpisode {episode+1}/{num_episodes} completed in {episode_time:.1f}s")
        print(f"Final Score: {episode_score:,} | Steps: {steps} | Max Tile: {max_tile}")
        print(f"Running Average Score: {total_score/(episode+1):,.0f}")
        print("-" * 50)
        
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
    run_expectimax(num_episodes=100, depth=5)