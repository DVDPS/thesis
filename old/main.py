import argparse
import logging
import os
import sys
import torch
import torch.optim as optim

# Import local modules
from agent import EnhancedPPOAgent
from game2048 import Game2048
from training import improved_train
from config import device

# Import performance optimizations
from performance_optimization import (
    optimize_gpu_utilization,
    fast_preprocess_state_onehot,
    FastPrioritizedReplayBuffer,
    compute_monotonicity_fast
)

def parse_args():
    """Parse command line arguments with sensible defaults"""
    parser = argparse.ArgumentParser(description="Train a 2048 RL agent")
    
    # Basic parameters
    parser.add_argument("--epochs", type=int, default=3000, 
                        help="Number of epochs to train")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints", 
                        help="Directory to save models")
    
    # Advanced training options
    parser.add_argument("--use-replay", action="store_true", default=True,
                        help="Use prioritized experience replay")
    parser.add_argument("--use-curriculum", action="store_true", default=True,
                        help="Use curriculum learning")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--replay-size", type=int, default=50000,
                        help="Size of replay buffer")
    
    # Exploration parameters
    parser.add_argument("--init-noise", type=float, default=1.5,
                        help="Initial exploration noise")
    parser.add_argument("--min-noise", type=float, default=0.1,
                        help="Minimum exploration noise")
    
    # Evaluation and visualization
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model without training")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize agent play")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play in evaluation")
    
    # Performance optimizations
    parser.add_argument("--optimize-gpu", action="store_true", default=True,
                        help="Apply GPU optimizations")
    parser.add_argument("--use-fast-preprocessing", action="store_true", default=True,
                        help="Use optimized state preprocessing")
    parser.add_argument("--use-optimized-buffer", action="store_true", default=True,
                        help="Use optimized replay buffer implementation")
    
    return parser.parse_args()

def setup_logging():
    """Configure logging with clear formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_agent(args):
    """Create agent with appropriate configuration"""
    agent = EnhancedPPOAgent(
        board_size=4,
        hidden_dim=256,
        input_channels=16,
        Vinit=320000.0
    )
    
    # Set exploration parameters
    agent.exploration_noise = args.init_noise
    agent.min_exploration_noise = args.min_noise
    
    return agent

def load_checkpoint(agent, optimizer, checkpoint_path):
    """Load model checkpoint if it exists"""
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('-inf')
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_reward = checkpoint.get('running_reward', float('-inf'))
    max_tile = checkpoint.get('max_tile', 0)
    
    logging.info(f"Resuming from epoch {start_epoch} with best reward: {best_reward:.1f}")
    logging.info(f"Previously achieved max tile: {max_tile}")
    
    return start_epoch, best_reward

def evaluate_agent(agent, env, num_games=10, visualize=False, use_fast_preprocessing=True):
    """Evaluate agent performance without training"""
    max_tiles = []
    total_scores = []
    
    # Pre-allocate buffer for state preprocessing if using fast method
    state_buffer = None
    if use_fast_preprocessing:
        import numpy as np
        state_buffer = np.zeros((16, 4, 4), dtype=np.float32)
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_score = 0
        
        # Play one game
        while not done:
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Process state with optimized preprocessing
            if use_fast_preprocessing:
                state_proc = fast_preprocess_state_onehot(state, state_buffer)
            else:
                from game2048 import preprocess_state_onehot
                state_proc = preprocess_state_onehot(state)
                
            state_tensor = torch.tensor(state_proc, dtype=torch.float, 
                                       device=device).unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                # Apply action mask for valid moves
                action_mask = torch.full((1, 4), float('-inf'), 
                                        device=device)
                action_mask[0, valid_moves] = 0
                
                # Get policy and value
                logits, _ = agent(state_tensor)
                logits = logits + action_mask
                
                # Take most likely action (no exploration)
                action = torch.argmax(logits, dim=1).item()
            
            # Execute action
            state, reward, done, info = env.step(action)
            game_score += info.get('merge_score', 0)
            
            # Visualize board if requested
            if visualize:
                print("\n" + "=" * 20)
                print(f"Move: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
                print_board(state)
                print(f"Score: {game_score}")
        
        # Record results
        max_tile = info.get('max_tile', 0)
        max_tiles.append(max_tile)
        total_scores.append(game_score)
        
        logging.info(f"Game {game+1}: Score = {game_score}, Max Tile = {max_tile}")
    
    # Print summary
    logging.info("=" * 40)
    logging.info(f"Evaluation over {num_games} games:")
    logging.info(f"Average Score: {sum(total_scores) / len(total_scores):.1f}")
    logging.info(f"Average Max Tile: {sum(max_tiles) / len(max_tiles):.1f}")
    logging.info(f"Best Max Tile: {max(max_tiles)}")
    
    # Count occurrences of each tile
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    logging.info("Tile distribution:")
    for tile, count in sorted(tile_counts.items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
        
    return max(max_tiles)

def print_board(board):
    """Pretty print the 2048 board"""
    print("-" * 21)
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("    |", end="")
            else:
                print(f"{cell:4d}|", end="")
        print("\n" + "-" * 21)

def create_optimized_replay_buffer(capacity, alpha=0.6):
    """Create an optimized replay buffer"""
    return FastPrioritizedReplayBuffer(capacity=capacity, alpha=alpha)

def main():
    """Main function"""
    # Parse arguments and set up environment
    args = parse_args()
    setup_logging()
    set_seeds(42)
    
    # Apply GPU optimizations if requested
    if args.optimize_gpu and torch.cuda.is_available():
        logging.info("Applying GPU optimizations")
        optimize_gpu_utilization()
    
    # Create environment and agent
    env = Game2048()
    agent = create_agent(args)
    
    # Create optimizer with improved defaults
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        eps=1e-5
    )
    
    # Handle checkpoints
    start_epoch = 0
    best_reward = float('-inf')
    if args.checkpoint:
        start_epoch, best_reward = load_checkpoint(agent, optimizer, args.checkpoint)
    
    # Evaluation mode
    if args.evaluate:
        logging.info("Running evaluation")
        evaluate_agent(agent, env, 
                      num_games=args.games, 
                      visualize=args.visualize,
                      use_fast_preprocessing=args.use_fast_preprocessing)
        return
    
    # Training mode
    logging.info("Starting training")
    logging.info(f"Training for {args.epochs} epochs")
    logging.info(f"Using replay buffer: {args.use_replay}")
    logging.info(f"Using optimized replay buffer: {args.use_optimized_buffer}")
    logging.info(f"Using curriculum learning: {args.use_curriculum}")
    logging.info(f"Using fast preprocessing: {args.use_fast_preprocessing}")
    
    # Create optimized replay buffer if requested
    optimized_buffer = None
    if args.use_optimized_buffer and args.use_replay:
        optimized_buffer = create_optimized_replay_buffer(args.replay_size)
        logging.info(f"Created optimized replay buffer with capacity {args.replay_size}")
    
    # Start training with optimization flags
    training_stats = improved_train(
        agent=agent,
        env=env,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_size,
        prioritized_replay=args.use_replay,
        curriculum_phases=args.use_curriculum,
        optimized_buffer=optimized_buffer if args.use_optimized_buffer else None,
        use_fast_preprocessing=args.use_fast_preprocessing
    )
    
    # Print final results
    logging.info("Training complete!")
    logging.info(f"Best reward achieved: {training_stats['best_reward']:.1f}")
    logging.info(f"Best max tile achieved: {training_stats['best_max_tile']}")
    
    # Run evaluation to confirm performance
    logging.info("Running final evaluation")
    best_tile = evaluate_agent(agent, env, 
                              num_games=args.games,
                              use_fast_preprocessing=args.use_fast_preprocessing)
    logging.info(f"Final best tile: {best_tile}")

if __name__ == "__main__":
    main() 