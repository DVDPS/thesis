import argparse
import logging
import os
import sys
import torch
import torch.optim as optim
from simplified_agent import SimpleAgent
from simplified_training import train_agent
from game2048 import Game2048, compute_monotonicity
from reward_function import apply_reward_function
from config import set_seeds, device

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a 2048 RL agent with simplified approach")
    
    parser.add_argument("--epochs", type=int, default=2000, 
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, 
                        help="Hidden dimension size")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, 
                        help="Log interval")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints", 
                        help="Directory to save models")
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate model without training")
    parser.add_argument("--games", type=int, default=10, 
                        help="Number of games to play in evaluation")
    
    return parser.parse_args()

def evaluate_agent(agent, env, num_games=10):
    """Evaluate agent performance without training"""
    max_tiles = []
    total_scores = []
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_score = 0
        
        # Play one game
        while not done:
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Process state
            from game2048 import preprocess_state_onehot
            state_proc = preprocess_state_onehot(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                # Apply action mask for valid moves
                action_mask = torch.full((1, 4), float('-inf'), device=device)
                action_mask[0, valid_moves] = 0
                
                # Get policy and value
                logits, _ = agent(state_tensor)
                logits = logits + action_mask
                
                # Take most likely action (no exploration)
                action = torch.argmax(logits, dim=1).item()
            
            # Execute action
            state, reward, done, info = env.step(action)
            game_score += info.get('merge_score', 0)
            
            # Print progress for first game
            if game == 0 and info.get('max_tile', 0) % 64 == 0:
                print(f"Current max tile: {info.get('max_tile', 0)}")
        
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

def main():
    """Main function"""
    # Parse args and set up environment
    args = parse_args()
    setup_logging()
    set_seeds(args.seed)
    
    # Apply improved reward function
    Game2048Enhanced = apply_reward_function(Game2048)
    
    # Create environment and agent
    env = Game2048Enhanced()
    agent = SimpleAgent(board_size=4, hidden_dim=args.hidden_dim, input_channels=16)
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logging.info(f"Resuming from epoch {start_epoch}")
    
    # Evaluation mode
    if args.evaluate:
        logging.info("Running evaluation")
        best_tile = evaluate_agent(agent, env, num_games=args.games)
        logging.info(f"Evaluation complete. Best tile: {best_tile}")
        return
    
    # Training mode
    logging.info("Starting training")
    train_agent(
        agent=agent,
        env=env,
        optimizer=optimizer,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        save_dir=args.output_dir,
        log_interval=args.log_interval
    )
    
    # Run final evaluation
    logging.info("Running final evaluation")
    best_tile = evaluate_agent(agent, env, num_games=args.games)
    logging.info(f"Final best tile: {best_tile}")

if __name__ == "__main__":
    main()