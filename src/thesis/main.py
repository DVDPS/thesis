#!/usr/bin/env python
"""
Main entry point for the 2048 RL training system.
This provides a unified interface to different training approaches.
"""

import argparse
import logging
import os
import sys
import torch
import torch.optim as optim

from thesis.agents.base_agent import PPOAgent
from thesis.agents.enhanced_agent import EnhancedAgent
from thesis.environment.game2048 import Game2048, preprocess_state_onehot
from thesis.environment.improved_reward import apply_improved_reward
from thesis.training.training import train
from thesis.training.simplified_training import train_agent
from thesis.utils.enhanced_exploration import balanced_exploration
from thesis.utils.curriculum_learning import curriculum_learning
from thesis.config import set_seeds, device


def setup_logging(log_file="training.log"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="2048 Reinforcement Learning Training System"
    )
    
    # General options
    parser.add_argument("--epochs", type=int, default=2000, 
                        help="Number of epochs to train (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=96, 
                        help="Batch size for training (default: 96)")
    parser.add_argument("--lr", type=float, default=8e-4, 
                        help="Learning rate (default: 8e-4)")
    parser.add_argument("--hidden-dim", type=int, default=256, 
                        help="Hidden dimension size (default: 256)")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed (default: 42)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints", 
                        help="Directory to save models (default: checkpoints)")
    
    # Training modes
    parser.add_argument("--mode", type=str, default="enhanced", 
                        choices=["standard", "simplified", "enhanced", "balanced"],
                        help="Training mode to use (default: enhanced)")
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate model without training")
    parser.add_argument("--games", type=int, default=20, 
                        help="Number of games to play in evaluation (default: 20)")
    
    # Dynamic batch size options
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch size scheduling")
    parser.add_argument("--min-batch-size", type=int, default=16,
                        help="Minimum batch size for dynamic scheduling (default: 16)")
    
    # Exploration options
    parser.add_argument("--exploration", type=float, default=None,
                        help="Initial exploration noise (overrides default)")
    parser.add_argument("--min-exploration", type=float, default=None,
                        help="Minimum exploration noise (overrides default)")
    
    # Curriculum learning options
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning after normal training")
    parser.add_argument("--curriculum-epochs", type=int, default=500,
                        help="Number of curriculum learning epochs (default: 500)")
    
    return parser.parse_args()


def evaluate_agent(agent, env, num_games=20):
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
            
            # Print progress for significant milestone tiles
            current_max = info.get('max_tile', 0)
            if current_max in [256, 512, 1024, 2048] and current_max not in max_tiles:
                print(f"Game {game+1}: Achieved {current_max} tile!")
        
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


def main():
    """Main function"""
    # Parse args and set up environment
    args = parse_args()
    setup_logging()
    set_seeds(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply improved reward function
    EnhancedGame2048 = apply_improved_reward(Game2048)
    
    # Create environment
    env = EnhancedGame2048()
    
    # Create agent based on mode
    if args.mode == "standard":
        agent = PPOAgent(board_size=4, hidden_dim=args.hidden_dim, input_channels=16)
        log_file = "standard_training.log"
    elif args.mode == "simplified":
        from thesis.agents.simplified_agent import SimpleAgent
        agent = SimpleAgent(board_size=4, hidden_dim=args.hidden_dim, input_channels=16)
        log_file = "simplified_training.log"
    elif args.mode == "enhanced":
        agent = EnhancedAgent(board_size=4, hidden_dim=args.hidden_dim, input_channels=16)
        log_file = "enhanced_training.log"
    elif args.mode == "balanced":
        agent = EnhancedAgent(board_size=4, hidden_dim=args.hidden_dim, input_channels=16)
        log_file = "balanced_training.log"
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")
    
    # Update logging for the selected mode
    setup_logging(log_file)
    
    # Override exploration parameters if specified
    if args.exploration is not None:
        agent.exploration_noise = args.exploration
        logging.info(f"Overriding exploration noise: {agent.exploration_noise}")
    
    if args.min_exploration is not None:
        agent.min_exploration_noise = args.min_exploration
        logging.info(f"Overriding minimum exploration noise: {agent.min_exploration_noise}")
    
    # Create optimizer based on mode
    if args.mode == "standard":
        optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    elif args.mode in ["simplified", "enhanced"]:
        optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.mode == "balanced":
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=5e-6, eps=1e-5)
    
    # Log configuration details
    logging.info(f"=== 2048 {args.mode.capitalize()} Training ===")
    logging.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Device: {device}")
    
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
    
    # Training mode - select appropriate training function
    if args.mode == "standard":
        train(
            agent=agent,
            env=env,
            optimizer=optimizer,
            epochs=args.epochs,
            mini_batch_size=args.batch_size,
            start_epoch=start_epoch,
            checkpoint_dir=args.output_dir
        )
    elif args.mode == "simplified":
        train_agent(
            agent=agent,
            env=env,
            optimizer=optimizer,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            save_dir=args.output_dir,
            log_interval=10
        )
    elif args.mode == "enhanced":
        train_agent(
            agent=agent,
            env=env,
            optimizer=optimizer,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            save_dir=args.output_dir,
            log_interval=10
        )
    elif args.mode == "balanced":
        balanced_exploration(
            agent=agent,
            optimizer=optimizer,
            checkpoint_path=args.checkpoint,
            epochs=args.epochs,
            use_dynamic_batch=args.dynamic_batch,
            min_batch_size=args.min_batch_size,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    
    # Run curriculum learning if specified
    if args.curriculum:
        logging.info("Starting curriculum learning phase")
        curriculum_learning(
            agent=agent,
            optimizer=optimizer,
            epochs=args.curriculum_epochs,
            target_tiles=[256, 512, 1024],
            output_dir=os.path.join(args.output_dir, "curriculum")
        )
    
    # Run final evaluation
    logging.info("Running final evaluation")
    best_tile = evaluate_agent(agent, env, num_games=args.games)
    logging.info(f"Final best tile: {best_tile}")


if __name__ == "__main__":
    main() 