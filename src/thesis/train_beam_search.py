#!/usr/bin/env python
"""
Training script for Beam Search agent on 2048 game.
Evaluates the performance of the beam search algorithm with human-like heuristics.
"""

import numpy as np
import logging
import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from .environment.game2048 import Game2048
from .agents.beam_search_agent import BeamSearchAgent
from .config import set_seeds

def evaluate_beam_search(
    agent: BeamSearchAgent,
    num_games: int = 100,
    max_steps: int = 1000,
    batch_size: int = 32
) -> dict:
    """
    Evaluate the beam search agent over multiple games using batched processing.
    
    Args:
        agent: Beam search agent to evaluate
        num_games: Number of games to play
        max_steps: Maximum steps per game
        batch_size: Number of games to process in parallel
        
    Returns:
        Dictionary with evaluation metrics
    """
    env = Game2048()
    scores = []
    max_tiles = []
    game_lengths = []
    
    # Process games in batches
    for batch_start in tqdm(range(0, num_games, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_games)
        batch_size_actual = batch_end - batch_start
        
        # Initialize batch of games
        batch_states = [env.reset() for _ in range(batch_size_actual)]
        batch_dones = [False] * batch_size_actual
        batch_scores = [0] * batch_size_actual
        batch_lengths = [0] * batch_size_actual
        batch_max_tiles = [0] * batch_size_actual
        
        # Run batch until all games are done
        while not all(batch_dones):
            # Get valid moves for each game
            batch_valid_moves = [env.get_possible_moves() for _ in range(batch_size_actual)]
            
            # Get actions for all games in batch
            batch_actions = []
            for i in range(batch_size_actual):
                if not batch_dones[i]:
                    action = agent.get_action(batch_states[i], batch_valid_moves[i])
                    batch_actions.append(action)
                else:
                    batch_actions.append(0)  # Dummy action for done games
            
            # Execute actions for all games
            for i in range(batch_size_actual):
                if not batch_dones[i]:
                    next_state, reward, done, _ = env.step(batch_actions[i])
                    
                    # Update metrics
                    batch_scores[i] += reward
                    batch_lengths[i] += 1
                    batch_max_tiles[i] = max(batch_max_tiles[i], np.max(next_state))
                    
                    # Update state and done flag
                    batch_states[i] = next_state
                    batch_dones[i] = done or batch_lengths[i] >= max_steps
        
        # Add batch results to overall metrics
        scores.extend(batch_scores)
        max_tiles.extend(batch_max_tiles)
        game_lengths.extend(batch_lengths)
    
    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_max_tile': np.mean(max_tiles),
        'std_max_tile': np.std(max_tiles),
        'avg_length': np.mean(game_lengths),
        'std_length': np.std(game_lengths),
        'max_tile_reached': np.max(max_tiles),
        'scores': scores,
        'max_tiles': max_tiles,
        'game_lengths': game_lengths
    }

def plot_evaluation_results(results: dict, output_dir: str):
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(2, 2, 1)
    plt.plot(results['scores'])
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Game Scores')
    
    # Plot max tiles
    plt.subplot(2, 2, 2)
    plt.plot(results['max_tiles'])
    plt.xlabel('Game')
    plt.ylabel('Max Tile')
    plt.title('Max Tiles Reached')
    
    # Plot game lengths
    plt.subplot(2, 2, 3)
    plt.plot(results['game_lengths'])
    plt.xlabel('Game')
    plt.ylabel('Steps')
    plt.title('Game Lengths')
    
    # Plot max tile distribution
    plt.subplot(2, 2, 4)
    unique_tiles, counts = np.unique(results['max_tiles'], return_counts=True)
    plt.bar([str(v) for v in unique_tiles], counts)
    plt.xlabel('Tile Value')
    plt.ylabel('Count')
    plt.title('Max Tile Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beam_search_results.png'))
    plt.close()

def main():
    """Main function to run the beam search evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Beam Search agent for 2048 game")
    
    # Evaluation parameters
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per game")
    parser.add_argument("--beam-width", type=int, default=10, help="Beam width for search")
    parser.add_argument("--search-depth", type=int, default=20, help="Search depth")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for parallel processing")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="beam_search_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    set_seeds(args.seed)
    
    # Log device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    
    # Create and evaluate agent
    agent = BeamSearchAgent(
        board_size=4,
        beam_width=args.beam_width,
        search_depth=args.search_depth
    )
    
    # Run evaluation
    start_time = time.time()
    results = evaluate_beam_search(
        agent,
        num_games=args.num_games,
        max_steps=args.max_steps,
        batch_size=args.batch_size
    )
    total_time = time.time() - start_time
    
    # Log results
    logging.info(f"Evaluation completed in {total_time:.2f} seconds")
    logging.info(f"Average score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
    logging.info(f"Average max tile: {results['avg_max_tile']:.2f} ± {results['std_max_tile']:.2f}")
    logging.info(f"Average game length: {results['avg_length']:.2f} ± {results['std_length']:.2f}")
    logging.info(f"Maximum tile reached: {results['max_tile_reached']}")
    
    # Plot results
    plot_evaluation_results(results, args.output_dir)
    
    # Save results to file
    np.save(os.path.join(args.output_dir, 'evaluation_results.npy'), results)

if __name__ == "__main__":
    main() 