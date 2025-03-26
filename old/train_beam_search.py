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
from tqdm import tqdm
from .environment.game2048 import Game2048
from .agents.beam_search_agent import BeamSearchAgent
from .config import set_seeds

def evaluate_beam_search(
    agent: BeamSearchAgent,
    num_games: int = 100,
    max_steps: int = 1000
) -> dict:
    """
    Evaluate the beam search agent over multiple games.
    
    Args:
        agent: Beam search agent to evaluate
        num_games: Number of games to play
        max_steps: Maximum steps per game
        
    Returns:
        Dictionary with evaluation metrics
    """
    env = Game2048()
    scores = []
    max_tiles = []
    game_lengths = []
    
    for game in tqdm(range(num_games), desc="Evaluating games"):
        state = env.reset()
        done = False
        game_score = 0
        game_length = 0
        game_max_tile = 0
        
        while not done and game_length < max_steps:
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Get action from beam search
            action = agent.get_action(state, valid_moves)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Update metrics
            game_score += reward
            game_length += 1
            game_max_tile = max(game_max_tile, np.max(next_state))
            
            # Update state
            state = next_state
        
        scores.append(game_score)
        max_tiles.append(game_max_tile)
        game_lengths.append(game_length)
    
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
        max_steps=args.max_steps
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