"""
Enhanced MCTS evaluation script.
This module provides functionality to evaluate the enhanced MCTS implementation.
"""

import torch
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt
from ...environment.game2048 import Game2048, preprocess_state_onehot
from ...config import device
from ...agents.dqn_agent import DQNAgent
from ..mcts import wrap_agent_with_mcts
from .evaluation import evaluate_agent

def evaluate_with_different_simulation_counts(agent, simulation_counts, num_games=3):
    """
    Evaluate the agent with different MCTS simulation counts.
    
    Args:
        agent: The agent to evaluate
        simulation_counts: List of simulation counts to try
        num_games: Number of games to play for each simulation count
        
    Returns:
        Dictionary with evaluation results for each simulation count
    """
    results = {}
    
    for sim_count in simulation_counts:
        logging.info(f"\nEvaluating with {sim_count} simulations...")
        mcts_agent = wrap_agent_with_mcts(agent, num_simulations=sim_count, temperature=0.5)
        results[sim_count] = evaluate_agent(mcts_agent, num_games=num_games)
        
    return results

def plot_simulation_results(results, output_path="mcts_simulation_comparison.png"):
    """
    Plot the results of different simulation counts.
    
    Args:
        results: Dictionary with evaluation results for each simulation count
        output_path: Path to save the plot
    """
    sim_counts = sorted(results.keys())
    avg_max_tiles = [results[sim]['avg_max_tile'] for sim in sim_counts]
    avg_scores = [results[sim]['avg_score'] for sim in sim_counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average max tile
    ax1.plot(sim_counts, avg_max_tiles, 'o-', linewidth=2)
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Average Max Tile')
    ax1.set_title('Effect of Simulation Count on Max Tile')
    ax1.grid(True)
    
    # Plot average score
    ax2.plot(sim_counts, avg_scores, 'o-', linewidth=2, color='orange')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Effect of Simulation Count on Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")

def main(checkpoint_path, num_games=5, mcts_simulations=200, run_comparison=False):
    """
    Main function to run the enhanced MCTS evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations
        run_comparison: Whether to run comparison with different simulation counts
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("enhanced_mcts_evaluation.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load model
    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = DQNAgent(hidden_dim=512)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Create MCTS-enhanced version
    logging.info(f"Creating enhanced MCTS agent with {mcts_simulations} simulations")
    mcts_agent = wrap_agent_with_mcts(agent, num_simulations=mcts_simulations, temperature=0.5)
    
    # Evaluate regular agent
    logging.info("\nEvaluating regular agent...")
    regular_results = evaluate_agent(agent, num_games=num_games)
    
    # Evaluate MCTS agent
    logging.info("\nEvaluating enhanced MCTS agent...")
    mcts_results = evaluate_agent(mcts_agent, num_games=num_games)
    
    # Print comparison
    logging.info("\n" + "=" * 50)
    logging.info("COMPARISON RESULTS:")
    logging.info(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, Enhanced MCTS = {mcts_results['avg_max_tile']:.1f}")
    logging.info(f"Average Score: Regular = {regular_results['avg_score']:.1f}, Enhanced MCTS = {mcts_results['avg_score']:.1f}")
    logging.info(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, Enhanced MCTS = {mcts_results['max_tile_reached']}")
    
    logging.info("\nRegular Agent Tile Distribution:")
    for tile, count in sorted(regular_results['tile_counts'].items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
        
    logging.info("\nEnhanced MCTS Agent Tile Distribution:")
    for tile, count in sorted(mcts_results['tile_counts'].items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
    
    # Run comparison with different simulation counts if requested
    if run_comparison:
        logging.info("\n" + "=" * 50)
        logging.info("RUNNING SIMULATION COUNT COMPARISON")
        sim_counts = [50, 100, 200, 300, 400]
        comparison_results = evaluate_with_different_simulation_counts(agent, sim_counts, num_games=2)
        
        # Print comparison results
        logging.info("\nSIMULATION COUNT COMPARISON RESULTS:")
        for sim_count, result in comparison_results.items():
            logging.info(f"Simulations: {sim_count}, Avg Max Tile: {result['avg_max_tile']:.1f}, Avg Score: {result['avg_score']:.1f}, Best Tile: {result['max_tile_reached']}")
        
        # Plot results
        plot_simulation_results(comparison_results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate enhanced MCTS for 2048 agents")
    parser.add_argument("--checkpoint", type=str, default="dqn_results/final_model.pt", help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=200, help="Number of MCTS simulations")
    parser.add_argument("--compare", action="store_true", help="Run comparison with different simulation counts")
    
    args = parser.parse_args()
    main(args.checkpoint, args.games, args.simulations, args.compare) 