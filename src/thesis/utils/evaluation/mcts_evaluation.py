"""
Evaluation script for MCTS enhanced agents.
This module provides functionality to compare MCTS-enhanced agents with regular agents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from ...environment.game2048 import Game2048, preprocess_state_onehot
from ...config import device
from ...agents.dqn_agent import DQNAgent
from ..mcts.mcts_agent_wrapper import wrap_agent_with_mcts
from ..visualization.game_analysis import analyze_game_trajectory
from ..visualization.mcts_visualization import visualize_mcts_search
from .evaluation import evaluate_agent

def compare_agents(regular_agent, mcts_agent, num_games=5, max_steps=1000, save_trajectories=False):
    """
    Compare performance between a regular agent and an MCTS-enhanced version.
    
    Args:
        regular_agent: Regular agent without MCTS
        mcts_agent: MCTS-enhanced agent
        num_games: Number of games to play with each agent
        max_steps: Maximum steps per game
        save_trajectories: Whether to save full game trajectories
        
    Returns:
        Dictionary with comparison results
    """
    logging.info("Evaluating regular agent...")
    regular_results = evaluate_agent(regular_agent, num_games=num_games, max_steps=max_steps, save_trajectories=save_trajectories)
    
    logging.info("\nEvaluating MCTS agent...")
    mcts_results = evaluate_agent(mcts_agent, num_games=num_games, max_steps=max_steps, save_trajectories=save_trajectories)
    
    # Print comparison
    logging.info("\n" + "=" * 50)
    logging.info("COMPARISON RESULTS:")
    logging.info(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, MCTS = {mcts_results['avg_max_tile']:.1f}")
    logging.info(f"Average Score: Regular = {regular_results['avg_score']:.1f}, MCTS = {mcts_results['avg_score']:.1f}")
    logging.info(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, MCTS = {mcts_results['max_tile_reached']}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Max tile histogram
    plt.subplot(2, 3, 1)
    bins = np.arange(0, 2100, 100)
    plt.hist(regular_results['max_tiles'], bins=bins, alpha=0.5, label='Regular Agent')
    plt.hist(mcts_results['max_tiles'], bins=bins, alpha=0.5, label='MCTS Agent')
    plt.xlabel('Max Tile')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Max Tile Distribution')
    
    # Score comparison
    plt.subplot(2, 3, 2)
    plt.boxplot([regular_results['scores'], mcts_results['scores']], labels=['Regular', 'MCTS'])
    plt.ylabel('Score')
    plt.title('Score Comparison')
    
    # Max tiles for each game
    plt.subplot(2, 3, 3)
    plt.plot(regular_results['max_tiles'], 'bo-', label='Regular Agent')
    plt.plot(mcts_results['max_tiles'], 'ro-', label='MCTS Agent')
    plt.xlabel('Game')
    plt.ylabel('Max Tile')
    plt.legend()
    plt.title('Max Tile by Game')
    
    # Tile progression comparison
    plt.subplot(2, 3, 4)
    
    # Get common tile values
    all_tiles = set(regular_results['tile_progression'].keys()) | set(mcts_results['tile_progression'].keys())
    sorted_tiles = sorted(all_tiles)
    
    reg_steps = [regular_results['tile_progression'].get(t, float('nan')) for t in sorted_tiles]
    mcts_steps = [mcts_results['tile_progression'].get(t, float('nan')) for t in sorted_tiles]
    
    x = np.arange(len(sorted_tiles))
    width = 0.35
    
    plt.bar(x - width/2, reg_steps, width, label='Regular Agent')
    plt.bar(x + width/2, mcts_steps, width, label='MCTS Agent')
    plt.xlabel('Tile Value')
    plt.ylabel('Avg Steps to Achieve')
    plt.title('Tile Achievement Speed')
    plt.xticks(x, [str(t) for t in sorted_tiles])
    plt.legend()
    
    # Average max tile by simulation count
    plt.subplot(2, 3, 5)
    
    # If we have different simulation counts data
    sim_counts = [5, 10, 25, 50, 100, 200]
    avg_tiles = []
    avg_scores = []
    
    if hasattr(mcts_agent, 'num_simulations'):
        orig_sim_count = mcts_agent.num_simulations
        
        for sim_count in sim_counts:
            if sim_count <= 200:  # Skip higher counts for faster testing
                logging.info(f"Testing with {sim_count} simulations...")
                mcts_agent.set_num_simulations(sim_count)
                results = evaluate_agent(mcts_agent, num_games=2, max_steps=max_steps)
                avg_tiles.append(results['avg_max_tile'])
                avg_scores.append(results['avg_score'])
        
        # Restore original simulation count
        mcts_agent.set_num_simulations(orig_sim_count)
        
        plt.plot(sim_counts[:len(avg_tiles)], avg_tiles, 'ro-')
        plt.xlabel('MCTS Simulations')
        plt.ylabel('Avg Max Tile')
        plt.title('Performance vs Simulation Count')
    else:
        plt.text(0.5, 0.5, 'Not applicable', ha='center', va='center')
    
    # Temperature comparison if available
    plt.subplot(2, 3, 6)
    
    if hasattr(mcts_agent, 'set_temperature'):
        orig_temp = mcts_agent.temperature
        
        temps = [0.1, 0.5, 1.0, 1.5, 2.0]
        temp_tiles = []
        
        for temp in temps:
            logging.info(f"Testing with temperature {temp}...")
            mcts_agent.set_temperature(temp)
            results = evaluate_agent(mcts_agent, num_games=2, max_steps=max_steps)
            temp_tiles.append(results['avg_max_tile'])
        
        # Restore original temperature
        mcts_agent.set_temperature(orig_temp)
        
        plt.plot(temps, temp_tiles, 'go-')
        plt.xlabel('Temperature')
        plt.ylabel('Avg Max Tile')
        plt.title('Performance vs Temperature')
    else:
        plt.text(0.5, 0.5, 'Not applicable', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png')
    plt.close()
    
    return {
        'regular': regular_results,
        'mcts': mcts_results
    }

def main(checkpoint_path, num_games=5, mcts_simulations=100):
    """
    Main function to run the MCTS evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations
    """
    # Create output directory
    os.makedirs("mcts_evaluation", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mcts_evaluation/evaluation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Load model
    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent = DQNAgent()
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Create MCTS-enhanced version
    logging.info(f"Creating MCTS agent with {mcts_simulations} simulations")
    mcts_agent = wrap_agent_with_mcts(agent, num_simulations=mcts_simulations, adaptive_simulations=True)
    
    # Compare both agents
    compare_results = compare_agents(agent, mcts_agent, num_games=num_games, save_trajectories=True)
    
    # Analyze a specific position
    env = Game2048()
    state = env.reset()
    
    # Play until we reach a state with a high-value tile
    steps = 0
    max_steps = 100
    target_value = 64  # Target a position with at least a 64 tile
    
    while np.max(state) < target_value and steps < max_steps:
        valid_moves = env.get_possible_moves()
        if not valid_moves:
            break
            
        # Use regular agent to generate the position
        state_proc = preprocess_state_onehot(state)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        
        with torch.no_grad():
            q_values, _ = agent(state_tensor)
            
        # Apply action mask
        action_mask = torch.full((1, 4), float('-inf'), device=device)
        action_mask[0, valid_moves] = 0
        masked_q_values = q_values + action_mask
        
        # Take best action
        action = torch.argmax(masked_q_values, dim=1).item()
        
        # Execute action
        state, _, done, _ = env.step(action)
        steps += 1
        
        if done:
            break
    
    # Analyze the position
    logging.info(f"Analyzing position after {steps} steps:")
    logging.info(state)
    
    # Visualize MCTS analysis
    visualize_mcts_search(mcts_agent, state, filename='mcts_evaluation/position_analysis.png')
    
    # Analyze a full game trajectory
    if compare_results['mcts']['trajectories']:
        # Find the game with the highest max tile
        best_game = max(compare_results['mcts']['trajectories'], key=lambda x: x['max_tile'])
        analyze_game_trajectory(best_game, filename='mcts_evaluation/best_game_analysis.png')
    
    # Log comparison summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Regular Agent - Avg Max Tile: {compare_results['regular']['avg_max_tile']:.1f}, Best: {compare_results['regular']['max_tile_reached']}")
    logging.info(f"MCTS Agent ({mcts_simulations} sims) - Avg Max Tile: {compare_results['mcts']['avg_max_tile']:.1f}, Best: {compare_results['mcts']['max_tile_reached']}")
    
    # Test with different simulation counts
    logging.info("\nTesting different simulation counts:")
    sim_counts = [50, 100, 200, 400]
    for sims in sim_counts:
        mcts_agent.set_num_simulations(sims)
        results = evaluate_agent(mcts_agent, num_games=2)
        logging.info(f"MCTS with {sims} simulations: Avg Max Tile = {results['avg_max_tile']:.1f}, Best = {results['max_tile_reached']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MCTS enhancement of 2048 agents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=100, help="Number of MCTS simulations")
    
    args = parser.parse_args()
    main(args.checkpoint, args.games, args.simulations) 