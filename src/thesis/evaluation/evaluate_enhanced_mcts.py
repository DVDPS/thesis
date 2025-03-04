"""
Evaluation script for testing the enhanced MCTS implementation with different parameters.
This script allows testing various configurations of the enhanced MCTS agent to find optimal settings.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from thesis.environment.game2048 import Game2048
from thesis.agents.dqn_agent import DQNAgent
from thesis.utils.mcts.mcts_agent_wrapper import MCTSAgentWrapper
from thesis.utils.mcts.mcts import MCTS, C_PUCT, DIRICHLET_ALPHA, MAX_DEPTH, TILE_BONUSES
from thesis.config import device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/enhanced_mcts_evaluation.log')
    ]
)

def evaluate_agent(agent, num_games=100, max_steps=10000, render=False, verbose=False):
    """
    Evaluate an agent by playing multiple games and recording statistics.
    
    Args:
        agent: The agent to evaluate
        num_games: Number of games to play
        max_steps: Maximum steps per game
        render: Whether to render the game
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with evaluation results
    """
    env = Game2048()
    
    scores = []
    max_tiles = []
    game_lengths = []
    win_rate = 0
    
    start_time = time.time()
    
    for game in tqdm(range(num_games), desc="Evaluating"):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            steps += 1
            
            if render and game == 0:  # Only render the first game
                env.render()
                time.sleep(0.1)
                
        scores.append(env.score)
        max_tiles.append(np.max(env.board))
        game_lengths.append(steps)
        
        if np.max(env.board) >= 2048:
            win_rate += 1
            
        if verbose and (game + 1) % 10 == 0:
            logging.info(f"Game {game+1}/{num_games}: Score={env.score}, Max Tile={np.max(env.board)}, Steps={steps}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    avg_game_length = np.mean(game_lengths)
    win_rate = (win_rate / num_games) * 100
    
    # Calculate tile distribution
    tile_counts = {}
    for tile in max_tiles:
        if tile not in tile_counts:
            tile_counts[tile] = 0
        tile_counts[tile] += 1
    
    # Convert to percentages
    tile_distribution = {tile: (count / num_games) * 100 for tile, count in tile_counts.items()}
    
    # Sort by tile value
    tile_distribution = {k: v for k, v in sorted(tile_distribution.items())}
    
    results = {
        "avg_score": avg_score,
        "avg_max_tile": avg_max_tile,
        "avg_game_length": avg_game_length,
        "win_rate": win_rate,
        "tile_distribution": tile_distribution,
        "total_time": total_time,
        "time_per_game": total_time / num_games
    }
    
    return results

def print_results(results, config=None):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation results
        config: Optional configuration dictionary
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if config:
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nPerformance Metrics:")
    print(f"  Average Score: {results['avg_score']:.2f}")
    print(f"  Average Max Tile: {results['avg_max_tile']:.2f}")
    print(f"  Average Game Length: {results['avg_game_length']:.2f} steps")
    print(f"  Win Rate (2048 tile): {results['win_rate']:.2f}%")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    print(f"  Time per Game: {results['time_per_game']:.2f} seconds")
    
    print("\nTile Distribution:")
    for tile, percentage in results['tile_distribution'].items():
        print(f"  {tile}: {percentage:.2f}%")
    
    print("="*50 + "\n")

def save_results(results, config, filename=None):
    """
    Save evaluation results to a CSV file.
    
    Args:
        results: Dictionary with evaluation results
        config: Configuration dictionary
        filename: Optional filename to save to
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/enhanced_mcts_evaluation_{timestamp}.csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Flatten the tile distribution
    flat_results = results.copy()
    del flat_results['tile_distribution']
    
    for tile, percentage in results['tile_distribution'].items():
        flat_results[f"tile_{tile}"] = percentage
    
    # Combine config and results
    data = {**config, **flat_results}
    
    # Convert to DataFrame and save
    df = pd.DataFrame([data])
    df.to_csv(filename, index=False)
    
    logging.info(f"Results saved to {filename}")
    
    return filename

def run_parameter_sweep(base_agent, param_grid, num_games=10, save_dir="results/parameter_sweep"):
    """
    Run a parameter sweep to find optimal MCTS parameters.
    
    Args:
        base_agent: Base agent to wrap with MCTS
        param_grid: Dictionary of parameters to sweep
        num_games: Number of games per configuration
        save_dir: Directory to save results
        
    Returns:
        DataFrame with all results
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    all_results = []
    
    for i, combination in enumerate(param_combinations):
        config = {param_names[j]: combination[j] for j in range(len(param_names))}
        
        logging.info(f"Evaluating configuration {i+1}/{len(param_combinations)}: {config}")
        
        # Create MCTS agent with this configuration
        mcts_agent = MCTSAgentWrapper(
            base_agent,
            num_simulations=config.get('num_simulations', 50),
            temperature=config.get('temperature', 0.5),
            adaptive_simulations=config.get('adaptive_simulations', True)
        )
        
        # Set any other MCTS parameters
        if 'c_puct' in config:
            MCTS.C_PUCT = config['c_puct']
        if 'dirichlet_alpha' in config:
            MCTS.DIRICHLET_ALPHA = config['dirichlet_alpha']
        if 'max_depth' in config:
            MCTS.MAX_DEPTH = config['max_depth']
        
        # Evaluate the agent
        results = evaluate_agent(mcts_agent, num_games=num_games)
        
        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/config_{i+1}_{timestamp}.csv"
        save_results(results, config, filename)
        
        # Add to all results
        flat_results = results.copy()
        del flat_results['tile_distribution']
        
        for tile, percentage in results['tile_distribution'].items():
            flat_results[f"tile_{tile}"] = percentage
        
        all_results.append({**config, **flat_results})
        
        # Print results
        print_results(results, config)
    
    # Combine all results and save
    df = pd.DataFrame(all_results)
    summary_file = f"{save_dir}/parameter_sweep_summary.csv"
    df.to_csv(summary_file, index=False)
    
    logging.info(f"Parameter sweep complete. Summary saved to {summary_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced MCTS agent")
    parser.add_argument("--model_path", type=str, default="models/dqn_agent.pt", help="Path to the base agent model")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of MCTS simulations")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for action selection")
    parser.add_argument("--c_puct", type=float, default=C_PUCT, help="Exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=DIRICHLET_ALPHA, help="Dirichlet noise alpha")
    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH, help="Maximum search depth")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive simulation count")
    parser.add_argument("--render", action="store_true", help="Render the first game")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--parameter_sweep", action="store_true", help="Run parameter sweep")
    
    args = parser.parse_args()
    
    # Load the base agent
    logging.info(f"Loading base agent from {args.model_path}")
    base_agent = DQNAgent()
    base_agent.load(args.model_path)
    
    if args.parameter_sweep:
        # Define parameter grid for sweep
        param_grid = {
            'num_simulations': [50, 100, 200],
            'temperature': [0.3, 0.5, 0.7],
            'c_puct': [1.0, 2.0, 3.0],
            'adaptive_simulations': [True, False]
        }
        
        # Run parameter sweep
        results_df = run_parameter_sweep(base_agent, param_grid, num_games=args.num_games)
        
        # Find best configuration
        best_config = results_df.loc[results_df['avg_max_tile'].idxmax()]
        logging.info(f"Best configuration by avg_max_tile: {best_config.to_dict()}")
        
        best_config_score = results_df.loc[results_df['avg_score'].idxmax()]
        logging.info(f"Best configuration by avg_score: {best_config_score.to_dict()}")
        
    else:
        # Set MCTS parameters
        MCTS.C_PUCT = args.c_puct
        MCTS.DIRICHLET_ALPHA = args.dirichlet_alpha
        MCTS.MAX_DEPTH = args.max_depth
        
        # Create MCTS agent
        mcts_agent = MCTSAgentWrapper(
            base_agent,
            num_simulations=args.num_simulations,
            temperature=args.temperature,
            adaptive_simulations=args.adaptive
        )
        
        # Create config dictionary
        config = {
            "model_path": args.model_path,
            "num_simulations": args.num_simulations,
            "temperature": args.temperature,
            "c_puct": args.c_puct,
            "dirichlet_alpha": args.dirichlet_alpha,
            "max_depth": args.max_depth,
            "adaptive_simulations": args.adaptive,
            "tile_bonuses": str(TILE_BONUSES)
        }
        
        # Evaluate the agent
        logging.info(f"Evaluating MCTS agent with {args.num_games} games")
        results = evaluate_agent(mcts_agent, num_games=args.num_games, render=args.render, verbose=args.verbose)
        
        # Print and save results
        print_results(results, config)
        save_results(results, config)

if __name__ == "__main__":
    main() 