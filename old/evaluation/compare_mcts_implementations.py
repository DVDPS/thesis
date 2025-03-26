"""
Comparison script for evaluating the enhanced MCTS implementation against the original MCTS.
This script runs both implementations with the same parameters and compares their performance.
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
import matplotlib.pyplot as plt
import seaborn as sns

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
        logging.FileHandler('logs/mcts_comparison.log')
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
        "time_per_game": total_time / num_games,
        "scores": scores,
        "max_tiles": max_tiles
    }
    
    return results

def print_comparison(original_results, enhanced_results, config=None):
    """
    Print comparison of evaluation results in a formatted way.
    
    Args:
        original_results: Dictionary with original MCTS results
        enhanced_results: Dictionary with enhanced MCTS results
        config: Optional configuration dictionary
    """
    print("\n" + "="*60)
    print("MCTS IMPLEMENTATION COMPARISON")
    print("="*60)
    
    if config:
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nPerformance Metrics:")
    print(f"  Average Score:       Original: {original_results['avg_score']:.2f}   Enhanced: {enhanced_results['avg_score']:.2f}   Improvement: {(enhanced_results['avg_score'] - original_results['avg_score']) / original_results['avg_score'] * 100:.2f}%")
    print(f"  Average Max Tile:    Original: {original_results['avg_max_tile']:.2f}   Enhanced: {enhanced_results['avg_max_tile']:.2f}   Improvement: {(enhanced_results['avg_max_tile'] - original_results['avg_max_tile']) / original_results['avg_max_tile'] * 100:.2f}%")
    print(f"  Average Game Length: Original: {original_results['avg_game_length']:.2f}   Enhanced: {enhanced_results['avg_game_length']:.2f}   Difference: {enhanced_results['avg_game_length'] - original_results['avg_game_length']:.2f}")
    print(f"  Win Rate (2048 tile): Original: {original_results['win_rate']:.2f}%   Enhanced: {enhanced_results['win_rate']:.2f}%   Difference: {enhanced_results['win_rate'] - original_results['win_rate']:.2f}%")
    print(f"  Time per Game:       Original: {original_results['time_per_game']:.2f}s   Enhanced: {enhanced_results['time_per_game']:.2f}s   Difference: {enhanced_results['time_per_game'] - original_results['time_per_game']:.2f}s")
    
    print("\nTile Distribution Comparison:")
    all_tiles = sorted(set(list(original_results['tile_distribution'].keys()) + list(enhanced_results['tile_distribution'].keys())))
    
    for tile in all_tiles:
        orig_pct = original_results['tile_distribution'].get(tile, 0)
        enh_pct = enhanced_results['tile_distribution'].get(tile, 0)
        diff = enh_pct - orig_pct
        print(f"  Tile {tile}:   Original: {orig_pct:.2f}%   Enhanced: {enh_pct:.2f}%   Difference: {diff:.2f}%")
    
    print("="*60 + "\n")

def save_comparison(original_results, enhanced_results, config, filename=None):
    """
    Save comparison results to a CSV file.
    
    Args:
        original_results: Dictionary with original MCTS results
        enhanced_results: Dictionary with enhanced MCTS results
        config: Configuration dictionary
        filename: Optional filename to save to
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/mcts_comparison_{timestamp}.csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare data
    data = {
        "metric": [],
        "original": [],
        "enhanced": [],
        "difference": [],
        "percent_improvement": []
    }
    
    # Add main metrics
    metrics = ["avg_score", "avg_max_tile", "avg_game_length", "win_rate", "time_per_game"]
    for metric in metrics:
        orig_val = original_results[metric]
        enh_val = enhanced_results[metric]
        diff = enh_val - orig_val
        pct_imp = (diff / orig_val * 100) if orig_val != 0 else float('inf')
        
        data["metric"].append(metric)
        data["original"].append(orig_val)
        data["enhanced"].append(enh_val)
        data["difference"].append(diff)
        data["percent_improvement"].append(pct_imp)
    
    # Add tile distribution
    all_tiles = sorted(set(list(original_results['tile_distribution'].keys()) + list(enhanced_results['tile_distribution'].keys())))
    for tile in all_tiles:
        orig_pct = original_results['tile_distribution'].get(tile, 0)
        enh_pct = enhanced_results['tile_distribution'].get(tile, 0)
        diff = enh_pct - orig_pct
        pct_imp = (diff / orig_pct * 100) if orig_pct != 0 else float('inf')
        
        data["metric"].append(f"tile_{tile}")
        data["original"].append(orig_pct)
        data["enhanced"].append(enh_pct)
        data["difference"].append(diff)
        data["percent_improvement"].append(pct_imp)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    # Also save configuration
    config_df = pd.DataFrame([config])
    config_filename = filename.replace(".csv", "_config.csv")
    config_df.to_csv(config_filename, index=False)
    
    logging.info(f"Comparison results saved to {filename}")
    
    return filename

def plot_comparison(original_results, enhanced_results, save_dir="results/plots"):
    """
    Create comparison plots and save them.
    
    Args:
        original_results: Dictionary with original MCTS results
        enhanced_results: Dictionary with enhanced MCTS results
        save_dir: Directory to save plots
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Tile distribution comparison
    plt.figure(figsize=(12, 8))
    
    all_tiles = sorted(set(list(original_results['tile_distribution'].keys()) + list(enhanced_results['tile_distribution'].keys())))
    orig_pcts = [original_results['tile_distribution'].get(tile, 0) for tile in all_tiles]
    enh_pcts = [enhanced_results['tile_distribution'].get(tile, 0) for tile in all_tiles]
    
    x = np.arange(len(all_tiles))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, orig_pcts, width, label='Original MCTS')
    rects2 = ax.bar(x + width/2, enh_pcts, width, label='Enhanced MCTS')
    
    ax.set_title('Tile Distribution Comparison')
    ax.set_xlabel('Tile Value')
    ax.set_ylabel('Percentage of Games (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_tiles)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tile_distribution_comparison.png", dpi=300)
    
    # 2. Score distribution comparison
    plt.figure(figsize=(12, 8))
    
    sns.histplot(original_results['scores'], kde=True, label='Original MCTS', alpha=0.6)
    sns.histplot(enhanced_results['scores'], kde=True, label='Enhanced MCTS', alpha=0.6)
    
    plt.title('Score Distribution Comparison')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/score_distribution_comparison.png", dpi=300)
    
    # 3. Max tile distribution comparison
    plt.figure(figsize=(12, 8))
    
    sns.histplot(original_results['max_tiles'], kde=True, label='Original MCTS', alpha=0.6)
    sns.histplot(enhanced_results['max_tiles'], kde=True, label='Enhanced MCTS', alpha=0.6)
    
    plt.title('Max Tile Distribution Comparison')
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/max_tile_distribution_comparison.png", dpi=300)
    
    # 4. Key metrics comparison
    plt.figure(figsize=(12, 8))
    
    metrics = ['avg_score', 'avg_max_tile', 'win_rate']
    orig_vals = [original_results[m] for m in metrics]
    enh_vals = [enhanced_results[m] for m in metrics]
    
    # Normalize values for better visualization
    norm_orig = [v / max(orig_vals[i], enh_vals[i]) * 100 for i, v in enumerate(orig_vals)]
    norm_enh = [v / max(orig_vals[i], enh_vals[i]) * 100 for i, v in enumerate(enh_vals)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, norm_orig, width, label='Original MCTS')
    rects2 = ax.bar(x + width/2, norm_enh, width, label='Enhanced MCTS')
    
    ax.set_title('Key Metrics Comparison (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Avg Score', 'Avg Max Tile', 'Win Rate (%)'])
    ax.legend()
    
    # Add actual value labels
    def autolabel_actual(rects, values):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{values[i]:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel_actual(rects1, orig_vals)
    autolabel_actual(rects2, enh_vals)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/key_metrics_comparison.png", dpi=300)
    
    logging.info(f"Comparison plots saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare original and enhanced MCTS implementations")
    parser.add_argument("--model_path", type=str, default="models/dqn_agent.pt", help="Path to the base agent model")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of MCTS simulations")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for action selection")
    parser.add_argument("--c_puct", type=float, default=C_PUCT, help="Exploration constant")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive simulation count")
    parser.add_argument("--render", action="store_true", help="Render the first game")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Load the base agent
    logging.info(f"Loading base agent from {args.model_path}")
    base_agent = DQNAgent()
    base_agent.load(args.model_path)
    
    # Create configuration dictionary
    config = {
        "model_path": args.model_path,
        "num_games": args.num_games,
        "num_simulations": args.num_simulations,
        "temperature": args.temperature,
        "c_puct": args.c_puct,
        "adaptive_simulations": args.adaptive
    }
    
    # Save original MCTS parameters
    original_c_puct = C_PUCT
    original_dirichlet_alpha = DIRICHLET_ALPHA
    original_max_depth = MAX_DEPTH
    
    # Evaluate original MCTS implementation
    logging.info("Evaluating original MCTS implementation...")
    
    # Reset to original parameters
    MCTS.C_PUCT = original_c_puct
    MCTS.DIRICHLET_ALPHA = original_dirichlet_alpha
    MCTS.MAX_DEPTH = original_max_depth
    
    # Create original MCTS agent
    original_mcts_agent = MCTSAgentWrapper(
        base_agent,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        adaptive_simulations=args.adaptive
    )
    
    # Evaluate original MCTS
    original_results = evaluate_agent(original_mcts_agent, num_games=args.num_games, render=args.render, verbose=args.verbose)
    
    # Evaluate enhanced MCTS implementation
    logging.info("Evaluating enhanced MCTS implementation...")
    
    # Create enhanced MCTS agent
    enhanced_mcts_agent = MCTSAgentWrapper(
        base_agent,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        adaptive_simulations=args.adaptive
    )
    
    # Evaluate enhanced MCTS
    enhanced_results = evaluate_agent(enhanced_mcts_agent, num_games=args.num_games, render=args.render, verbose=args.verbose)
    
    # Print and save comparison
    print_comparison(original_results, enhanced_results, config)
    save_comparison(original_results, enhanced_results, config)
    
    # Create comparison plots
    plot_comparison(original_results, enhanced_results)

if __name__ == "__main__":
    main() 