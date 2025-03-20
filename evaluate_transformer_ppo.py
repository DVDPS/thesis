#!/usr/bin/env python
"""
Evaluation script for the Transformer-based PPO agent on 2048 game.
Loads a saved model and evaluates its performance on multiple games.
"""

import torch
import numpy as np
import argparse
import os
import logging
from src.thesis.environment.game2048 import Game2048, preprocess_state_onehot
from src.thesis.agents.transformer_ppo_agent import TransformerPPOAgent
from src.thesis.config import device, set_seeds
import matplotlib.pyplot as plt
from custom_load import patch_agent_load_method

def evaluate_model(model_path, num_games=100, render=False, deterministic=True, seed=42):
    """
    Evaluate a saved model on multiple games of 2048.
    
    Args:
        model_path: Path to the saved model
        num_games: Number of games to evaluate on
        render: Whether to render the game states
        deterministic: Whether to use deterministic policy
        seed: Random seed
    
    Returns:
        Dictionary of evaluation results
    """
    # Set random seed for reproducibility
    set_seeds(seed)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load the agent
    logging.info(f"Loading model from {model_path}")
    
    # Create a dummy agent with default parameters
    agent = TransformerPPOAgent(
        board_size=4,
        embed_dim=128,  # Using reduced embedding dimension from our optimized settings
        num_heads=4,
        num_layers=4,
        input_channels=16
    )
    
    # Patch the agent's load method for PyTorch 2.6+ compatibility
    agent = patch_agent_load_method(agent)
    
    # Load the saved weights with patched method
    agent.load(model_path)
    logging.info("Model loaded successfully")
    
    # Set evaluation mode
    agent.network.eval()
    
    # Initialize metrics
    scores = []
    max_tiles = []
    game_lengths = []
    
    # Initialize tracking variables for best performance
    best_score = 0
    best_max_tile = 0
    
    # Evaluate on multiple games
    logging.info(f"Evaluating on {num_games} games (deterministic={deterministic})")
    
    for game_idx in range(num_games):
        env = Game2048()
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        current_max_tile = 0
        
        while not done:
            # Process state
            state_proc = preprocess_state_onehot(state)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
            
            # Select action
            action, _, _ = agent.get_action(state_proc, valid_moves, deterministic=deterministic)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Check for new max tile in this state
            state_max_tile = np.max(next_state)
            if state_max_tile > current_max_tile:
                current_max_tile = state_max_tile
                if current_max_tile > best_max_tile:
                    best_max_tile = current_max_tile
                    logging.info(f"Game {game_idx+1}/{num_games}: New best max tile reached: {best_max_tile}!")
                elif game_idx < 10 or (game_idx + 1) % 100 == 0:  # Log for early games or every 100 games
                    logging.info(f"Game {game_idx+1}/{num_games}: Current game max tile: {current_max_tile}")
            
            # Render if requested
            if render and game_idx == 0:  # Only render the first game
                print(f"Step {episode_length}, Action: {action}, Reward: {reward}")
                print(next_state)
                print("--------------------")
            
            # Update state
            state = next_state
        
        # Record metrics
        scores.append(episode_reward)
        max_tiles.append(current_max_tile)
        game_lengths.append(episode_length)
        
        # Check if this game has the best score
        if episode_reward > best_score:
            best_score = episode_reward
            logging.info(f"Game {game_idx+1}/{num_games}: New best score: {best_score}!")
        
        # Log progress
        if (game_idx + 1) % 100 == 0:  # Changed from 10 to 100 for larger evaluation runs
            logging.info(f"Completed {game_idx + 1}/{num_games} games | " +
                         f"Current Avg Score: {np.mean(scores):.2f} | " +
                         f"Current Avg Max Tile: {np.mean(max_tiles):.2f} | " +
                         f"Best Max Tile So Far: {best_max_tile}")
            
            # Create a quick summary of tile distribution so far
            if (game_idx + 1) % 1000 == 0:  # Detailed stats every 1000 games
                current_tiles, current_counts = np.unique(max_tiles, return_counts=True)
                current_distribution = {int(tile): count for tile, count in zip(current_tiles, current_counts)}
                current_percentages = {tile: count / len(max_tiles) * 100 for tile, count in current_distribution.items()}
                
                logging.info("Current Tile Distribution:")
                for tile, count in sorted(current_distribution.items()):
                    if tile >= 512:  # Only show high tiles
                        logging.info(f"  Tile {tile}: {count} games ({current_percentages[tile]:.2f}%)")
    
    # Calculate statistics
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    avg_length = np.mean(game_lengths)
    max_tile_reached = np.max(max_tiles)
    
    # Count occurrences of each max tile
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    tile_distribution = {int(tile): count for tile, count in zip(unique_tiles, counts)}
    
    # Calculate percentages
    percentages = {tile: count / num_games * 100 for tile, count in tile_distribution.items()}
    
    # Log results
    logging.info("Evaluation Results:")
    logging.info(f"Average Score: {avg_score:.2f}")
    logging.info(f"Average Max Tile: {avg_max_tile:.2f}")
    logging.info(f"Average Game Length: {avg_length:.2f}")
    logging.info(f"Best Max Tile: {max_tile_reached}")
    logging.info("Tile Distribution:")
    for tile, count in sorted(tile_distribution.items()):
        logging.info(f"  Tile {tile}: {count} games ({percentages[tile]:.2f}%)")
    
    # Plot results
    plot_evaluation_results(scores, max_tiles, game_lengths)
    
    # Return results
    return {
        "avg_score": avg_score,
        "avg_max_tile": avg_max_tile,
        "avg_length": avg_length,
        "max_tile_reached": max_tile_reached,
        "tile_distribution": tile_distribution,
        "percentages": percentages,
        "scores": scores,
        "max_tiles": max_tiles,
        "game_lengths": game_lengths
    }

def plot_evaluation_results(scores, max_tiles, game_lengths):
    """
    Plot the evaluation results.
    
    Args:
        scores: List of scores
        max_tiles: List of max tiles
        game_lengths: List of game lengths
    """
    plt.figure(figsize=(15, 10))
    
    # Plot score distribution
    plt.subplot(2, 2, 1)
    plt.hist(scores, bins=20)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    
    # Plot max tile distribution
    plt.subplot(2, 2, 2)
    plt.hist(max_tiles, bins=np.unique(max_tiles))
    plt.xlabel('Max Tile')
    plt.ylabel('Count')
    plt.title('Max Tile Distribution')
    plt.xticks(np.unique(max_tiles))
    
    # Plot game length distribution
    plt.subplot(2, 2, 3)
    plt.hist(game_lengths, bins=20)
    plt.xlabel('Game Length')
    plt.ylabel('Count')
    plt.title('Game Length Distribution')
    
    # Plot max tile vs score
    plt.subplot(2, 2, 4)
    plt.scatter(max_tiles, scores, alpha=0.5)
    plt.xlabel('Max Tile')
    plt.ylabel('Score')
    plt.title('Max Tile vs Score')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
    
    # Create a second figure for individual max tile distribution
    plt.figure(figsize=(12, 6))
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    percentages = counts / len(max_tiles) * 100
    
    plt.bar([str(tile) for tile in unique_tiles], percentages)
    plt.xlabel('Max Tile')
    plt.ylabel('Percentage (%)')
    plt.title('Max Tile Distribution (%)')
    
    # Add percentage labels on top of bars
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage + 1, f"{percentage:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('max_tile_distribution.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Transformer PPO agent on 2048 game")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to evaluate on")
    parser.add_argument("--render", action="store_true", help="Render the game states of the first game")
    parser.add_argument("--non-deterministic", action="store_true", help="Use non-deterministic policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    evaluate_model(
        args.model_path, 
        num_games=args.num_games, 
        render=args.render, 
        deterministic=not args.non_deterministic,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 