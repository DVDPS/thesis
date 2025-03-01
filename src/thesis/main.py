#!/usr/bin/env python
"""
Main entry point for the 2048 MCTS evaluation system.
"""

import torch
import numpy as np
import logging
import os
import time
from .environment.game2048 import Game2048, preprocess_state_onehot
from .agents.enhanced_agent import EnhancedAgent
from .utils.mcts import wrap_agent_with_mcts
from .utils.evaluation import evaluate_agent
from .utils.cli import setup_logging, parse_args
from .config import device

def main():
    """Main function to run the 2048 MCTS evaluation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging("mcts_evaluation.log")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create game environment
    env = Game2048()
    
    # Create agent
    agent = EnhancedAgent(hidden_dim=args.hidden_dim)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
    else:
        logging.info("No checkpoint provided, using randomly initialized agent")
    
    # Create MCTS agent if needed
    if args.mcts_simulations > 0:
        logging.info(f"Creating MCTS agent with {args.mcts_simulations} simulations and temperature {args.mcts_temperature}")
        mcts_agent = wrap_agent_with_mcts(
            agent, 
            num_simulations=args.mcts_simulations,
            temperature=args.mcts_temperature
        )
    else:
        mcts_agent = None
    
    # Log configuration details
    logging.info(f"MCTS Simulations: {args.mcts_simulations}")
    logging.info(f"MCTS Temperature: {args.mcts_temperature}")
    logging.info(f"Number of games: {args.games}")
    logging.info(f"Device: {device}")
    
    # Evaluate agent(s)
    if args.compare and mcts_agent:
        # Compare regular agent with MCTS-enhanced version
        logging.info("Comparing regular agent with MCTS-enhanced version")
        
        # Evaluate regular agent
        logging.info("Evaluating regular agent...")
        regular_results = evaluate_agent(
            agent, 
            env=env,
            num_games=args.games,
            render=args.render,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories
        )
        
        # Evaluate MCTS agent
        logging.info("Evaluating MCTS agent...")
        mcts_results = evaluate_agent(
            mcts_agent,
            env=env,
            num_games=args.games,
            render=args.render,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories
        )
        
        # Print comparison
        logging.info("\n" + "=" * 50)
        logging.info("COMPARISON RESULTS:")
        logging.info(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, MCTS = {mcts_results['avg_max_tile']:.1f}")
        logging.info(f"Average Score: Regular = {regular_results['avg_score']:.1f}, MCTS = {mcts_results['avg_score']:.1f}")
        logging.info(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, MCTS = {mcts_results['max_tile_reached']}")
        
        # Save results
        results_file = os.path.join(args.output_dir, "comparison_results.txt")
        with open(results_file, "w") as f:
            f.write("COMPARISON RESULTS:\n")
            f.write(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, MCTS = {mcts_results['avg_max_tile']:.1f}\n")
            f.write(f"Average Score: Regular = {regular_results['avg_score']:.1f}, MCTS = {mcts_results['avg_score']:.1f}\n")
            f.write(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, MCTS = {mcts_results['max_tile_reached']}\n")
            
            f.write("\nRegular Agent Tile Distribution:\n")
            for tile, count in sorted(regular_results['tile_counts'].items()):
                f.write(f"  {tile}: {count} games ({count/args.games*100:.1f}%)\n")
                
            f.write("\nMCTS Agent Tile Distribution:\n")
            for tile, count in sorted(mcts_results['tile_counts'].items()):
                f.write(f"  {tile}: {count} games ({count/args.games*100:.1f}%)\n")
        
        logging.info(f"Results saved to {results_file}")
        
    elif mcts_agent:
        # Evaluate only the MCTS agent
        logging.info("Evaluating MCTS agent...")
        results = evaluate_agent(
            mcts_agent,
            env=env,
            num_games=args.games,
            render=args.render,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, "mcts_results.txt")
        with open(results_file, "w") as f:
            f.write("MCTS EVALUATION RESULTS:\n")
            f.write(f"Average Max Tile: {results['avg_max_tile']:.1f}\n")
            f.write(f"Average Score: {results['avg_score']:.1f}\n")
            f.write(f"Average Steps: {results['avg_steps']:.1f}\n")
            f.write(f"Best Max Tile: {results['max_tile_reached']}\n")
            
            f.write("\nTile Distribution:\n")
            for tile, count in sorted(results['tile_counts'].items()):
                f.write(f"  {tile}: {count} games ({count/args.games*100:.1f}%)\n")
        
        logging.info(f"Results saved to {results_file}")
    else:
        # Evaluate only the regular agent
        logging.info("Evaluating regular agent...")
        results = evaluate_agent(
            agent,
            env=env,
            num_games=args.games,
            render=args.render,
            max_steps=args.max_steps,
            save_trajectories=args.save_trajectories
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, "regular_results.txt")
        with open(results_file, "w") as f:
            f.write("REGULAR AGENT EVALUATION RESULTS:\n")
            f.write(f"Average Max Tile: {results['avg_max_tile']:.1f}\n")
            f.write(f"Average Score: {results['avg_score']:.1f}\n")
            f.write(f"Average Steps: {results['avg_steps']:.1f}\n")
            f.write(f"Best Max Tile: {results['max_tile_reached']}\n")
            
            f.write("\nTile Distribution:\n")
            for tile, count in sorted(results['tile_counts'].items()):
                f.write(f"  {tile}: {count} games ({count/args.games*100:.1f}%)\n")
        
        logging.info(f"Results saved to {results_file}")
    
    logging.info("Evaluation complete!")

if __name__ == "__main__":
    main() 