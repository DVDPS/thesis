"""
Comprehensive evaluation script for 2048 MCTS agents.
This script runs evaluations with different MCTS simulation counts and temperatures.
"""

import os
import logging
import argparse
import subprocess
import time
from ...config import device
from ..cli import setup_logging

def run_comprehensive_evaluation(checkpoint_path, games_per_config=5, output_dir="comprehensive_results"):
    """
    Run a comprehensive evaluation with different MCTS configurations.
    
    Args:
        checkpoint_path: Path to model checkpoint
        games_per_config: Number of games to play for each configuration
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "comprehensive_evaluation.log")
    setup_logging(log_file)
    
    # Log start of evaluation
    logging.info("=" * 60)
    logging.info("Starting comprehensive evaluation")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Games per configuration: {games_per_config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device}")
    logging.info("=" * 60)
    
    # Define configurations to evaluate
    simulation_counts = [0, 50, 100, 200, 400]  # 0 means no MCTS
    temperatures = [0.5]  # Default temperature
    
    # Run evaluations for each configuration
    for simulations in simulation_counts:
        for temperature in temperatures:
            config_name = f"sims_{simulations}_temp_{temperature}"
            config_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            
            # Log start of configuration evaluation
            logging.info(f"\nEvaluating configuration: {config_name}")
            logging.info(f"Simulations: {simulations}, Temperature: {temperature}")
            
            # Build command
            cmd = [
                "python", "-m", "src.thesis.run_evaluation",
                "--checkpoint", checkpoint_path,
                "--games", str(games_per_config),
                "--output-dir", config_dir,
                "--save-trajectories"
            ]
            
            # Add MCTS parameters if using MCTS
            if simulations > 0:
                cmd.extend(["--mcts-simulations", str(simulations)])
                cmd.extend(["--mcts-temperature", str(temperature)])
            
            # Log command
            logging.info(f"Running command: {' '.join(cmd)}")
            
            # Run evaluation
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logging.info(f"Evaluation completed successfully")
                    logging.info(result.stdout)
                else:
                    logging.error(f"Evaluation failed with return code {result.returncode}")
                    logging.error(result.stderr)
            except Exception as e:
                logging.error(f"Error running evaluation: {e}")
            
            # Log time taken
            elapsed_time = time.time() - start_time
            logging.info(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Run analysis on all results
    logging.info("\n" + "=" * 60)
    logging.info("Running analysis on all results")
    
    try:
        analysis_cmd = [
            "python", "-m", "src.thesis.utils.evaluation.analyze_results",
            "--results-dir", output_dir,
            "--output-dir", os.path.join(output_dir, "analysis")
        ]
        
        logging.info(f"Running command: {' '.join(analysis_cmd)}")
        
        result = subprocess.run(analysis_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Analysis completed successfully")
            logging.info(result.stdout)
        else:
            logging.error(f"Analysis failed with return code {result.returncode}")
            logging.error(result.stderr)
    except Exception as e:
        logging.error(f"Error running analysis: {e}")
    
    # Log completion
    logging.info("\n" + "=" * 60)
    logging.info("Comprehensive evaluation completed")
    logging.info(f"Results saved to {output_dir}")
    logging.info("=" * 60)

def main():
    """Main function to run the comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation of 2048 MCTS agents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=5, help="Number of games per configuration")
    parser.add_argument("--output-dir", type=str, default="comprehensive_results", help="Directory to save results")
    
    args = parser.parse_args()
    run_comprehensive_evaluation(args.checkpoint, args.games, args.output_dir)

if __name__ == "__main__":
    main() 