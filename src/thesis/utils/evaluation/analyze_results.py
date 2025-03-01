"""
Utility to analyze and compare results from multiple evaluations.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

def parse_results_file(file_path):
    """
    Parse a results file and extract key metrics.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Dictionary with parsed results
    """
    results = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Extract average max tile
        avg_max_tile_match = re.search(r'Average Max Tile: (\d+\.\d+)', content)
        if avg_max_tile_match:
            results['avg_max_tile'] = float(avg_max_tile_match.group(1))
        
        # Extract average score
        avg_score_match = re.search(r'Average Score: (\d+\.\d+)', content)
        if avg_score_match:
            results['avg_score'] = float(avg_score_match.group(1))
        
        # Extract best max tile
        best_max_tile_match = re.search(r'Best Max Tile: (\d+)', content)
        if best_max_tile_match:
            results['best_max_tile'] = int(best_max_tile_match.group(1))
        
        # Extract tile distribution
        tile_distribution = {}
        tile_matches = re.finditer(r'(\d+): (\d+) games \((\d+\.\d+)%\)', content)
        for match in tile_matches:
            tile = int(match.group(1))
            count = int(match.group(2))
            percentage = float(match.group(3))
            tile_distribution[tile] = (count, percentage)
        
        results['tile_distribution'] = tile_distribution
    
    return results

def analyze_multiple_results(results_dir, output_dir="analysis_results"):
    """
    Analyze multiple result files and generate comparison visualizations.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "*_results.txt"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Parse all result files
    all_results = {}
    for file_path in result_files:
        file_name = os.path.basename(file_path)
        config_name = file_name.replace("_results.txt", "")
        all_results[config_name] = parse_results_file(file_path)
    
    # Generate comparison visualizations
    generate_comparisons(all_results, output_dir)

def generate_comparisons(all_results, output_dir):
    """
    Generate comparison visualizations from multiple results.
    
    Args:
        all_results: Dictionary with parsed results
        output_dir: Directory to save visualizations
    """
    # Extract configurations and metrics
    configs = list(all_results.keys())
    avg_max_tiles = [results.get('avg_max_tile', 0) for results in all_results.values()]
    avg_scores = [results.get('avg_score', 0) for results in all_results.values()]
    best_max_tiles = [results.get('best_max_tile', 0) for results in all_results.values()]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Average max tile comparison
    plt.subplot(2, 2, 1)
    plt.bar(configs, avg_max_tiles)
    plt.xlabel('Configuration')
    plt.ylabel('Average Max Tile')
    plt.title('Average Max Tile Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Average score comparison
    plt.subplot(2, 2, 2)
    plt.bar(configs, avg_scores)
    plt.xlabel('Configuration')
    plt.ylabel('Average Score')
    plt.title('Average Score Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Best max tile comparison
    plt.subplot(2, 2, 3)
    plt.bar(configs, best_max_tiles)
    plt.xlabel('Configuration')
    plt.ylabel('Best Max Tile')
    plt.title('Best Max Tile Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Tile distribution comparison
    plt.subplot(2, 2, 4)
    
    # Get all unique tiles across all configurations
    all_tiles = set()
    for results in all_results.values():
        all_tiles.update(results.get('tile_distribution', {}).keys())
    
    # Sort tiles
    sorted_tiles = sorted(all_tiles)
    
    # Create grouped bar chart
    bar_width = 0.8 / len(configs)
    for i, (config, results) in enumerate(all_results.items()):
        tile_dist = results.get('tile_distribution', {})
        percentages = [tile_dist.get(tile, (0, 0))[1] for tile in sorted_tiles]
        x = np.arange(len(sorted_tiles))
        plt.bar(x + i * bar_width - 0.4 + bar_width/2, percentages, bar_width, label=config)
    
    plt.xlabel('Tile Value')
    plt.ylabel('Percentage of Games')
    plt.title('Tile Distribution Comparison')
    plt.xticks(np.arange(len(sorted_tiles)), [str(tile) for tile in sorted_tiles])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()
    
    # Generate summary table
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("COMPARISON SUMMARY\n")
        f.write("=================\n\n")
        
        # Table header
        f.write(f"{'Configuration':<20} {'Avg Max Tile':<15} {'Avg Score':<15} {'Best Max Tile':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Table rows
        for config, results in all_results.items():
            f.write(f"{config:<20} {results.get('avg_max_tile', 0):<15.1f} {results.get('avg_score', 0):<15.1f} {results.get('best_max_tile', 0):<15}\n")
        
        f.write("\n\nTILE DISTRIBUTION\n")
        f.write("================\n\n")
        
        # Table header for tile distribution
        header = "Tile".ljust(10)
        for config in configs:
            header += config.ljust(15)
        f.write(header + "\n")
        f.write("-" * (10 + 15 * len(configs)) + "\n")
        
        # Table rows for tile distribution
        for tile in sorted_tiles:
            row = str(tile).ljust(10)
            for config, results in all_results.items():
                tile_dist = results.get('tile_distribution', {})
                percentage = tile_dist.get(tile, (0, 0))[1]
                row += f"{percentage:.1f}%".ljust(15)
            f.write(row + "\n")
    
    print(f"Analysis results saved to {output_dir}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze and compare evaluation results")
    parser.add_argument("--results-dir", type=str, default="evaluation_results", 
                        help="Directory containing result files")
    parser.add_argument("--output-dir", type=str, default="analysis_results", 
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyze_multiple_results(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main() 