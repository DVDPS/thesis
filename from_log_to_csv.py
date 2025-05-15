import re
import csv
import os
import statistics
from datetime import datetime

def convert_logs_to_csv(log_files, output_dir='.'):
    """
    Convert 2048 game log files to CSV format.
    
    Args:
        log_files: Dictionary with algorithm names as keys and file paths as values
        output_dir: Directory to save the CSV files
    
    Returns:
        Dictionary with algorithm names and paths to created CSV files
    """
    csv_files = {}
    all_episodes = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for algorithm, file_path in log_files.items():
        # Read the log file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract episode data using regex
        episodes = []
        
        # Pattern to match final score lines
        pattern = r"Episode\s+(\d+)(?:/\d+)?.*?completed in (\d+\.\d+)s\nFinal Score: ([\d,]+) \| Steps: (\d+) \| Max Tile: ([\d,.]+)"
        
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            episode_num = int(match.group(1))
            time_taken = float(match.group(2))
            score = int(match.group(3).replace(',', ''))
            steps = int(match.group(4))
            max_tile = float(match.group(5).replace(',', ''))
            
            episode_data = {
                'Episode': episode_num,
                'Score': score,
                'Steps': steps,
                'Max_Tile': max_tile,
                'Time_Taken': time_taken,
                'Algorithm': algorithm
            }
            
            episodes.append(episode_data)
            all_episodes.append(episode_data)
        
        # Sort episodes by episode number
        episodes.sort(key=lambda x: x['Episode'])
        
        # Calculate summary statistics
        if episodes:
            scores = [ep['Score'] for ep in episodes]
            steps = [ep['Steps'] for ep in episodes]
            times = [ep['Time_Taken'] for ep in episodes]
            max_tiles = [ep['Max_Tile'] for ep in episodes]
            
            summary = {
                'Algorithm': algorithm,
                'Episodes': len(episodes),
                'Mean_Score': statistics.mean(scores),
                'Median_Score': statistics.median(scores),
                'StdDev_Score': statistics.stdev(scores) if len(scores) > 1 else 0,
                'Min_Score': min(scores),
                'Max_Score': max(scores),
                'Mean_Steps': statistics.mean(steps),
                'Median_Steps': statistics.median(steps),
                'StdDev_Steps': statistics.stdev(steps) if len(steps) > 1 else 0,
                'Mean_Time': statistics.mean(times),
                'Total_Time': sum(times),
                'Max_Tile_Reached': max(max_tiles),
                '2048_Tiles': sum(1 for t in max_tiles if t >= 2048),
                '1024_Tiles': sum(1 for t in max_tiles if t >= 1024),
                '512_Tiles': sum(1 for t in max_tiles if t >= 512),
                '256_Tiles': sum(1 for t in max_tiles if t >= 256)
            }
            
            # Print summary statistics
            print(f"\nSummary for {algorithm}:")
            print(f"Episodes: {summary['Episodes']}")
            print(f"Average Score: {summary['Mean_Score']:.1f} ± {summary['StdDev_Score']:.1f}")
            print(f"Average Steps: {summary['Mean_Steps']:.1f} ± {summary['StdDev_Steps']:.1f}")
            print(f"Highest Tile: {summary['Max_Tile_Reached']}")
            print(f"Total Time: {summary['Total_Time']:.1f}s")
        
        # Write episode data to CSV
        csv_path = os.path.join(output_dir, f"{algorithm.replace(' ', '_')}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Episode', 'Score', 'Steps', 'Max_Tile', 'Time_Taken', 'Algorithm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for episode in episodes:
                writer.writerow(episode)
        
        csv_files[algorithm] = csv_path
        
        print(f"Created CSV for {algorithm} with {len(episodes)} episodes: {csv_path}")
        
        # Write summary to a separate file
        if episodes:
            summary_path = os.path.join(output_dir, f"{algorithm.replace(' ', '_')}_summary.csv")
            with open(summary_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
                writer.writeheader()
                writer.writerow(summary)
    
    # Create a combined CSV with all algorithms
    combined_csv_path = os.path.join(output_dir, "all_algorithms.csv")
    with open(combined_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Score', 'Steps', 'Max_Tile', 'Time_Taken', 'Algorithm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # Sort by algorithm first, then episode number
        all_episodes.sort(key=lambda x: (x['Algorithm'], x['Episode']))
        for episode in all_episodes:
            writer.writerow(episode)
    
    print(f"\nCreated combined CSV with all algorithms: {combined_csv_path}")
    
    # Create a summary CSV with statistics for all algorithms
    summary_csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Algorithm', 'Episodes', 'Mean_Score', 'Median_Score', 'StdDev_Score', 
            'Min_Score', 'Max_Score', 'Mean_Steps', 'Median_Steps', 'StdDev_Steps', 
            'Mean_Time', 'Total_Time', 'Max_Tile_Reached', 
            '2048_Tiles', '1024_Tiles', '512_Tiles', '256_Tiles'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for algorithm in log_files.keys():
            summary_path = os.path.join(output_dir, f"{algorithm.replace(' ', '_')}_summary.csv")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        writer.writerow(row)
    
    print(f"Created algorithm comparison CSV: {summary_csv_path}")
    
    return csv_files

def main():
    # Get current timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'csv_output_{timestamp}'
    
    # Define log files with correct path
    log_dir = 'logs and csv'
    log_files = {
        'Beam Search': os.path.join(log_dir, 'beam_search.log'),
        'Beam Search Depth 5': os.path.join(log_dir, 'beam_search_depth_5.log'),
        'Expectimax': os.path.join(log_dir, 'expectimax.log'),
        'Hybrid Beam Search': os.path.join(log_dir, 'hybrid_beam_search.log')
    }
    
    # Convert logs to CSV
    csv_files = convert_logs_to_csv(log_files, output_dir)
    
    print(f"\nAll CSV files have been generated in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()