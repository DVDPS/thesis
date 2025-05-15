import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data(csv_folder):
    """Load the CSV data files."""
    # Load all algorithms data
    all_data_path = os.path.join(csv_folder, 'all_algorithms.csv')
    all_data = pd.read_csv(all_data_path)
    
    # Load algorithm comparison summary
    comparison_path = os.path.join(csv_folder, 'algorithm_comparison.csv')
    comparison = pd.read_csv(comparison_path)
    
    return all_data, comparison

def plot_mean_scores(comparison_df, output_folder):
    """Create bar chart comparing mean scores with error bars."""
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(comparison_df['Algorithm'], comparison_df['Mean_Score'])
    
    # Add error bars
    plt.errorbar(
        x=range(len(comparison_df)), 
        y=comparison_df['Mean_Score'],
        yerr=comparison_df['StdDev_Score'],
        fmt='none', 
        ecolor='black', 
        capsize=5
    )
    
    # Add labels and title
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Score')
    plt.title('Mean Scores by Algorithm with Standard Deviation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_folder, 'mean_scores.png'), dpi=300)
    plt.close()

def plot_median_scores(comparison_df, output_folder):
    """Create bar chart comparing median scores."""
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Algorithm'], comparison_df['Median_Score'])
    plt.xlabel('Algorithm')
    plt.ylabel('Median Score')
    plt.title('Median Scores by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'median_scores.png'), dpi=300)
    plt.close()

def plot_max_tile_counts(comparison_df, output_folder):
    """Create stacked bar chart showing max tile achievement counts."""
    # Prepare data for stacked bar chart
    tile_data = comparison_df[['Algorithm', '2048_Tiles', '1024_Tiles', '512_Tiles', '256_Tiles']]
    
    # Transpose for plotting
    plot_data = tile_data.set_index('Algorithm').T
    
    plt.figure(figsize=(12, 7))
    plot_data.plot(kind='bar', stacked=False)
    plt.xlabel('Max Tile Value')
    plt.ylabel('Number of Episodes')
    plt.title('Tile Achievement Counts by Algorithm')
    plt.xticks(rotation=0)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'max_tile_counts.png'), dpi=300)
    plt.close()
    
    # Also create a 100% stacked bar chart showing percentages
    plt.figure(figsize=(12, 7))
    
    # Calculate the percentage of episodes reaching each tile
    for col in ['2048_Tiles', '1024_Tiles', '512_Tiles', '256_Tiles']:
        comparison_df[f'{col}_Pct'] = comparison_df[col] / comparison_df['Episodes'] * 100
    
    pct_data = comparison_df[['Algorithm', '2048_Tiles_Pct', '1024_Tiles_Pct', '512_Tiles_Pct', '256_Tiles_Pct']]
    pct_plot_data = pct_data.set_index('Algorithm').T
    
    pct_plot_data.plot(kind='bar', stacked=False)
    plt.xlabel('Max Tile Value')
    plt.ylabel('Percentage of Episodes (%)')
    plt.title('Percentage of Episodes Reaching Each Tile by Algorithm')
    plt.xticks(range(4), labels=['2048', '1024', '512', '256'])
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'max_tile_percentages.png'), dpi=300)
    plt.close()

def plot_2048_achievement(comparison_df, output_folder):
    """Create bar chart showing percentage of games reaching 2048."""
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage of games reaching 2048
    comparison_df['2048_Percentage'] = comparison_df['2048_Tiles'] / comparison_df['Episodes'] * 100
    
    plt.bar(comparison_df['Algorithm'], comparison_df['2048_Percentage'])
    plt.xlabel('Algorithm')
    plt.ylabel('Percentage of Games (%)')
    plt.title('Percentage of Games Reaching 2048 Tile')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2048_achievement.png'), dpi=300)
    plt.close()

def plot_mean_time(comparison_df, output_folder):
    """Create bar chart comparing mean time per episode."""
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Algorithm'], comparison_df['Mean_Time'])
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Time (seconds)')
    plt.title('Mean Time per Episode by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mean_time.png'), dpi=300)
    plt.close()

def plot_score_distribution(all_data_df, output_folder):
    """Create box plots and violin plots of score distribution."""
    # Box plot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Algorithm', y='Score', data=all_data_df)
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.title('Score Distribution by Algorithm (Box Plot)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'score_boxplot.png'), dpi=300)
    plt.close()
    
    # Violin plot
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Algorithm', y='Score', data=all_data_df)
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.title('Score Distribution by Algorithm (Violin Plot)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'score_violinplot.png'), dpi=300)
    plt.close()

def plot_score_vs_steps(all_data_df, output_folder):
    """Create scatter plot of score vs steps."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Steps', y='Score', hue='Algorithm', data=all_data_df, alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.title('Score vs Steps by Algorithm')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'score_vs_steps.png'), dpi=300)
    plt.close()

def plot_max_tile_vs_score(all_data_df, output_folder):
    """Create visualizations of max tile vs score relationship."""
    # Scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Max_Tile', y='Score', hue='Algorithm', data=all_data_df, alpha=0.7)
    plt.xlabel('Max Tile')
    plt.ylabel('Score')
    plt.title('Max Tile vs Score by Algorithm')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'max_tile_vs_score_scatter.png'), dpi=300)
    plt.close()
    
    # Box plots for score distribution by max tile and algorithm
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='Max_Tile', y='Score', hue='Algorithm', data=all_data_df)
    plt.xlabel('Max Tile')
    plt.ylabel('Score')
    plt.title('Score Distribution by Max Tile and Algorithm')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'max_tile_vs_score_boxplot.png'), dpi=300)
    plt.close()

def plot_tradeoff_analysis(comparison_df, output_folder):
    """Create scatter plot of mean score vs mean time."""
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot
    sns.scatterplot(x='Mean_Time', y='Mean_Score', data=comparison_df, s=100)
    
    # Add labels to each point
    for i, row in comparison_df.iterrows():
        plt.annotate(row['Algorithm'], 
                     (row['Mean_Time'], row['Mean_Score']),
                     xytext=(5, 5),
                     textcoords='offset points')
    
    plt.xlabel('Mean Time per Episode (seconds)')
    plt.ylabel('Mean Score')
    plt.title('Performance vs Efficiency Trade-off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tradeoff_analysis.png'), dpi=300)
    plt.close()

def main():
    # Settings
    csv_folder = 'csv_output_20250515_122226'
    output_folder = 'visualization_output'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the data
    all_data, comparison = load_data(csv_folder)
    
    # Handle potential duplicate episodes in Expectimax data
    # For visualization purposes, we'll keep all data points
    # but add a note about this in the documentation
    
    # Generate all plots
    plot_mean_scores(comparison, output_folder)
    plot_median_scores(comparison, output_folder)
    plot_max_tile_counts(comparison, output_folder)
    plot_2048_achievement(comparison, output_folder)
    plot_mean_time(comparison, output_folder)
    plot_score_distribution(all_data, output_folder)
    plot_score_vs_steps(all_data, output_folder)
    plot_max_tile_vs_score(all_data, output_folder)
    plot_tradeoff_analysis(comparison, output_folder)
    
    print(f"All visualizations have been generated in the '{output_folder}' directory.")

if __name__ == "__main__":
    main()
