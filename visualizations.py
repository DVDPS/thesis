# --- START OF FILE visualizations.py ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker

# --- Configuration ---
DATA_FILE = 'expectimax_data.csv'
OUTPUT_DIR = '.' # Save plots in the current directory
PLOT_STYLE = 'seaborn-v0_8-whitegrid' # Cleaner style
PALETTE = 'viridis' # Color palette

# --- Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data from {DATA_FILE}")
    print(f"Data shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure numeric types where expected
for col in ['Header Steps', 'Header Score', 'Header Max Tile', 'Completed Time (s)',
            'Final Score', 'Final Steps', 'Final Max Tile', 'Running Average Score']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values that might have resulted from coercion
df.dropna(inplace=True)
if df.empty:
    print("Error: No valid data remaining after cleaning.")
    exit()

# Convert Max Tile to integer for clearer plotting if possible
if 'Final Max Tile' in df.columns:
     df['Final Max Tile'] = df['Final Max Tile'].astype(int)
if 'Header Max Tile' in df.columns:
     df['Header Max Tile'] = df['Header Max Tile'].astype(int)


# --- Set Plot Style ---
plt.style.use(PLOT_STYLE)
sns.set_palette(PALETTE)

# --- Plotting Functions ---

def plot_performance_trends(df, filename="expectimax_performance_trends.png"):
    """Plots score, running average, and steps over episodes."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color=color)
    line1 = ax1.plot(df['Episode'], df['Final Score'], color=color, alpha=0.6, label='Final Score')
    line2 = ax1.plot(df['Episode'], df['Running Average Score'], color='tab:red', alpha=0.9, linewidth=2, label='Running Average Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d')) # Format as integer

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Steps', color=color)
    line3 = ax2.plot(df['Episode'], df['Final Steps'], color=color, alpha=0.5, linestyle='--', label='Final Steps')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d')) # Format as integer

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fig.suptitle('Performance Trends Over Episodes (CNN-Expectimax)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filename}")

def plot_max_tile_distribution(df, filename="expectimax_max_tile_distribution.png"):
    """Plots the distribution of the final maximum tile achieved."""
    plt.figure(figsize=(10, 6))
    # Ensure Max Tiles are treated as categories for plotting
    df['Final Max Tile Str'] = df['Final Max Tile'].astype(str)
    # Order the categories numerically
    tile_order = sorted(df['Final Max Tile'].unique())
    tile_order_str = [str(t) for t in tile_order]

    ax = sns.countplot(data=df, x='Final Max Tile Str', order=tile_order_str, palette=PALETTE)

    plt.title('Distribution of Final Maximum Tiles Achieved')
    plt.xlabel('Max Tile Value')
    plt.ylabel('Frequency (Number of Episodes)')

    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container)

    # Optional: Log scale if frequencies vary wildly (uncomment if needed)
    # plt.yscale('log')
    # plt.ylabel('Frequency (Log Scale)')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def plot_correlations(df, filename="expectimax_correlations.png"):
    """Plots correlations between score, steps, and time."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Steps vs Score
    sns.scatterplot(data=df, x='Final Steps', y='Final Score', alpha=0.7, ax=axes[0], hue='Final Max Tile', palette=PALETTE, size='Final Max Tile', sizes=(20, 200))
    axes[0].set_title('Steps vs Score (Colored by Max Tile)')
    axes[0].set_xlabel('Number of Steps')
    axes[0].set_ylabel('Final Score')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(title='Max Tile') # Add legend for hue/size

    # Time vs Score
    sns.scatterplot(data=df, x='Completed Time (s)', y='Final Score', alpha=0.7, ax=axes[1], hue='Final Max Tile', palette=PALETTE, size='Final Max Tile', sizes=(20, 200))
    axes[1].set_title('Completion Time vs Score (Colored by Max Tile)')
    axes[1].set_xlabel('Completed Time (s)')
    axes[1].set_ylabel('Final Score')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(title='Max Tile') # Add legend for hue/size

    fig.suptitle('Correlation Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filename}")

def plot_score_by_max_tile(df, filename="expectimax_score_by_max_tile.png"):
    """Plots the distribution of scores for each max tile achieved using boxplots."""
    plt.figure(figsize=(12, 7))

    # Order the categories numerically
    tile_order = sorted(df['Final Max Tile'].unique())

    sns.boxplot(data=df, x='Final Max Tile', y='Final Score', order=tile_order, palette=PALETTE)
    # Overlay swarmplot for individual points (optional, can be slow for many points)
    # sns.swarmplot(data=df, x='Final Max Tile', y='Final Score', order=tile_order, color=".25", size=3, alpha=0.5)

    plt.title('Score Distribution by Final Max Tile Achieved')
    plt.xlabel('Max Tile Value')
    plt.ylabel('Final Score')
    plt.yscale('log') # Use log scale for score often helps visibility
    plt.ylabel('Final Score (Log Scale)')
    plt.grid(True, which="both", ls="--", linewidth=0.5, axis='y') # Grid lines for log scale

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def plot_time_distribution(df, filename="expectimax_time_distribution.png"):
    """Plots the distribution of episode completion times."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Completed Time (s)', kde=True, bins=15) # Added KDE curve
    median_time = df['Completed Time (s)'].median()
    plt.axvline(median_time, color='red', linestyle='--', label=f'Median Time: {median_time:.1f}s')
    plt.title('Distribution of Episode Completion Times')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Number of Episodes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

# --- Generate Plots ---
print("\nGenerating visualizations...")
plot_performance_trends(df)
plot_max_tile_distribution(df)
plot_correlations(df)
plot_score_by_max_tile(df)
plot_time_distribution(df)

# --- Print Statistics ---
print("\n--- Performance Statistics ---")
print(f"Number of Episodes Analyzed: {len(df)}")
print(f"Average Final Score: {df['Final Score'].mean():,.2f}")
print(f"Median Final Score: {df['Final Score'].median():,.2f}")
print(f"Best Final Score: {df['Final Score'].max():,.2f}")
print(f"Standard Deviation of Score: {df['Final Score'].std():,.2f}")
print("-" * 30)
print(f"Average Final Steps: {df['Final Steps'].mean():.2f}")
print(f"Median Final Steps: {df['Final Steps'].median():.2f}")
print(f"Max Final Steps: {df['Final Steps'].max():.2f}")
print("-" * 30)
print(f"Average Completion Time: {df['Completed Time (s)'].mean():.2f} seconds")
print(f"Median Completion Time: {df['Completed Time (s)'].median():.2f} seconds")
print(f"Total Run Time (Approx): {df['Completed Time (s)'].sum() / 3600:.2f} hours")
print("-" * 30)
print(f"Max Tile Achieved Overall: {df['Final Max Tile'].max()}")
print("Max Tile Frequencies:")
print(df['Final Max Tile'].value_counts().sort_index())
print("-" * 30)
# Count how many times 2048 or higher was reached
reached_2048_count = df[df['Final Max Tile'] >= 2048].shape[0]
print(f"Reached 2048 or higher: {reached_2048_count} times ({reached_2048_count / len(df) * 100:.2f}%)")
print("--- End of Statistics ---")

# --- END OF FILE visualizations.py ---