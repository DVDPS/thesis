import matplotlib.pyplot as plt
import numpy as np

# Game data
scores = [7764, 15824, 1276, 13820, 12224]
max_tiles = [512, 1024, 128, 1024, 1024]
game_indices = list(range(1, 6))

# Create figure with subplots
plt.figure(figsize=(15, 10))

# 1. Score Distribution
plt.subplot(2, 2, 1)
plt.bar(game_indices, scores, color='skyblue')
plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.0f}')
plt.xlabel('Game Number')
plt.ylabel('Score')
plt.title('Score Distribution Across Games')
plt.legend()

# 2. Max Tile Distribution
plt.subplot(2, 2, 2)
plt.bar(game_indices, max_tiles, color='lightgreen')
plt.axhline(y=np.mean(max_tiles), color='r', linestyle='--', label=f'Mean: {np.mean(max_tiles):.0f}')
plt.xlabel('Game Number')
plt.ylabel('Max Tile')
plt.title('Max Tile Distribution Across Games')
plt.legend()

# 3. Tile Achievement Rate
unique_tiles = sorted(set(max_tiles))
tile_counts = {tile: max_tiles.count(tile) for tile in unique_tiles}
plt.subplot(2, 2, 3)
tiles = list(tile_counts.keys())
counts = list(tile_counts.values())
plt.bar(tiles, [count/len(max_tiles)*100 for count in counts], color='salmon')
plt.xlabel('Tile Value')
plt.ylabel('Achievement Rate (%)')
plt.title('Tile Achievement Rate')

# 4. Score vs Max Tile Correlation
plt.subplot(2, 2, 4)
plt.scatter(max_tiles, scores, color='purple', alpha=0.6)
plt.xlabel('Max Tile')
plt.ylabel('Score')
plt.title('Score vs Max Tile Correlation')

plt.tight_layout()
plt.savefig('analysis_results/game_analysis.png')
plt.close()

# Create a second figure for performance metrics
plt.figure(figsize=(12, 6))

# 5. Success Rate by Threshold
thresholds = [128, 256, 512, 1024]
success_rates = [sum(1 for tile in max_tiles if tile >= threshold) / len(max_tiles) * 100 
                for threshold in thresholds]

plt.bar(thresholds, success_rates, color='lightblue')
plt.xlabel('Tile Threshold')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Tile Threshold')

plt.tight_layout()
plt.savefig('analysis_results/success_rates.png')
plt.close() 