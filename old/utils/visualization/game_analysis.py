"""
Game analysis utilities for 2048.
This module provides functions to analyze and visualize game trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_game_trajectory(trajectory, filename='trajectory_analysis.png'):
    """
    Analyze a full game trajectory to understand agent behavior.
    
    Args:
        trajectory: Game trajectory dictionary
        filename: Output file name
    """
    states = trajectory['states']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    max_tile = trajectory['max_tile']
    score = trajectory['score']
    
    # Calculate statistics
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # UP, RIGHT, DOWN, LEFT
    for action in actions:
        action_counts[action] += 1
    
    # Track max tile progression
    max_tiles = []
    for state in states:
        max_tiles.append(np.max(state))
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Action distribution
    plt.subplot(2, 2, 1)
    action_labels = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    plt.bar(action_labels, [action_counts[i] for i in range(4)])
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Distribution')
    
    # Max tile progression
    plt.subplot(2, 2, 2)
    plt.plot(max_tiles)
    plt.xlabel('Step')
    plt.ylabel('Max Tile')
    plt.title('Max Tile Progression')
    
    # Reward progression
    plt.subplot(2, 2, 3)
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Progression')
    
    # Cumulative reward
    plt.subplot(2, 2, 4)
    plt.plot(np.cumsum(rewards))
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Score Progression')
    
    plt.suptitle(f'Game Analysis: Max Tile = {max_tile}, Score = {score}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def visualize_board_trajectory(boards, filename='board_trajectory.png', title=None):
    """
    Visualize a sequence of board states from a game.
    
    Args:
        boards: List of board states (numpy arrays)
        filename: Output file name
        title: Optional title for the plot
    """
    # Determine grid size based on number of boards
    n_boards = len(boards)
    grid_size = int(np.ceil(np.sqrt(n_boards)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot each board
    for i, board in enumerate(boards):
        if i < n_boards:
            ax = axes[i]
            im = ax.imshow(board, cmap='viridis')
            
            # Add text labels for tile values
            for row in range(board.shape[0]):
                for col in range(board.shape[1]):
                    val = board[row, col]
                    if val > 0:
                        ax.text(col, row, str(int(val)), ha='center', va='center',
                                color='white' if val > 16 else 'black', fontsize=10, fontweight='bold')
            
            ax.set_title(f"Step {i}")
            ax.axis('off')
        else:
            axes[i].axis('off')
    
    # Add overall title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_heatmap(data, title='Heatmap', filename='heatmap.png'):
    """
    Create a heatmap visualization of 2D data.
    
    Args:
        data: 2D numpy array
        title: Title for the heatmap
        filename: Output file name
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    
    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center',
                    color='white' if data[i, j] > np.max(data)/2 else 'black')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 