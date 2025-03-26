"""
MCTS visualization utilities.
This module provides functions to visualize MCTS search processes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

from ...config import device

def visualize_mcts_search(agent, state, num_simulations=100, filename='mcts_analysis.png'):
    """
    Visualize the MCTS search process on a specific board state.
    
    Args:
        agent: MCTS agent wrapper
        state: Board state to analyze
        num_simulations: Number of simulations for the analysis
        filename: Output file name
        
    Returns:
        Analysis results
    """
    # Ensure we have an MCTSAgentWrapper
    if not hasattr(agent, 'analyze'):
        logging.info("Agent doesn't have MCTS analysis capabilities")
        return None
    
    # Analyze the position
    results = agent.analyze(state, num_simulations=num_simulations)
    
    # Print analysis results
    logging.info(f"MCTS Analysis with {num_simulations} simulations:")
    logging.info(f"Time taken: {results['time_taken']:.2f} seconds")
    logging.info(f"Network value estimate: {results['network_value']:.4f}")
    
    logging.info("\nAction analysis:")
    for action in results['action_stats']:
        direction = ['UP', 'RIGHT', 'DOWN', 'LEFT'][action['action']]
        logging.info(f"{direction}: Visits={action['visits']}, Value={action['value']:.4f}, Prior={action['prior']:.4f}, Max Tile={action.get('max_tile', 'N/A')}")
    
    # Visualize the state and MCTS statistics
    plt.figure(figsize=(15, 10))
    
    # Display the board state
    plt.subplot(2, 3, 1)
    if isinstance(state, np.ndarray) and state.shape == (4, 4):
        board = state
    else:
        # Convert tensor to board
        state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        if state_np.ndim > 2:
            # Assume one-hot encoded
            board = np.zeros((4, 4), dtype=np.int32)
            for i in range(1, min(16, state_np.shape[0])):  # Skip channel 0 (empty)
                mask = state_np[i] > 0.5
                board[mask] = 2 ** i
        else:
            board = state_np
    
    # Create a visual representation of the board
    plt.imshow(board, cmap='viridis')
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val > 0:
                plt.text(j, i, str(val), ha='center', va='center', 
                        color='white' if val > 16 else 'black', fontsize=14, fontweight='bold')
    plt.title('Board State')
    plt.axis('off')
    
    # Display action visit counts
    plt.subplot(2, 3, 2)
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    visits = [0, 0, 0, 0]
    for action in results['action_stats']:
        visits[action['action']] = action['visits']
    
    plt.bar(actions, visits)
    plt.xlabel('Action')
    plt.ylabel('Visit Count')
    plt.title('MCTS Action Visit Distribution')
    
    # Display action values
    plt.subplot(2, 3, 3)
    values = [0, 0, 0, 0]
    for action in results['action_stats']:
        values[action['action']] = action['value']
    
    plt.bar(actions, values)
    plt.xlabel('Action')
    plt.ylabel('Action Value')
    plt.title('MCTS Action Value Estimates')
    
    # Compare network policy vs MCTS policy
    plt.subplot(2, 3, 4)
    network_policy = np.zeros(4)
    mcts_policy = np.zeros(4)
    
    for action in results['action_stats']:
        a = action['action']
        network_policy[a] = action['network_policy']
        mcts_policy[a] = action['mcts_policy']
    
    width = 0.35
    x = np.arange(len(actions))
    plt.bar(x - width/2, network_policy, width, label='Network Policy')
    plt.bar(x + width/2, mcts_policy, width, label='MCTS Policy')
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Network vs MCTS Policy')
    plt.xticks(x, actions)
    plt.legend()
    
    # Display max tiles for each action (if available)
    plt.subplot(2, 3, 5)
    max_tiles = [0, 0, 0, 0]
    
    if 'max_tile' in results['action_stats'][0]:
        for action in results['action_stats']:
            max_tiles[action['action']] = action.get('max_tile', 0)
        
        plt.bar(actions, max_tiles)
        plt.xlabel('Action')
        plt.ylabel('Max Tile Seen')
        plt.title('Max Tile by Action Path')
    else:
        plt.text(0.5, 0.5, 'Max tile data not available', ha='center', va='center')
    
    # Display search depth distribution (if available)
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 'Search depth analysis\nnot implemented yet', ha='center', va='center')
    plt.title('Search Depth Distribution')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return results 