"""
Utility to play and visualize a single game of 2048.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from ...environment.game2048 import Game2048, preprocess_state_onehot
from ...config import device
from .game_analysis import visualize_board_trajectory
from ..mcts import wrap_agent_with_mcts

from ...agents.dqn_agent import DQNAgent

# 2048 color scheme
TILE_COLORS = {
    0: '#CCC0B3',  # Empty tile
    2: '#EEE4DA',
    4: '#EDE0C8',
    8: '#F2B179',
    16: '#F59563',
    32: '#F67C5F',
    64: '#F65E3B',
    128: '#EDCF72',
    256: '#EDCC61',
    512: '#EDC850',
    1024: '#EDC53F',
    2048: '#EDC22E',
    4096: '#3C3A32',
    8192: '#3C3A32',
    16384: '#3C3A32',
    32768: '#3C3A32',
    65536: '#3C3A32'
}

# Text colors
TEXT_COLORS = {
    0: '#776E65',
    2: '#776E65',
    4: '#776E65',
    8: '#F9F6F2',
    16: '#F9F6F2',
    32: '#F9F6F2',
    64: '#F9F6F2',
    128: '#F9F6F2',
    256: '#F9F6F2',
    512: '#F9F6F2',
    1024: '#F9F6F2',
    2048: '#F9F6F2',
    4096: '#F9F6F2',
    8192: '#F9F6F2',
    16384: '#F9F6F2',
    32768: '#F9F6F2',
    65536: '#F9F6F2'
}

def play_single_game(agent, render=True, max_steps=1000, save_visualization=True, output_dir="game_visualization", real_time_viz=False):
    """
    Play a single game of 2048 with the given agent and visualize the results.
    
    Args:
        agent: Agent to play the game
        render: Whether to render the game in the console
        max_steps: Maximum number of steps
        save_visualization: Whether to save visualization of the game
        output_dir: Directory to save visualization
        real_time_viz: Whether to show real-time visualization
        
    Returns:
        Dictionary with game results
    """
    # Create output directory if needed
    if save_visualization:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize game
    env = Game2048()
    state = env.reset()
    done = False
    step_count = 0
    
    # Track game progress
    game_states = [state.copy()]
    game_actions = []
    game_rewards = []
    max_tile_seen = 0
    
    # Set up real-time visualization if requested
    if real_time_viz:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Create a function to update the visualization
        def update_viz(board, action=None, reward=None, value=None):
            ax.clear()  # Clear the axis instead of removing individual artists
            
            # Create a colored grid for the board
            colored_board = np.zeros((4, 4, 3))
            for i in range(4):
                for j in range(4):
                    val = board[i, j]
                    color_hex = TILE_COLORS.get(val, TILE_COLORS[0])
                    # Convert hex to RGB
                    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                    colored_board[i, j] = color_rgb
            
            # Display the board
            ax.imshow(colored_board)
            
            # Add text for each cell
            for i in range(4):
                for j in range(4):
                    val = board[i, j]
                    if val > 0:
                        text_color = TEXT_COLORS.get(val, TEXT_COLORS[2048])
                        ax.text(j, i, str(int(val)), ha='center', va='center', 
                                color=text_color, fontsize=20, fontweight='bold')
            
            # Add grid lines
            for i in range(5):
                ax.axhline(i - 0.5, color='gray', linewidth=2)
                ax.axvline(i - 0.5, color='gray', linewidth=2)
            
            # Add game info
            info_text = f"Step: {step_count}"
            if action is not None:
                info_text += f" | Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}"
            if reward is not None:
                info_text += f" | Reward: {reward:.1f}"
            if value is not None:
                info_text += f" | Value est: {value:.4f}"
            
            ax.set_title(info_text, fontsize=14)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add score and max tile info
            score_text = f"Score: {env.score}"
            max_tile_text = f"Max Tile: {max_tile_seen}"
            ax.text(0.02, -0.08, score_text, transform=ax.transAxes, fontsize=12)
            ax.text(0.7, -0.08, max_tile_text, transform=ax.transAxes, fontsize=12)
            
            # Update the display
            fig.canvas.draw()
            plt.pause(0.01)
    
    # Play the game
    while not done and step_count < max_steps:
        # Process state
        state_proc = preprocess_state_onehot(state)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        
        # Get action from agent
        with torch.no_grad():
            logits, value = agent(state_tensor)
            
        # Apply action mask for valid moves
        valid_moves = env.get_possible_moves()
        if not valid_moves:
            break
            
        action_mask = torch.full((1, 4), float('-inf'), device=device)
        action_mask[0, valid_moves] = 0
        masked_logits = logits + action_mask
        
        # Take best action
        action = torch.argmax(masked_logits, dim=1).item()
        game_actions.append(action)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        game_rewards.append(reward)
        game_states.append(next_state.copy())
        
        # Update max tile seen
        current_max_tile = np.max(next_state)
        max_tile_seen = max(max_tile_seen, current_max_tile)
        
        # Update state
        state = next_state
        step_count += 1
        
        # Render if needed
        if render:
            print(f"Step {step_count}")
            print(f"Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
            print(f"Reward: {reward:.1f}")
            print(f"Value estimate: {value.item():.4f}")
            print(env.board)
            print()
            
        # Update real-time visualization if enabled
        if real_time_viz:
            update_viz(state, action, reward, value.item())
            
        # Slow down rendering if requested
        if render or real_time_viz:
            time.sleep(0.3)  # Slightly faster rendering
    
    # Game results
    max_tile = max_tile_seen
    score = env.score
    
    # Print final results
    print("\n" + "=" * 40)
    print(f"Game completed in {step_count} steps")
    print(f"Max Tile: {max_tile}")
    print(f"Score: {score}")
    print("=" * 40)
    
    # Close the real-time visualization
    if real_time_viz:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)
    
    # Visualize the game
    if save_visualization:
        # Select key frames (beginning, middle, end)
        if len(game_states) > 20:
            # Sample frames throughout the game
            indices = np.linspace(0, len(game_states)-1, 16, dtype=int)
            key_frames = [game_states[i] for i in indices]
        else:
            key_frames = game_states
        
        # Visualize board trajectory
        visualize_board_trajectory(
            key_frames, 
            filename=os.path.join(output_dir, "game_trajectory.png"),
            title=f"2048 Game: Max Tile = {max_tile}, Score = {score}"
        )
        
        # Create action distribution chart
        plt.figure(figsize=(10, 6))
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for action in game_actions:
            action_counts[action] += 1
        
        plt.bar(['UP', 'RIGHT', 'DOWN', 'LEFT'], [action_counts[i] for i in range(4)])
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title('Action Distribution')
        plt.savefig(os.path.join(output_dir, "action_distribution.png"))
        plt.close()
        
        # Create reward progression chart
        plt.figure(figsize=(10, 6))
        plt.plot(game_rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.savefig(os.path.join(output_dir, "reward_progression.png"))
        plt.close()
    
    # Return game results
    return {
        'states': game_states,
        'actions': game_actions,
        'rewards': game_rewards,
        'max_tile': max_tile,
        'score': score,
        'steps': step_count
    }

def main(checkpoint_path, mcts_simulations=0, mcts_temperature=0.5, real_time_viz=True, agent_class=None):
    """
    Main function to run a single game visualization.
    
    Args:
        checkpoint_path: Path to model checkpoint
        mcts_simulations: Number of MCTS simulations (0 for regular agent)
        mcts_temperature: MCTS temperature parameter
        real_time_viz: Whether to show real-time visualization
        agent_class: Class of the agent to use (defaults to EnhancedAgent)
    """
    if agent_class is None:
        from ...agents.enhanced_agent import EnhancedAgent
        agent_class = EnhancedAgent
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = agent_class()
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    if mcts_simulations > 0:
        logging.info(f"Creating MCTS agent with {mcts_simulations} simulations")
        agent = wrap_agent_with_mcts(
            agent, 
            num_simulations=mcts_simulations,
            temperature=mcts_temperature
        )
    play_single_game(agent, render=True, save_visualization=True, real_time_viz=real_time_viz)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play and visualize a single game of 2048")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mcts-simulations", type=int, default=0, help="Number of MCTS simulations (0 for regular agent)")
    parser.add_argument("--mcts-temperature", type=float, default=0.5, help="MCTS temperature parameter")
    parser.add_argument("--real-time-viz", action="store_true", help="Show real-time visualization")
    parser.add_argument("--agent-type", type=str, default="enhanced", help="Type of agent to use (enhanced or dqn)")
    
    args = parser.parse_args()
    
    # Select agent class based on type
    if args.agent_type.lower() == "dqn":
        from ...agents.dqn_agent import DQNAgent
        agent_class = DQNAgent
    else:
        from ...agents.enhanced_agent import EnhancedAgent
        agent_class = EnhancedAgent
        
    main(args.checkpoint, args.mcts_simulations, args.mcts_temperature, args.real_time_viz, agent_class) 