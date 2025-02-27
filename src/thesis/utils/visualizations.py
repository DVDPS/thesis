import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from ..environment.game2048 import Game2048, preprocess_state_onehot
from ..agents.base_agent import PPOAgent

# Add safe globals for model loading
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])

# Color mapping for the 2048 game
TILE_COLORS = {
    0: "#CCC0B3",      # Empty
    2: "#EEE4DA",      # 2
    4: "#EDE0C8",      # 4
    8: "#F2B179",      # 8
    16: "#F59563",     # 16
    32: "#F67C5F",     # 32
    64: "#F65E3B",     # 64
    128: "#EDCF72",    # 128
    256: "#EDCC61",    # 256
    512: "#EDC850",    # 512
    1024: "#EDC53F",   # 1024
    2048: "#EDC22E",   # 2048
    4096: "#3E3933"    # 4096
}

def create_action_labels():
    """Create labels for the action heatmap."""
    return ["Up", "Right", "Down", "Left"]

def onehot_to_board(onehot_state):
    """Convert a one-hot encoded state back to regular board format."""
    board = np.zeros((4, 4), dtype=np.int32)
    for i in range(onehot_state.shape[0]):  # iterate through channels
        if i == 0:  # skip the empty tile channel
            continue
        # Where this channel has 1s, put 2^i in the board
        board += (onehot_state[i] * (2 ** i)).astype(np.int32)
    return board

def create_tile_colormap():
    """Create a colormap for the 2048 game tiles."""
    colors = []
    # Add colors for 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    for tile in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        colors.append(TILE_COLORS.get(tile, "#3C3A32"))  # Default dark color for very high tiles
    # Add one more color for tiles beyond 4096 to match bounds in visualize_game_playthrough
    colors.append("#2C2A22")  # Even darker color for extremely high tiles
    return ListedColormap(colors)

def visualize_board_trajectory(board_states, filename="board_trajectory.png", title=None):
    """
    Visualize a sequence of board states from a game.
    
    Args:
        board_states: List of board states (can be one-hot encoded or regular)
        filename: Where to save the visualization
        title: Optional title for the plot
    """
    # Check if we have onehot encoded states and convert them if needed
    if len(board_states) > 0 and isinstance(board_states[0], np.ndarray) and board_states[0].ndim > 2:
        board_states = [onehot_to_board(state) for state in board_states]
    
    # Select a subset of states to display if there are many
    num_states = len(board_states)
    if num_states == 0:
        print("No board states to visualize")
        return
    
    # Choose number of states to display based on the count
    if num_states <= 9:
        states_to_show = board_states
        grid_size = int(np.ceil(np.sqrt(num_states)))
    else:
        # Sample states evenly across the trajectory
        indices = np.linspace(0, num_states-1, 9, dtype=int)
        states_to_show = [board_states[i] for i in indices]
        grid_size = 3
    
    # Create plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Flatten axes for easier iteration
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create colormap for the tiles
    cmap = create_tile_colormap()
    bounds = np.arange(-0.5, 14.5, 1)  # Bounds for the colormap
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot each state
    for i, (ax, board) in enumerate(zip(axes, states_to_show)):
        # Convert board values to log2 scale for color mapping
        display_board = np.zeros_like(board)
        mask = board > 0
        display_board[mask] = np.log2(board[mask]).astype(int)
        
        # Plot the board
        im = ax.imshow(display_board, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Annotate cells with tile values
        for row in range(4):
            for col in range(4):
                val = board[row, col]
                if val > 0:
                    text_color = "black" if val <= 4 else "white"
                    ax.text(col, row, str(val), ha='center', va='center', 
                            color=text_color, fontsize=12, fontweight='bold')
        
        # Set title for each subplot showing move number
        if num_states <= 9:
            ax.set_title(f"Move {i}")
        else:
            ax.set_title(f"Move {indices[i]}")
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for i in range(len(states_to_show), len(axes)):
        axes[i].axis('off')
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved board trajectory visualization to {filename}")
    return filename

def create_action_heatmap(agent, filename="action_heatmap.png"):
    """
    Create a heatmap showing what actions the agent prefers for different 
    board configurations with high-value tiles in different positions.
    """
    device = next(agent.parameters()).device
    
    # Create a base board with some medium tiles
    base_board = np.zeros((4, 4), dtype=np.int32)
    base_board[1, 1] = 8
    base_board[1, 2] = 4
    base_board[2, 1] = 4
    
    # We'll place a high-value tile at different positions
    positions = [(i, j) for i in range(4) for j in range(4)]
    high_values = [32, 64, 128, 256]
    
    # Create figure
    fig, axes = plt.subplots(len(high_values), 1, figsize=(10, 4*len(high_values)))
    
    for val_idx, high_value in enumerate(high_values):
        action_probs = np.zeros((4, 4, 4))  # (row, col, action)
        
        for pos in positions:
            # Create a board with the high value tile at this position
            board = base_board.copy()
            board[pos] = high_value
            
            # Get agent's action probabilities
            state = preprocess_state_onehot(board)
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent(state_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Store the action probabilities
            action_probs[pos[0], pos[1]] = probs
        
        # Find the preferred action at each position
        preferred_actions = np.argmax(action_probs, axis=2)
        
        # Plot preferred action
        ax = axes[val_idx]
        im = ax.imshow(preferred_actions, cmap='viridis', vmin=0, vmax=3)
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                action_idx = preferred_actions[i, j]
                action_name = ["Up", "Right", "Down", "Left"][action_idx]
                prob = action_probs[i, j, action_idx]
                ax.text(j, i, f"{action_name}\n{prob:.2f}", ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
        
        ax.set_title(f"Preferred actions with {high_value} tile at different positions")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved action heatmap to {filename}")
    return filename

def visualize_game_playthrough(agent, max_steps=100, seed=None, filename="game_animation.gif"):
    """
    Create an animation of the agent playing through a game
    """
    # Initialize game environment
    env = Game2048(seed=seed)
    state = env.reset()
    
    # Track states, actions, and values
    states = [state.copy()]
    values = []
    action_probs_list = []
    rewards = []
    
    # Set up device
    device = next(agent.parameters()).device
    
    # Play through a game
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Process state
        state_proc = preprocess_state_onehot(state)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        
        # Get agent's action
        with torch.no_grad():
            logits, value = agent(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            action = np.argmax(probs)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store information
        states.append(next_state.copy())
        values.append(value.item())
        action_probs_list.append(probs)
        rewards.append(reward)
        
        # Update state
        state = next_state
        step += 1
    
    # Create the animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set up colormap
    cmap = create_tile_colormap()
    bounds = np.arange(-0.5, 14.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Initial board display
    board = states[0]
    display_board = np.zeros_like(board)
    mask = board > 0
    display_board[mask] = np.log2(board[mask]).astype(int)
    
    img = ax1.imshow(display_board, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Text annotations for the board
    texts = []
    for i in range(4):
        for j in range(4):
            txt = ax1.text(j, i, "", ha='center', va='center', fontsize=14, fontweight='bold')
            texts.append(txt)
    
    # Action probability bars
    action_names = create_action_labels()
    bars = ax2.bar(action_names, [0.25, 0.25, 0.25, 0.25])
    ax2.set_ylim(0, 1)
    ax2.set_title("Action Probabilities")
    
    # Value/reward text
    value_text = ax1.text(0.5, -0.1, "", ha='center', va='center', transform=ax1.transAxes)
    
    # Update function for animation
    def update(frame):
        # Update board display
        board = states[frame]
        display_board = np.zeros_like(board)
        mask = board > 0
        display_board[mask] = np.log2(board[mask]).astype(int)
        img.set_array(display_board)
        
        # Update text annotations
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                val = board[i, j]
                if val > 0:
                    texts[idx].set_text(str(val))
                    texts[idx].set_color("black" if val <= 4 else "white")
                else:
                    texts[idx].set_text("")
        
        # Update action probabilities if available
        if frame > 0 and frame - 1 < len(action_probs_list):
            probs = action_probs_list[frame - 1]
            for bar, prob in zip(bars, probs):
                bar.set_height(prob)
        
        # Update value/reward text
        if frame > 0 and frame - 1 < len(values):
            value = values[frame - 1]
            reward = rewards[frame - 1]
            value_text.set_text(f"Value: {value:.1f}, Reward: {reward:.1f}")
        
        # Title with move number and max tile
        max_tile = np.max(board)
        ax1.set_title(f"Move {frame}, Max Tile: {max_tile}")
        
        return [img, value_text] + texts + list(bars)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(states), interval=500, blit=True)
    anim.save(filename, writer='pillow', fps=2)
    print(f"Saved game animation to {filename}")
    return filename

def visualize_value_grid(agent, filename="value_grid.png"):
    """
    Create a grid of game states with their corresponding value estimates.
    Places the max value tile in different positions to see how the agent values them.
    """
    device = next(agent.parameters()).device
    
    # We'll test different patterns with high value tiles
    test_configs = []
    
    # Test different positions for the max tile (256 or 512)
    for high_value in [256, 512]:
        for i in range(4):
            for j in range(4):
                # Create a board with some structure and the high value in different positions
                board = np.zeros((4, 4), dtype=np.int32)
                board[i, j] = high_value
                
                # Add some smaller tiles nearby (to simulate a real board)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4:
                        board[ni, nj] = 16  # Add smaller neighboring tiles
                
                # Add a few other small tiles
                empty_positions = [(r, c) for r in range(4) for c in range(4) 
                                 if board[r, c] == 0 and (r != i or c != j)]
                if empty_positions:
                    for val, pos in zip([8, 4, 2], np.random.choice(len(empty_positions), 3, replace=False)):
                        r, c = empty_positions[pos]
                        board[r, c] = val
                
                test_configs.append((board, f"{high_value} at ({i},{j})"))
    
    # Create a grid of plots
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()
    
    # Sample a subset of configurations to display
    display_indices = np.linspace(0, len(test_configs)-1, rows*cols, dtype=int)
    
    # Create colormap
    cmap = create_tile_colormap()
    bounds = np.arange(-0.5, 14.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    for i, idx in enumerate(display_indices):
        board, title = test_configs[idx]
        ax = axes[i]
        
        # Get the agent's value estimate
        state_proc = preprocess_state_onehot(board)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            _, value = agent(state_tensor)
        
        # Display the board
        display_board = np.zeros_like(board)
        mask = board > 0
        display_board[mask] = np.log2(board[mask]).astype(int)
        
        im = ax.imshow(display_board, cmap=cmap, norm=norm)
        
        # Add tile values
        for r in range(4):
            for c in range(4):
                val = board[r, c]
                if val > 0:
                    text_color = "black" if val <= 4 else "white"
                    ax.text(c, r, str(val), ha='center', va='center', 
                           color=text_color, fontsize=10, fontweight='bold')
        
        # Add value estimate
        ax.set_title(f"Value: {value.item():.1f}\n{title}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved value grid to {filename}")
    return filename

def analyze_model(model_path, output_dir="visualizations"):
    """
    Run a comprehensive analysis of a model, generating all visualizations.
    
    Args:
        model_path: Path to the saved model checkpoint
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Recreate the agent - must match the architecture used in training
    agent = PPOAgent(simple=False, input_channels=16, optimistic=True, Vinit=320000.0, hidden_dim=128)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Extract model info from checkpoint
    epoch = checkpoint.get('epoch', 0)
    max_tile = checkpoint.get('max_tile', 0)
    running_reward = checkpoint.get('running_reward', 0)
    
    # Create a base filename with model info
    base_filename = f"{os.path.splitext(os.path.basename(model_path))[0]}"
    
    # Generate all visualizations
    results = {}
    
    # Generate a few game playthroughs with different seeds
    for seed in range(3):
        animation_file = os.path.join(output_dir, f"{base_filename}_playthrough_seed{seed}.gif")
        results[f"animation_{seed}"] = visualize_game_playthrough(agent, seed=seed, filename=animation_file)
    
    # Create action heatmap
    heatmap_file = os.path.join(output_dir, f"{base_filename}_action_heatmap.png")
    results["action_heatmap"] = create_action_heatmap(agent, filename=heatmap_file)
    
    # Create value grid
    value_file = os.path.join(output_dir, f"{base_filename}_value_grid.png")
    results["value_grid"] = visualize_value_grid(agent, filename=value_file)
    
    # Generate summary image of game states from a longer playthrough
    env = Game2048(seed=42)
    state = env.reset()
    states = [state.copy()]
    device = next(agent.parameters()).device
    
    # Play through a full game
    done = False
    while not done:
        state_proc = preprocess_state_onehot(state)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent(state_tensor)
            action = torch.argmax(logits).item()
        next_state, _, done, _ = env.step(action)
        states.append(next_state.copy())
        state = next_state
    
    # Visualize the board trajectory
    traj_file = os.path.join(output_dir, f"{base_filename}_board_trajectory.png")
    results["board_trajectory"] = visualize_board_trajectory(
        states, 
        filename=traj_file, 
        title=f"Game Trajectory - Epoch {epoch}, Max Tile: {max_tile}, Reward: {running_reward:.1f}"
    )
    
    print(f"Analysis complete. All visualizations saved to {output_dir}")
    return results

# Curriculum learning helper functions
def generate_board_with_high_tile(tile_value):
    """Generate a realistic board state containing the target high-value tile."""
    board = np.zeros((4, 4), dtype=np.int32)
    
    # Place the high-value tile in a strategic position (corner or edge)
    positions = [(0, 0), (0, 3), (3, 0), (3, 3)]  # Corners
    pos = np.random.choice(positions)
    board[pos] = tile_value
    
    # Add some supporting tiles (typically following a pattern that got to the high tile)
    # For example, if the high tile is in the top-left, add decreasing values to the right
    row, col = pos
    
    # Direction to place supporting tiles
    directions = []
    if row == 0:
        directions.append((1, 0))  # Down
    if row == 3:
        directions.append((-1, 0))  # Up
    if col == 0:
        directions.append((0, 1))  # Right
    if col == 3:
        directions.append((0, -1))  # Left
        
    # Place supporting tiles in decreasing value
    for dr, dc in directions[:2]:  # Limit to 2 directions
        r, c = row, col
        value = tile_value // 2
        
        # Add a couple tiles in this direction
        for _ in range(2):
            r += dr
            c += dc
            if 0 <= r < 4 and 0 <= c < 4:
                board[r, c] = value
                value = value // 2
    
    # Fill some random cells with low-value tiles (2, 4, 8)
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if empty_cells:
        for _ in range(min(6, len(empty_cells))):
            r, c = np.random.choice(empty_cells)
            empty_cells.remove((r, c))
            board[r, c] = np.random.choice([2, 4, 8])
            
    return board

def augment_board(board):
    """Apply random rotation and reflection to the board."""
    # Rotate 0, 90, 180, or 270 degrees
    k = np.random.randint(0, 3)
    board = np.rot90(board, k=k)
    
    # Reflect horizontally with 50% probability
    if np.random.random() < 0.5:
        board = np.fliplr(board)
        
    return board 