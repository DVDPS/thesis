"""
Evaluation script for MCTS enhanced agents.
This module provides functionality to evaluate and visualize the performance 
of MCTS-enhanced agents compared to regular agents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from collections import defaultdict
from ..environment.game2048 import Game2048, preprocess_state_onehot
from ..config import device
from .mcts_agent_wrapper import wrap_agent_with_mcts
from .visualizations import visualize_board_trajectory

def evaluate_agent(agent, num_games=10, render=False, max_steps=1000, save_trajectories=False):
    """
    Evaluate an agent's performance on the 2048 game.
    
    Args:
        agent: Agent to evaluate
        num_games: Number of games to play
        render: Whether to render the games (print board states)
        max_steps: Maximum number of steps per game
        save_trajectories: Whether to save full game trajectories for analysis
        
    Returns:
        Dictionary with evaluation results
    """
    env = Game2048()
    max_tiles = []
    scores = []
    steps = []
    tile_progression = defaultdict(list)  # Track when each tile value is first achieved
    all_trajectories = []  # Store full game trajectories if requested
    
    for game_idx in range(num_games):
        state = env.reset()
        done = False
        step_count = 0
        game_states = []
        game_actions = []
        game_rewards = []
        max_tile_seen = 0
        tiles_achieved = set()  # Track tiles we've already logged for this game
        
        logging.info(f"Starting game {game_idx+1}/{num_games}")
        
        while not done and step_count < max_steps:
            # Store state for visualization
            game_states.append(state.copy())
            
            # Process state
            state_proc = preprocess_state_onehot(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get action from agent
            # If this is an MCTS agent wrapper, it will handle the MCTS search
            with torch.no_grad():
                logits, _ = agent(state_tensor)
                
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
            
            # Check for new max tile
            current_max_tile = np.max(next_state)
            if current_max_tile > max_tile_seen:
                max_tile_seen = current_max_tile
                # Only log when a new maximum tile is achieved for this game and hasn't been logged yet
                if current_max_tile not in tiles_achieved and current_max_tile >= 64:  # Only log 64 and above
                    tiles_achieved.add(current_max_tile)
                    logging.info(f"Game {game_idx+1}: Achieved {current_max_tile} tile!")
            
            # Track when each tile value is first achieved
            if current_max_tile not in tiles_achieved and current_max_tile >= 16:
                tiles_achieved.add(current_max_tile)
                tile_progression[current_max_tile].append(step_count)
            
            state = next_state
            step_count += 1
            
            # Render if needed
            if render:
                print(f"Game {game_idx+1}, Step {step_count}")
                print(f"Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
                print(f"Reward: {reward:.1f}")
                print(env.board)
                print()
        
        # Record final state for visualization
        game_states.append(state.copy())
        
        # Store game results
        max_tile = np.max(state)
        score = env.score
        max_tiles.append(max_tile)
        scores.append(score)
        steps.append(step_count)
        
        # Save full trajectory if requested
        if save_trajectories:
            all_trajectories.append({
                'states': game_states,
                'actions': game_actions,
                'rewards': game_rewards,
                'max_tile': max_tile,
                'score': score,
                'steps': step_count
            })
        
        # Report final game results
        logging.info(f"Game {game_idx+1}/{num_games} completed: Score = {score}, Max Tile = {max_tile}, Steps = {step_count}")
        
        # Save board trajectory
        if render:
            visualize_board_trajectory(
                game_states, 
                filename=f"game_{game_idx+1}_trajectory.png",
                title=f"Game {game_idx+1}: Max Tile = {max_tile}, Score = {score}"
            )
    
    # Calculate statistics
    avg_max_tile = np.mean(max_tiles)
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps)
    max_tile_reached = np.max(max_tiles)
    
    # Count occurrences of each tile
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    # Calculate average step when each tile value is first achieved
    avg_tile_progression = {}
    for tile, step_list in tile_progression.items():
        if step_list:
            avg_tile_progression[tile] = sum(step_list) / len(step_list)
    
    # Print summary
    logging.info("=" * 40)
    logging.info(f"Evaluation over {num_games} games:")
    logging.info(f"Average Max Tile: {avg_max_tile:.1f}")
    logging.info(f"Average Score: {avg_score:.1f}")
    logging.info(f"Average Steps: {avg_steps:.1f}")
    logging.info(f"Best Max Tile: {max_tile_reached}")
    
    logging.info("Tile distribution:")
    for tile, count in sorted(tile_counts.items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
    
    logging.info("Average steps to achieve tile:")
    for tile, avg_steps in sorted(avg_tile_progression.items()):
        logging.info(f"  {tile}: {avg_steps:.1f} steps")
    
    return {
        'max_tiles': max_tiles,
        'scores': scores,
        'steps': steps,
        'avg_max_tile': avg_max_tile,
        'avg_score': avg_score,
        'avg_steps': avg_steps,
        'max_tile_reached': max_tile_reached,
        'tile_counts': tile_counts,
        'tile_progression': avg_tile_progression,
        'trajectories': all_trajectories if save_trajectories else None
    }

def compare_agents(regular_agent, mcts_agent, num_games=5, max_steps=1000, save_trajectories=False):
    """
    Compare performance between a regular agent and an MCTS-enhanced version.
    
    Args:
        regular_agent: Regular agent without MCTS
        mcts_agent: MCTS-enhanced agent
        num_games: Number of games to play with each agent
        max_steps: Maximum steps per game
        save_trajectories: Whether to save full game trajectories
        
    Returns:
        Dictionary with comparison results
    """
    logging.info("Evaluating regular agent...")
    regular_results = evaluate_agent(regular_agent, num_games=num_games, max_steps=max_steps, save_trajectories=save_trajectories)
    
    logging.info("\nEvaluating MCTS agent...")
    mcts_results = evaluate_agent(mcts_agent, num_games=num_games, max_steps=max_steps, save_trajectories=save_trajectories)
    
    # Print comparison
    logging.info("\n" + "=" * 50)
    logging.info("COMPARISON RESULTS:")
    logging.info(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, MCTS = {mcts_results['avg_max_tile']:.1f}")
    logging.info(f"Average Score: Regular = {regular_results['avg_score']:.1f}, MCTS = {mcts_results['avg_score']:.1f}")
    logging.info(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, MCTS = {mcts_results['max_tile_reached']}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Max tile histogram
    plt.subplot(2, 3, 1)
    bins = np.arange(0, 2100, 100)
    plt.hist(regular_results['max_tiles'], bins=bins, alpha=0.5, label='Regular Agent')
    plt.hist(mcts_results['max_tiles'], bins=bins, alpha=0.5, label='MCTS Agent')
    plt.xlabel('Max Tile')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Max Tile Distribution')
    
    # Score comparison
    plt.subplot(2, 3, 2)
    plt.boxplot([regular_results['scores'], mcts_results['scores']], labels=['Regular', 'MCTS'])
    plt.ylabel('Score')
    plt.title('Score Comparison')
    
    # Max tiles for each game
    plt.subplot(2, 3, 3)
    plt.plot(regular_results['max_tiles'], 'bo-', label='Regular Agent')
    plt.plot(mcts_results['max_tiles'], 'ro-', label='MCTS Agent')
    plt.xlabel('Game')
    plt.ylabel('Max Tile')
    plt.legend()
    plt.title('Max Tile by Game')
    
    # Tile progression comparison
    plt.subplot(2, 3, 4)
    
    # Get common tile values
    all_tiles = set(regular_results['tile_progression'].keys()) | set(mcts_results['tile_progression'].keys())
    sorted_tiles = sorted(all_tiles)
    
    reg_steps = [regular_results['tile_progression'].get(t, float('nan')) for t in sorted_tiles]
    mcts_steps = [mcts_results['tile_progression'].get(t, float('nan')) for t in sorted_tiles]
    
    x = np.arange(len(sorted_tiles))
    width = 0.35
    
    plt.bar(x - width/2, reg_steps, width, label='Regular Agent')
    plt.bar(x + width/2, mcts_steps, width, label='MCTS Agent')
    plt.xlabel('Tile Value')
    plt.ylabel('Avg Steps to Achieve')
    plt.title('Tile Achievement Speed')
    plt.xticks(x, [str(t) for t in sorted_tiles])
    plt.legend()
    
    # Average max tile by simulation count
    plt.subplot(2, 3, 5)
    
    # If we have different simulation counts data
    sim_counts = [5, 10, 25, 50, 100, 200]
    avg_tiles = []
    avg_scores = []
    
    if hasattr(mcts_agent, 'num_simulations'):
        orig_sim_count = mcts_agent.num_simulations
        
        for sim_count in sim_counts:
            if sim_count <= 200:  # Skip higher counts for faster testing
                logging.info(f"Testing with {sim_count} simulations...")
                mcts_agent.set_num_simulations(sim_count)
                results = evaluate_agent(mcts_agent, num_games=2, max_steps=max_steps)
                avg_tiles.append(results['avg_max_tile'])
                avg_scores.append(results['avg_score'])
        
        # Restore original simulation count
        mcts_agent.set_num_simulations(orig_sim_count)
        
        plt.plot(sim_counts[:len(avg_tiles)], avg_tiles, 'ro-')
        plt.xlabel('MCTS Simulations')
        plt.ylabel('Avg Max Tile')
        plt.title('Performance vs Simulation Count')
    else:
        plt.text(0.5, 0.5, 'Not applicable', ha='center', va='center')
    
    # Temperature comparison if available
    plt.subplot(2, 3, 6)
    
    if hasattr(mcts_agent, 'set_temperature'):
        orig_temp = mcts_agent.temperature
        
        temps = [0.1, 0.5, 1.0, 1.5, 2.0]
        temp_tiles = []
        
        for temp in temps:
            logging.info(f"Testing with temperature {temp}...")
            mcts_agent.set_temperature(temp)
            results = evaluate_agent(mcts_agent, num_games=2, max_steps=max_steps)
            temp_tiles.append(results['avg_max_tile'])
        
        # Restore original temperature
        mcts_agent.set_temperature(orig_temp)
        
        plt.plot(temps, temp_tiles, 'go-')
        plt.xlabel('Temperature')
        plt.ylabel('Avg Max Tile')
        plt.title('Performance vs Temperature')
    else:
        plt.text(0.5, 0.5, 'Not applicable', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png')
    plt.close()
    
    return {
        'regular': regular_results,
        'mcts': mcts_results
    }

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


def main(checkpoint_path, num_games=5, mcts_simulations=100):
    """
    Main function to run the MCTS evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations
    """
    from ..agents.enhanced_agent import EnhancedAgent
    
    # Create output directory
    os.makedirs("mcts_evaluation", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mcts_evaluation/evaluation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Load model
    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    agent = EnhancedAgent()
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Create MCTS-enhanced version
    logging.info(f"Creating MCTS agent with {mcts_simulations} simulations")
    mcts_agent = wrap_agent_with_mcts(agent, num_simulations=mcts_simulations, adaptive_simulations=True)
    
    # Compare both agents
    compare_results = compare_agents(agent, mcts_agent, num_games=num_games, save_trajectories=True)
    
    # Analyze a specific position
    env = Game2048()
    state = env.reset()
    
    # Play until we reach a state with a high-value tile
    steps = 0
    max_steps = 100
    target_value = 64  # Target a position with at least a 64 tile
    
    while np.max(state) < target_value and steps < max_steps:
        valid_moves = env.get_possible_moves()
        if not valid_moves:
            break
            
        # Use regular agent to generate the position
        state_proc = preprocess_state_onehot(state)
        state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = agent(state_tensor)
            
        # Apply action mask
        action_mask = torch.full((1, 4), float('-inf'), device=device)
        action_mask[0, valid_moves] = 0
        masked_logits = logits + action_mask
        
        # Take best action
        action = torch.argmax(masked_logits, dim=1).item()
        
        # Execute action
        state, _, done, _ = env.step(action)
        steps += 1
        
        if done:
            break
    
    # Analyze the position
    logging.info(f"Analyzing position after {steps} steps:")
    logging.info(state)
    
    # Visualize MCTS analysis
    visualize_mcts_search(mcts_agent, state, filename='mcts_evaluation/position_analysis.png')
    
    # Analyze a full game trajectory
    if compare_results['mcts']['trajectories']:
        # Find the game with the highest max tile
        best_game = max(compare_results['mcts']['trajectories'], key=lambda x: x['max_tile'])
        analyze_game_trajectory(best_game, filename='mcts_evaluation/best_game_analysis.png')
    
    # Log comparison summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Regular Agent - Avg Max Tile: {compare_results['regular']['avg_max_tile']:.1f}, Best: {compare_results['regular']['max_tile_reached']}")
    logging.info(f"MCTS Agent ({mcts_simulations} sims) - Avg Max Tile: {compare_results['mcts']['avg_max_tile']:.1f}, Best: {compare_results['mcts']['max_tile_reached']}")
    
    # Test with different simulation counts
    logging.info("\nTesting different simulation counts:")
    sim_counts = [50, 100, 200, 400]
    for sims in sim_counts:
        mcts_agent.set_num_simulations(sims)
        results = evaluate_agent(mcts_agent, num_games=2)
        logging.info(f"MCTS with {sims} simulations: Avg Max Tile = {results['avg_max_tile']:.1f}, Best = {results['max_tile_reached']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MCTS enhancement of 2048 agents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=100, help="Number of MCTS simulations")
    
    args = parser.parse_args()
    main(args.checkpoint, args.games, args.simulations) 