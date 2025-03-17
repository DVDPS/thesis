import torch
import numpy as np
import logging
from collections import defaultdict
from ...environment.game2048 import Game2048, preprocess_state_onehot
from ...config import device

def evaluate_agent(agent, env=None, num_games=10, render=False, max_steps=1000, save_trajectories=False):
    if env is None:
        env = Game2048()
        
    max_tiles = []
    scores = []
    steps = []
    tile_progression = defaultdict(list)
    all_trajectories = []   
    
    for game_idx in range(num_games):
        state = env.reset()
        done = False
        step_count = 0
        game_states = []
        game_actions = []
        game_rewards = []
        max_tile_seen = 0
        
        while not done and step_count < max_steps:
            game_states.append(state.copy())
            
            state_proc = preprocess_state_onehot(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get valid moves
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Use get_action instead of calling the agent directly
            with torch.no_grad():
                if hasattr(agent, 'get_action'):
                    # For PPOAgent which has get_action method
                    action, _, _ = agent.get_action(state_proc, valid_moves, deterministic=True)
                else:
                    # For other agents that might be directly callable (like neural networks)
                    logits, _ = agent(state_tensor)
                    action_mask = torch.full((1, 4), float('-inf'), device=device)
                    action_mask[0, valid_moves] = 0
                    masked_logits = logits + action_mask
                    action = torch.argmax(masked_logits, dim=1).item()
            
            game_actions.append(action)
            
            next_state, reward, done, info = env.step(action)
            game_rewards.append(reward)
            
            current_max_tile = np.max(next_state)
            max_tile_seen = max(max_tile_seen, current_max_tile)
            
            state = next_state
            step_count += 1
            
            if render:
                print(f"Game {game_idx+1}, Step {step_count}")
                print(f"Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
                print(f"Reward: {reward:.1f}")
                print(env.board)
                print()
        
        game_states.append(state.copy())
        
        max_tile = max_tile_seen
        score = env.score
        max_tiles.append(max_tile)
        scores.append(score)
        steps.append(step_count)
        
        if save_trajectories:
            all_trajectories.append({
                'states': game_states,
                'actions': game_actions,
                'rewards': game_rewards,
                'max_tile': max_tile,
                'score': score,
                'steps': step_count
            })
        
        logging.info(f"Game {game_idx+1}/{num_games} completed: Max Tile = {max_tile}")
    
    # Calculate statistics
    avg_max_tile = np.mean(max_tiles)
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps)
    max_tile_reached = np.max(max_tiles)
    
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
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