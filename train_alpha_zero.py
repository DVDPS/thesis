import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from agents.alpha_zero_2048 import AlphaZeroAgent, AlphaZeroNet, MCTSNode
from src.thesis.environment.game2048 import Game2048, compute_monotonicity
import time
import os
from typing import List, Tuple
import random

class ExperienceDataset(Dataset):
    """Dataset for storing training experiences"""
    def __init__(self, experiences: List[Tuple[np.ndarray, np.ndarray, float]]):
        self.experiences = experiences
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        state, policy, value = self.experiences[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(policy, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)

def to_numpy(arr):
    """Convert tensor to numpy if needed"""
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return arr

def self_play(agent: AlphaZeroAgent, num_games: int = 100) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate training data through self-play"""
    experiences = []
    game = Game2048()
    
    for game_idx in range(num_games):
        game.reset()
        game_history = []
        start_time = time.time()
        move_count = 0
        last_max_tile = 0
        
        while not game.is_game_over():
            # Get current state
            state = game.get_state()
            current_max_tile = np.max(state)
            
            # Get move probabilities from MCTS
            root = MCTSNode(state)
            for sim_idx in range(agent.num_simulations):
                node = agent.select_node(root)
                if not node.is_terminal and node.visits > 0:
                    node = agent.expand_node(node)
                if node.is_terminal:
                    value = 0.0
                else:
                    # Calculate value using game's heuristics
                    if isinstance(node.state, torch.Tensor):
                        max_tile = node.state.max().item()
                        empty_cells = torch.sum(node.state == 0).item()
                    else:
                        max_tile = np.max(node.state)
                        empty_cells = np.sum(node.state == 0)
                    
                    # Convert to numpy for monotonicity calculation
                    monotonicity = compute_monotonicity(to_numpy(node.state))
                    value = (np.log2(max_tile) / 15.0) + (empty_cells / 16.0) + (monotonicity / 10.0)
                agent.backpropagate(node, value)
                
                # Print progress every 100 simulations
                if (sim_idx + 1) % 100 == 0:
                    print(f"Game {game_idx + 1}/{num_games}, Move {move_count + 1}, Simulation {sim_idx + 1}/{agent.num_simulations}")
            
            # Get visit counts as policy
            visit_counts = np.zeros(4)
            for action, child in root.children.items():
                visit_counts[action] = child.visits
            policy = visit_counts / (np.sum(visit_counts) + 1e-8)  # Add epsilon to avoid division by zero
            
            # Store experience
            game_history.append((state, policy, None))  # Value will be filled later
            
            # Choose move with highest visit count
            action = agent.get_move(state)
            
            # Execute move
            new_board, score, changed = game._move(game.board, action)
            if changed:
                game.board = new_board
                game.add_random_tile()
                move_count += 1
                
                # Get max tile safely, handling both numpy and pytorch tensors
                if isinstance(game.board, torch.Tensor):
                    new_max_tile = game.board.max().item()
                else:
                    new_max_tile = np.max(game.board)
                
                # Print move progress
                print(f"Game {game_idx + 1}/{num_games}, Move {move_count}, Max tile: {new_max_tile}, Action: {action}")
                
                # Update last max tile
                last_max_tile = new_max_tile
            else:
                print(f"Warning: Move {action} did not change the board state!")
                break
        
        # Calculate game result
        final_score = game.get_score()
        
        # Get max tile safely, handling both numpy and pytorch tensors
        if isinstance(game.board, torch.Tensor):
            max_tile = game.board.max().item()
        else:
            max_tile = np.max(game.board)
            
        result = np.log2(max_tile) / 15.0  # Normalize to [0, 1]
        
        # Add experiences with final result
        for state, policy, _ in game_history:
            experiences.append((state, policy, result))
        
        # Print game completion time
        game_time = time.time() - start_time
        print(f"Game {game_idx + 1}/{num_games} completed in {game_time:.2f}s - Max tile: {max_tile}, Score: {final_score}, Total moves: {move_count}")
    
    return experiences

def train_alpha_zero(
    num_iterations: int = 3,
    num_games_per_iteration: int = 3,
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.00005,
    save_interval: int = 1
):
    """Train AlphaZero agent through self-play"""
    # Initialize agent and model
    agent = AlphaZeroAgent(num_simulations=400, c_puct=2.0)
    model = agent.model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Training loop
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Generate training data through self-play
        print("Generating training data...")
        experiences = self_play(agent, num_games_per_iteration)
        
        # Create dataset and dataloader
        dataset = ExperienceDataset(experiences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train model
        print("Training model...")
        model.train()
        for epoch in range(num_epochs):
            total_policy_loss = 0
            total_value_loss = 0
            
            for states, policies, values in dataloader:
                states = states.to(agent.device)
                policies = policies.to(agent.device)
                values = values.to(agent.device)
                
                # Forward pass
                policy_pred, value_pred = model(states)
                
                # Calculate losses
                policy_loss = policy_criterion(policy_pred, policies)
                value_loss = value_criterion(value_pred.squeeze(), values)
                loss = policy_loss + value_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            
            avg_policy_loss = total_policy_loss / len(dataloader)
            avg_value_loss = total_value_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}: Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
        
        # Save model periodically
        if (iteration + 1) % save_interval == 0:
            save_path = f"alpha_zero_model_iter_{iteration + 1}.pth"
            agent.save_model(save_path)
            print(f"Saved model to {save_path}")
        
        # Set model back to eval mode
        model.eval()

if __name__ == "__main__":
    train_alpha_zero() 