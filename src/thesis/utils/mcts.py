"""
Monte Carlo Tree Search (MCTS) implementation for 2048 game.
This module integrates with existing neural network agents to guide the search.
"""

import math
import numpy as np
import torch
import time
import random
from ..environment.game2048 import preprocess_state_onehot, Game2048
from ..config import device

# Constants for MCTS
C_PUCT = 2.0  # Increased exploration constant for better exploration
DIRICHLET_ALPHA = 0.3  # Dirichlet noise parameter for exploration
DIRICHLET_WEIGHT = 0.25  # Weight for Dirichlet noise
MAX_DEPTH = 100  # Increased maximum search depth
VIRTUAL_LOSS = 3.0  # Virtual loss to encourage thread diversity
PROGRESSIVE_WIDENING_BASE = 4  # Base value for progressive widening
PROGRESSIVE_WIDENING_POWER = 0.5  # Power for progressive widening formula


class MCTSNode:
    """
    Monte Carlo Tree Search node representing a game state and its statistics.
    """
    def __init__(self, prior=0.0, parent=None, action=None, max_tile=0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # Maps actions to MCTSNodes
        self.prior = prior  # Prior probability from neural network
        self.parent = parent
        self.action = action  # Action that led to this node
        self.is_expanded = False
        self.max_tile = max_tile  # Track the maximum tile in this state
        self.virtual_loss = 0  # Virtual loss for parallel MCTS
        
    def expanded(self):
        """Check if node has been expanded."""
        return self.is_expanded
        
    def value(self):
        """Average value of all simulations through this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, depth=0):
        """
        Select a child node using the PUCT algorithm with progressive widening.
        Balances exploration and exploitation using priors from neural network.
        
        Args:
            depth: Current search depth for adaptive exploration
        """
        # Find the child that maximizes the UCB score
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # Adaptive exploration - reduce exploration as depth increases
        depth_factor = max(0.5, 1.0 - depth / 50.0)
        exploration_factor = math.sqrt(self.visit_count) * C_PUCT * depth_factor
        
        # Progressive widening - consider only the top N most visited children
        # where N grows with the square root of the parent's visit count
        if self.visit_count > PROGRESSIVE_WIDENING_BASE:
            num_children_to_consider = int(PROGRESSIVE_WIDENING_BASE * (self.visit_count ** PROGRESSIVE_WIDENING_POWER))
            # Sort children by visit count
            sorted_children = sorted(
                self.children.items(), 
                key=lambda item: item[1].visit_count + item[1].prior * 10,  # Consider both visits and prior
                reverse=True
            )
            children_to_consider = dict(sorted_children[:num_children_to_consider])
        else:
            children_to_consider = self.children
        
        for action, child in children_to_consider.items():
            # UCB-like score with prior and virtual loss
            # Virtual loss temporarily reduces the value to discourage other threads from selecting the same node
            adjusted_value = (child.value_sum - child.virtual_loss) / max(child.visit_count, 1)
            score = adjusted_value + exploration_factor * child.prior / (1 + child.visit_count)
            
            # Bonus for nodes with higher max tiles
            tile_bonus = 0.01 * math.log2(max(child.max_tile, 2))
            score += tile_bonus
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def expand(self, actions, priors, max_tile=0):
        """
        Expand the node with all possible actions and their prior probabilities.
        
        Args:
            actions: List of valid actions
            priors: Policy prior probabilities from neural network
            max_tile: Maximum tile value in the current state
        """
        # Add Dirichlet noise to the priors for root exploration
        if self.parent is None and DIRICHLET_WEIGHT > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
            for i, action in enumerate(actions):
                priors[action] = (1 - DIRICHLET_WEIGHT) * priors[action] + DIRICHLET_WEIGHT * noise[i]
        
        # Normalize priors for valid actions
        prior_sum = sum(priors[a] for a in actions)
        for action in actions:
            if prior_sum > 0:
                # Create a child node with normalized prior
                self.children[action] = MCTSNode(
                    prior=priors[action] / prior_sum,
                    parent=self,
                    action=action,
                    max_tile=max_tile
                )
            else:
                # Handle the case where all priors are zero
                self.children[action] = MCTSNode(
                    prior=1.0 / len(actions),
                    parent=self,
                    action=action,
                    max_tile=max_tile
                )
                
        self.is_expanded = True
        self.max_tile = max_tile
    
    def update(self, value):
        """Update statistics with a new simulation result."""
        self.visit_count += 1
        self.value_sum += value
        self.virtual_loss = max(0, self.virtual_loss - VIRTUAL_LOSS)  # Remove virtual loss
        
    def add_virtual_loss(self):
        """Add virtual loss to discourage other threads from selecting this node."""
        self.virtual_loss += VIRTUAL_LOSS
        
    def backup(self, value):
        """Propagate the value up the tree."""
        node = self
        while node is not None:
            node.update(value)
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search implementation for 2048 game.
    Uses a neural network to guide the search and evaluate positions.
    """
    def __init__(self, agent, num_simulations=100, temperature=1.0):
        """
        Initialize MCTS with the given neural network agent.
        
        Args:
            agent: Neural network agent that provides policy and value predictions
            num_simulations: Number of simulations to run for each search
            temperature: Temperature for action selection (higher = more exploration)
        """
        self.agent = agent
        self.num_simulations = num_simulations
        self.temperature = temperature
        
    def search(self, root_state):
        """
        Perform MCTS search from the given root state.
        
        Args:
            root_state: The game state to search from
            
        Returns:
            Policy for each action, represented as visit counts
        """
        # Create a fresh game for simulations
        env = Game2048()
        env.board = np.copy(root_state)
        
        # Initialize root node
        root = MCTSNode()
        
        # Get valid actions and initial policy from agent
        valid_actions = env.get_possible_moves()
        if not valid_actions:
            return None, None  # No valid moves
            
        # Get policy and value from neural network
        state_tensor = torch.tensor(
            preprocess_state_onehot(root_state), 
            dtype=torch.float, 
            device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.agent(state_tensor)
            # Apply softmax to get probabilities
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Get maximum tile in the current state
        max_tile = np.max(root_state)
        
        # Expand root with valid actions and policy priors
        priors = {a: policy[a] for a in range(4)}
        root.expand(valid_actions, priors, max_tile=max_tile)
        
        # Simulation phase - run multiple simulations
        for _ in range(self.num_simulations):
            # Clone the environment for this simulation
            sim_env = Game2048()
            sim_env.board = np.copy(root_state)
            
            # Selection phase - traverse tree to a leaf node
            node = root
            search_path = [node]
            current_depth = 0
            
            # Add virtual loss to the root to encourage diversity
            node.add_virtual_loss()
            
            # Select child nodes until we reach an unexpanded node
            while node.expanded() and current_depth < MAX_DEPTH:  # Increased depth limit
                action, node = node.select_child(depth=current_depth)
                
                # Apply the action to the environment
                if action != -1:
                    _, reward, done, _ = sim_env.step(action)
                    
                    # Update max tile information
                    current_max_tile = np.max(sim_env.board)
                    node.max_tile = max(node.max_tile, current_max_tile)
                    
                    if done:
                        break
                        
                # Add virtual loss to encourage thread diversity
                node.add_virtual_loss()
                
                search_path.append(node)
                current_depth += 1
            
            # Check for end of game
            if sim_env.is_game_over():
                # Game over - use a more sophisticated terminal value
                # Penalize game over, but consider the max tile achieved
                max_tile = np.max(sim_env.board)
                # Logarithmic reward based on max tile
                value = math.log2(max_tile) - 15  # Penalty that scales with max tile
            elif not node.expanded():
                # Expansion phase - expand the node if it's not expanded
                valid_actions = sim_env.get_possible_moves()
                
                if valid_actions:
                    # Get network predictions for this state
                    state_tensor = torch.tensor(
                        preprocess_state_onehot(sim_env.board), 
                        dtype=torch.float, 
                        device=device
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        policy_logits, value_tensor = self.agent(state_tensor)
                        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                        value = value_tensor.item()
                    
                    # Get maximum tile in the current state
                    current_max_tile = np.max(sim_env.board)
                    
                    # Add a bonus based on the maximum tile
                    # This encourages paths that lead to higher tiles
                    tile_bonus = 0.1 * math.log2(current_max_tile) if current_max_tile > 0 else 0
                    value += tile_bonus
                    
                    # Expand the node with network priors
                    priors = {a: policy[a] for a in range(4)}
                    node.expand(valid_actions, priors, max_tile=current_max_tile)
                else:
                    # No valid moves - use a negative value
                    value = -10
            
            # Backup phase - propagate the value up the tree
            for node in reversed(search_path):
                node.backup(value)
        
        # Calculate improved policy based on visit counts
        improved_policy = np.zeros(4, dtype=np.float32)
        
        if self.temperature == 0:
            # Deterministic policy - choose the most visited action
            visits = [child.visit_count for action, child in root.children.items()]
            best_action = list(root.children.keys())[int(np.argmax(visits))]
            improved_policy[best_action] = 1.0
        else:
            # Stochastic policy based on visit counts and temperature
            visits = np.array([child.visit_count for action, child in sorted(root.children.items())])
            actions = np.array(sorted(root.children.keys()))
            
            if sum(visits) > 0:
                # Apply temperature and normalize
                if self.temperature == 1.0:
                    probs = visits / sum(visits)
                else:
                    # Sharpen the distribution with temperature
                    visits = visits ** (1.0 / self.temperature)
                    probs = visits / sum(visits)
                
                # Set policy for each action
                for action, prob in zip(actions, probs):
                    improved_policy[action] = prob
        
        return improved_policy, root

    def get_action(self, state, deterministic=False):
        """
        Get the best action for the given state based on MCTS search.
        
        Args:
            state: Current game state
            deterministic: Whether to select deterministically or sample
            
        Returns:
            Selected action and policy
        """
        # Run MCTS search
        policy, root = self.search(state)
        
        if policy is None:
            return None, None  # No valid moves
        
        # Get valid actions
        valid_actions = list(root.children.keys())
        
        # For high-value states (with tiles >= 256), use deterministic selection
        # This helps maintain structure in the late game
        max_tile = np.max(state)
        if max_tile >= 256:
            deterministic = True
        
        if deterministic or self.temperature < 0.1:
            # Choose the action with highest probability
            action = valid_actions[int(np.argmax([policy[a] for a in valid_actions]))]
        else:
            # Sample from the policy
            action_probs = np.array([policy[a] for a in valid_actions])
            if sum(action_probs) > 0:
                action_probs /= sum(action_probs)
                action_idx = np.random.choice(len(valid_actions), p=action_probs)
                action = valid_actions[action_idx]
            else:
                # Fallback to deterministic if probabilities sum to zero
                action = valid_actions[int(np.argmax([root.children[a].visit_count for a in valid_actions]))]
        
        return action, policy


def mcts_action(agent, state, num_simulations=100, temperature=1.0):
    """
    Utility function to get an action using MCTS with the given agent.
    
    Args:
        agent: Neural network agent
        state: Current game state
        num_simulations: Number of MCTS simulations
        temperature: Temperature for action selection
        
    Returns:
        Selected action
    """
    mcts = MCTS(agent, num_simulations=num_simulations, temperature=temperature)
    action, _ = mcts.get_action(state)
    return action


def analyze_position(agent, state, num_simulations=200):
    """
    Analyze a position using MCTS and return detailed statistics.
    Useful for debugging and understanding the search results.
    
    Args:
        agent: Neural network agent
        state: Game state to analyze
        num_simulations: Number of MCTS simulations
    
    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()
    mcts = MCTS(agent, num_simulations=num_simulations, temperature=1.0)
    policy, root = mcts.search(state)
    
    # Get original network predictions
    state_tensor = torch.tensor(
        preprocess_state_onehot(state), 
        dtype=torch.float, 
        device=device
    ).unsqueeze(0)
    
    with torch.no_grad():
        network_policy, network_value = agent(state_tensor)
        network_policy = torch.softmax(network_policy, dim=1).cpu().numpy()[0]
    
    # Gather statistics for each action
    action_stats = []
    for action in sorted(root.children.keys()):
        child = root.children[action]
        action_stats.append({
            'action': action,
            'visits': child.visit_count,
            'value': child.value(),
            'prior': child.prior,
            'mcts_policy': policy[action],
            'network_policy': network_policy[action],
            'max_tile': child.max_tile
        })
    
    # Sort by visit count (descending)
    action_stats.sort(key=lambda x: x['visits'], reverse=True)
    
    return {
        'time_taken': time.time() - start_time,
        'num_simulations': num_simulations,
        'network_value': network_value.item(),
        'action_stats': action_stats
    } 