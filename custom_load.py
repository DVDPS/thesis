#!/usr/bin/env python
"""
Utility for patching the TransformerPPOAgent load method to handle PyTorch 2.6+ serialization changes.
"""

import torch
from src.thesis.config import device

def patch_agent_load_method(agent):
    """
    Monkey-patch the load method of a TransformerPPOAgent to handle PyTorch 2.6+ serialization changes.
    
    Args:
        agent: The TransformerPPOAgent instance to patch
        
    Returns:
        The patched agent
    """
    original_load = agent.load
    
    def patched_load(path):
        """
        Patched load method that explicitly sets weights_only=False.
        """
        print(f"Loading checkpoint from {path} with weights_only=False for PyTorch 2.6+ compatibility")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        agent.update_count = checkpoint.get('update_count', 0)
        agent.training_stats = checkpoint.get('training_stats', [])
        return agent
    
    # Replace the load method
    agent.load = patched_load
    
    return agent 