import sys
from game2048 import Game2048
from agent import PPOAgent
from training import train
from test_utils import test_merge_row, test_game_over_condition
import torch
import torch.optim as optim
import os
import torch.serialization

def main():
    # Add this at the start of main
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])

    # If test flags are provided, run tests and exit.
    if "--test-merge" in sys.argv:
        test_merge_row()
        sys.exit(0)
    if "--test-gameover" in sys.argv:
        test_game_over_condition()
        sys.exit(0)

    env = Game2048()
    agent = PPOAgent(simple=False, input_channels=16, optimistic=True, Vinit=320000.0)

    initial_lr = 5e-4  
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, weight_decay=1e-4)
    
    # Load the checkpoint if it exists
    checkpoint_path = os.path.join("checkpoints", "best_model.pt")
    start_epoch = 0
    best_running_reward = float('-inf')
    
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        # Don't load optimizer state to use new learning rate
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_running_reward = checkpoint['running_reward']
        print(f"Resuming from epoch {start_epoch} with running reward: {best_running_reward:.2f}")
    
    train(agent, env, optimizer,
          epochs=7000,
          mini_batch_size=32,
          ppo_epochs=12,
          clip_param=0.5,
          gamma=0.99,
          lam=0.95,
          entropy_coef=1.2,  # Increased from 0.8
          max_grad_norm=0.5,
          steps_per_update=500,
          start_epoch=start_epoch,
          best_running_reward=best_running_reward)

if __name__ == "__main__":
    main() 