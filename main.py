import sys
from game2048 import Game2048
from agent import PPOAgent
from training import train
from test_utils import test_merge_row, test_game_over_condition
import torch
import torch.optim as optim

def main():
    # If test flags are provided, run tests and exit.
    if "--test-merge" in sys.argv:
        test_merge_row()
        sys.exit(0)
    if "--test-gameover" in sys.argv:
        test_game_over_condition()
        sys.exit(0)

    env = Game2048()
    agent = PPOAgent(simple=False, input_channels=16, optimistic=True, Vinit=320000.0)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4)
    
    train(agent, env, optimizer,
          epochs=2500,
          mini_batch_size=32,
          ppo_epochs=12,
          clip_param=0.3,
          gamma=0.99,
          lam=0.95,
          entropy_coef=0.8,
          max_grad_norm=0.5,
          steps_per_update=500)

if __name__ == "__main__":
    main() 