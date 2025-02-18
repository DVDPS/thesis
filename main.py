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
    agent = PPOAgent()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4)
    
    train(agent, env, optimizer,
          epochs=15000,
          mini_batch_size=128,
          ppo_epochs=8,
          clip_param=0.2,
          gamma=0.99,
          lam=0.95,
          entropy_coef=0.8,
          max_grad_norm=0.5,
          steps_per_update=500)

if __name__ == "__main__":
    main() 