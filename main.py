from agents.expectimax import ExpectimaxAgent
from src.thesis.environment.game2048 import Game2048
import numpy as np


def clean_render(game):
    board = game.board
    max_tile = np.max(board)
    score = game.score
    
    print(f"\nScore: {score:,} | Max Tile: {max_tile}")
    
    for row in board:
        row_str = "|"
        for cell in row:
            if cell == 0:
                row_str += "    |"
            else:
                spaces = 4 - len(str(int(cell)))
                row_str += " " * spaces + str(int(cell)) + " |"
        print(row_str)
    print()


if __name__ == "__main__":
    agent = ExpectimaxAgent(depth=5, use_gpu=True)
    game = Game2048()
    state = game.reset()
    
    done = False
    move_counter = 0
    prev_max_tile = 0
    
    while not done:
        action = agent.get_move(state)
        state, reward, done, info = game.step(action)
        move_counter += 1
        
        if move_counter % 10 == 0:
            print(f"Move {move_counter}")
            clean_render(game)
        
        current_max_tile = info['max_tile']
        if current_max_tile > prev_max_tile:
            print(f"\n[!] New max tile reached: {current_max_tile} at move {move_counter}")
            clean_render(game)
            prev_max_tile = current_max_tile
    
    print(f"\nGame Over! Final score: {info['score']:,} | Max tile: {info['max_tile']} | Total moves: {move_counter}")