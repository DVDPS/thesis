import numpy as np
from game2048 import Game2048

def test_merge_row():
    print("Testing _merge_row with sample rows:")
    test_rows = [
        np.array([2, 2, 2, 2], dtype=np.int32),
        np.array([2, 2, 4, 4], dtype=np.int32),
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([2, 0, 2, 4], dtype=np.int32)
    ]
    game = Game2048()
    for row in test_rows:
        merged_row, score, changed = game._merge_row(row)
        print("Original row:", row.tolist())
        print("Merged row:  ", merged_row.tolist())
        print("Score gained:", score, "| Changed:", changed)
        print("-" * 40)

def test_game_over_condition():
    print("Testing game over condition:")
    board1 = np.array([
        [2, 2, 4, 8],
        [16, 32, 64, 128],
        [2, 2, 4, 8],
        [16, 32, 64, 128]
    ], dtype=np.int32)
    
    board2 = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 2, 2]
    ], dtype=np.int32)
    
    game = Game2048()
    
    game.board = board1.copy()
    print("Case 1: Board with mergeable tiles")
    print(game.board)
    print("is_game_over:", game.is_game_over())
    print("-" * 40)
    
    game.board = board2.copy()
    print("Case 2: Board without mergeable tiles")
    print(game.board)
    print("is_game_over:", game.is_game_over())
    print("-" * 40)
    # NOTE: For production tests, consider converting these prints into assert statements using a testing framework (e.g. pytest). 