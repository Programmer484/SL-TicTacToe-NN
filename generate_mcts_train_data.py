"""
Returns a list of (board, best_move) pairs, where board is a
Board object and best_move is a 3x3 nested list of move probabilities
"""
import itertools
import pickle

import tqdm

from game_logic import *


def valid_turns(board):
    # Input: 1D list
    if board.count(1) == board.count(-1) or board.count(1) - 1  == board.count(-1):
        return True
    return False


def outcome_1D(board):
    win_conditions= [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
    ]
    for condition in win_conditions:
        square = condition[0]
        if board[square] != 0:
            win = True
            token = board[square]
            for square in condition[1:]:
                if board[square] != token:
                    win = False
                    break
            if win == True:
                return True
    return False


def player_turn(board):
    if board.count(1) == board.count(-1):
        return 1
    elif board.count(1) - 1 == board.count(-1):
        return -1  


possible_items = [1, -1, 0]
all_boards = list(list(tup) for tup in itertools.product(possible_items, repeat=9))

valid_boards = [board for board in all_boards if 0 in board]  # Ensure at least 1 empty square
valid_boards = [board for board in valid_boards if valid_turns(board)]
valid_boards = [board for board in valid_boards if not outcome_1D(board)]
boards_with_turn = [(board, player_turn(board)) for board in valid_boards]

# Convert flat boards to Board objects for MCTS to work with
boards = []
for flat_board, turn in boards_with_turn:
    flat_board = [flat_board[i] * turn for i in range(9)]  # Ensure player to move is always represented by '1' token
    board = [flat_board[i:i+3] for i in range(0, 9, 3)]
    empties = [(x, y) for y in range(3) for x in range(3) if board[y][x] == 0]
    board = Board(3, 3, 3, 1, board, empties)
    boards.append(board)

# Evaluate boards with MCTS
training_data = []
iterations = 200
for board in tqdm.tqdm(boards, desc="Game Generation Progress"):
    mcts = MCTS(board)
    move_probs = mcts.search(iterations, 10)
    training_data.append((board, move_probs))

# Save training data
with open(f'train_data/mcts{iterations}_iterations.pkl', 'wb') as f:
    pickle.dump(training_data, f)
 
# Print a random sample from training data
for i in range (1, 10):
    sample = random.choice(training_data)
    print("\nPlayer to move:")
    print("X" if sample[0].turn == 1 else "O")
    print("Board state:")
    print(sample[0])
    print("Move probabilities:")
    for row in sample[1]:
        print([round(p, 3) for p in row])