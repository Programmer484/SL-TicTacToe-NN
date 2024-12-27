"""
Returns a list of (board, best_move) pairs, where board is a
Board object and best_move is a 3x3 nested list of move probabilities
"""
import itertools
import pickle

import tqdm

from game_logic import *


def determine_valid_boards():
    def outcome_1D(board):
        wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
        return any(board[a] == board[b] == board[c] != 0 for a,b,c in wins)

    # Possible items in a board: 1 (X), -1 (O), 0 (empty)
    possible_items = [1, -1, 0]
    all_boards = list(list(tup) for tup in itertools.product(possible_items, repeat=9))

    valid_boards = [board for board in all_boards if 0 in board]  # Ensure at least 1 empty square
    # Keep only boards where sum of pieces is 0 (equal pieces) or 1 (X player has one more piece than O player)
    valid_boards = [board for board in valid_boards if sum(board) in (0, 1)]
    valid_boards = [board for board in valid_boards if not outcome_1D(board)]

    return valid_boards


def generate_mcts_move_probabilities(valid_boards, search_iterations):

    def determine_active_player(board):
        return 1 if sum(board) == 0 else -1
    
    boards_with_turn = [(board, determine_active_player(board)) for board in valid_boards]
    # Convert flat boards to Board objects for MCTS to work with
    board_objects = []
    for flat_board, active_player in boards_with_turn:
        normalized_board = [flat_board[i] * active_player for i in range(9)]  # Ensure player to move is always represented by '1' token
        board_2d = [normalized_board[i:i+3] for i in range(0, 9, 3)]
        empty_positions = [(x, y) for y in range(3) for x in range(3) if board_2d[y][x] == 0]
        board_object = Board(3, 3, 3, 1, board_2d, empty_positions)
        board_objects.append(board_object)

    # Evaluate boards with MCTS
    training_data = []
    for board in tqdm.tqdm(board_objects, desc="Game Generation Progress"):
        mcts_engine = MCTS(board)
        move_probabilities = mcts_engine.search(search_iterations, 10)
        training_data.append((board.state, move_probabilities))
    
    with open(f'train_data/mcts{search_iterations}_iterations.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    return training_data


# Code adapted from the tictactoe position evaluator by @frkns (Github username)
# Code available at: https://colab.research.google.com/drive/1njFJ6u6HGxhlCwA6AUrHoJeKs5Cwzaow
def generate_minimax_move_probabilities(valid_boards):
    memo = {}
    INF = int(2e9)

    def determine_active_player(board):
        return 1 if sum(board) == 0 else -1

    def winner(board):
        for j in range(3):  # row
            if 0 != board[j*3 + 0] == board[j*3 + 1] == board[j*3 + 2]:
                return board[j*3 + 0]
        for i in range(3):  # col
            if 0 != board[0 + i] == board[3 + i] == board[6 + i]:
                return board[0 + i]
        # diags
        if 0 != board[0] == board[4] == board[8]: return board[0]
        if 0 != board[2] == board[4] == board[6]: return board[2]
        return 0

    def gameeval(board) -> int:
        hboard = tuple(board)  # hashable
        if hboard in memo:
            return memo[hboard]

        won = winner(board)
        if won != 0:
            memo[hboard] = won
            return won

        if 0 not in board:  # whole board is filled
            memo[hboard] = 0
            return 0

        # player 1 maximizes and player 2 minimizes
        # assumes valid game state and X always goes first
        player = 1 if sum(board) == 0 else -1  # based on board parity
        best = -INF if player == 1 else INF
        for cell in range(9):
            if board[cell] == 0:  # if empty, test it
                board[cell] = player
                score = gameeval(board)
                best = max(best, score) if player == 1 else min(best, score)
                board[cell] = 0  # backtrack

        memo[hboard] = best
        return best

    def get_optimal_move_probabilities(board):
        player = 1 if sum(board) == 0 else -1
        move_scores = [-2 for _ in range(9)]
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                move_scores[i] = gameeval(board)
                board[i] = 0
        move_scores = [(score * player) if score != -2 else -2 for score in move_scores]  # score of 1 represents a win for the active player
        best_score = max(move_scores)
        move_scores = [1 if score == best_score else 0 for score in move_scores]
        move_probs = [score/sum(move_scores) for score in move_scores]  # convert to probabilities
        return move_probs

    training_data = []
    for board in tqdm.tqdm(valid_boards, desc="Game Generation Progress"):
        move_probs = get_optimal_move_probabilities(board)
        board = [board[i] * determine_active_player(board) for i in range(9)]  # Ensure player to move is always represented by '1' token
        training_data.append((board, move_probs))

    # Convert 1D boards and move probabilities to 2D format
    training_data = [(
        [board[i:i+3] for i in range(0, 9, 3)],
        [move_probs[i:i+3] for i in range(0, 9, 3)]
    ) for board, move_probs in training_data]
    
    with open('train_data/minimax.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    return training_data

valid_boards = determine_valid_boards()
minimax_training_data = generate_minimax_move_probabilities(valid_boards)
mcts_training_data = generate_mcts_move_probabilities(valid_boards, 200)


print("MCTS training data:")
for i in range (1, 5):
    sample = random.choice(mcts_training_data)
    print("Board state:")
    for row in sample[0]:
        print(row)
    print("Move probabilities:")
    for row in sample[1]:
        print([round(p, 3) for p in row])

print("\n\n\n\nMinimax training data:")
for i in range(1, 5):
    sample = random.choice(minimax_training_data)
    print("Board state:")
    for row in sample[0]:
        print(row)
    print("Move probabilities:")
    for row in sample[1]:
        print([round(p, 3) for p in row])