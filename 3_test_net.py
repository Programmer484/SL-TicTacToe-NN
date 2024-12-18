import torch
from game_logic import *
import importlib
module = importlib.import_module('2_train_net') # Allows us to import a module name that starts with a digit
TTTNet = getattr(module, 'TTTNet')


class HumanPlayer:
    def move(*args, **kwargs):
        while True:
            try:
                move = tuple(map(int, input("Enter move as x,y: ").split(',')))
                return (move)
            except:
                print("Invalid move, try again")


class NetPlayer:
    def __init__(self, net, deterministic=False):
        self.net = net
        self.deterministic = deterministic

    def move(self, board, print_probs=False):
        board_tensor = torch.tensor(board.state, dtype=torch.float).flatten()
        board_tensor = board_tensor * board.turn
        output = self.net(board_tensor)
        # If a move is illegal, set its probability to 0
        for move in range(9):
            if (move % 3, move // 3) not in board.empties:
                output[move] = 0.0
        if self.deterministic:
            move = torch.argmax(output).item()
        else:
            move = torch.multinomial(output, 1).item()
        if print_probs:
            print("\nMove probabilities:")
            for y in range(3):
                print([f"{output[y*3 + x]:.3f}" for x in range(3)])
        return (move % 3, move // 3)


def play_game(board, player1, player2, print_game=False):
    outcome = None
    illegal_move_count = 0
    while outcome == None:
        if print_game:
            player = 'X' if board.turn==1 else 'O'
            print(f"\n{'='*7} {player} player's turn {'='*7}")
            print(f"\nBoard:\n{board}")
        if board.turn == 1:
            move = player1.move(board, print_probs=print_game)
        else:
            move = player2.move(board, print_probs=print_game)
        try:
            board.make_move(move)
        except:
            if illegal_move_count < 3:
                print("Illegal move, try again")
                illegal_move_count += 1
                continue
            else:
                print("Too many illegal moves, aborting game")
                return None
        outcome = board.outcome(move)
    if print_game:
        print(f"\nGame over!")
        print(f"\nFinal board:\n{board}")
    return outcome


def net_vs_net(net1_player, net2_player, iterations):
    net1_wins = 0
    net2_wins = 0
    net_info = [[net1_player, net1_wins], [net2_player, net2_wins]]
    for i in range(iterations):
        outcome = play_game(Board(3, 3, 3, 1), net_info[0][0], net_info[1][0], print_game=False)
        if outcome == 1:
            net_info[0][1] += 1
        elif outcome == -1:
            net_info[1][1] += 1
        net_info.reverse()  # Switch who goes first
    return net_info


def load_model(hidden_size, train_split):
    train_pct = int(train_split * 100) # Convert to percentage for model path
    model_path = f'saved_models/model_{hidden_size}h_{train_pct}tr.pth'
    net = TTTNet(hidden_size=hidden_size)
    net.load_state_dict(torch.load(model_path))
    return net


play_game(Board(3, 3, 3, 1), NetPlayer(load_model(36, 0.8), deterministic=True), HumanPlayer(), print_game=True)