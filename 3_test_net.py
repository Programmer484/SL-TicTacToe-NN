import importlib
import torch
import tqdm
from abc import ABC, abstractmethod


from game_logic import *

# Allows us to import a module name that starts with a digit
module = importlib.import_module('2_train_net')
TTTNet = getattr(module, 'TTTNet')


class Player(ABC):
    @abstractmethod
    def move(self, board, **kwargs):
        """Return a move in the form of (x, y) coordinates.
        
        Args:
            board: The current game board
            **kwargs: Additional arguments specific to player types
            
        Returns:
            tuple: (x, y) coordinates of the chosen move
        """
        pass


class HumanPlayer(Player):
    def move(self, board, **kwargs):
        while True:
            try:
                move = tuple(map(int, input("Enter move as x,y: ").split(',')))
                return move
            except:
                print("Invalid move, try again")


class NetPlayer(Player):
    def __init__(self, net, deterministic=False):
        self.net = net
        self.deterministic = deterministic

    def move(self, board, print_probs=False):
        board_tensor = torch.tensor(board.state, dtype=torch.float).flatten()
        board_tensor = board_tensor * board.turn
        move_probs = self.net(board_tensor)

        if print_probs:
            print("\nNetwork probabilities:")
            for y in range(3):
                print([f"{move_probs[y*3 + x]:.3f}" for x in range(3)])

        # If a move is illegal, set its probability to 0
        for move in range(9):
            if (move % 3, move // 3) not in board.empties:
                move_probs[move] = 0.0    
        
        if self.deterministic:
            best_move_idx = torch.argmax(move_probs).item()
        else:
            best_move_idx = torch.multinomial(move_probs, 1).item()
        best_move = (best_move_idx % 3, best_move_idx // 3)

        return best_move


class RandomPlayer(Player):
    def move(self, board, **kwargs):
        return random.choice(board.empties)


class MCTSPlayer(Player):
    def __init__(self, iterations, deterministic=False):
        self.deterministic = deterministic
        self.iterations = iterations

    def move(self, board, print_probs=False):
        mcts = MCTS(board)
        move_probs = mcts.search(self.iterations)
        move_probs = torch.tensor(move_probs).flatten()

        if print_probs:
            print("\nMCTS probabilities:")
            for row in move_probs:
                print([f"{prob:.3f}" for prob in row])
        
        if self.deterministic:
            best_move_idx = torch.argmax(move_probs).item()
        else:
            best_move_idx = torch.multinomial(move_probs, 1).item()
        best_move = (best_move_idx % 3, best_move_idx // 3)

        return best_move


def play_game(board, player_a, player_b, print_game=False):
    outcome = None
    illegal_move_count = 0
    while outcome == None:
        if print_game:
            player = 'X' if board.turn==1 else 'O'
            print(f"\n{'='*7} {player} player's turn {'='*7}")
            print(f"\nBoard:\n{board}")
        player = player_a if board.turn == 1 else player_b
        move = player.move(board, print_probs=print_game)
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


def run_match_series(player_a, player_b, num_games, board_params=(3, 3, 3)):
    """
    Run a series of games between two players, where each player gets an equal
    opportunity to play first. The function keeps track of wins for each player across all games.
    """
    # Initialize win counters for both players
    player_a_wins = 0
    player_b_wins = 0
    player_list = [[player_a, player_a_wins], 
                   [player_b, player_b_wins]]
    
    # Play specified number of games, alternating who goes first
    for game in tqdm.tqdm(range(num_games)):
        current_first = player_list[0][0]
        current_second = player_list[1][0]
        outcome = play_game(Board(*board_params, 1), current_first, current_second, print_game=False)
        
        # Update win counts based on game outcome
        if outcome == 1:
            player_list[0][1] += 1    # First player won
        elif outcome == -1:
            player_list[1][1] += 1    # Second player won

        player_list.reverse()
    
    # Return in original order: player_a first, player_b second
    return player_list if player_list[0][0] == player_a else player_list.reverse()


def load_model(hidden_size, train_split):
    train_pct = int(train_split * 100) # Convert to percentage for model path
    model_path = f'saved_models/model_{hidden_size}h_{train_pct}tr.pth'
    net = TTTNet(hidden_size=hidden_size)
    net.load_state_dict(torch.load(model_path))
    return net


print(run_match_series(NetPlayer(load_model(36, 0.8), deterministic=True), RandomPlayer(), 1000))
print(run_match_series(NetPlayer(load_model(36, 0.8), deterministic=False), MCTSPlayer(1000, deterministic=False), 1000))