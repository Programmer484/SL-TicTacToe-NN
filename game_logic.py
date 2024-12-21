import math
import random


"""Board"""
class Board():
    def __init__(self, width, height, win_length, turn, state=None, empties=None):
        self.width = width
        self.height = height
        self.win_length = win_length
        self.turn = turn
        if state == None:
            self.state = tuple([0 for _ in range (width)] for _ in range(height))
            self.empties = [(x, y) for y in range (height) for x in range(width)]
        else:  # Ensures a deepcopy is created
            self.state = tuple(map(list, state))
            self.empties = [*empties]
    
    def __str__(self):
        symbols = {1: "X", -1: "O", 0: "_"}
        return "\n".join(str([symbols[token] for token in row]) for row in self.state)
    
    def deepcopy(self):
        return Board(self.width, self.height, self.win_length, self.turn, self.state, self.empties)
        
    def out_of_bounds(self, square):
        x, y = square
        return (x < 0 or             # Left edge
                y < 0 or             # Top edge
                x >= self.width or   # Right edge
                y >= self.height)    # Bottom edge

    def outcome(self, last_move):
        player = self.state[last_move[1]][last_move[0]]
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1),          ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]
        for dx, dy in directions:
            line_length = 1
            # Check in one direction
            x, y = last_move[0] + dx, last_move[1] + dy
            while not self.out_of_bounds((x, y)) and self.state[y][x] == player:
                line_length += 1
                x += dx
                y += dy
            # Check in the opposite direction
            x, y = last_move[0] - dx, last_move[1] - dy
            while not self.out_of_bounds((x, y)) and self.state[y][x] == player:
                line_length += 1
                x -= dx
                y -= dy
            if line_length >= self.win_length:
                return player
        if len(self.empties) == 0:
            return 0
        return None

    def make_move(self, move_coords: tuple):
        try:
            self.empties.remove(move_coords)
            self.state[move_coords[1]][move_coords[0]] = self.turn
            self.turn *= -1
        except:
            raise ValueError("Illegal move")
    

"""MCTS"""
class MCTS:
    class Node:
        def __init__(self, parent, board, move):
            self.parent = parent
            self.children = []
            self.visit_count = 0
            self.total_value = 0
            self.board = board
            self.move = move

        def is_terminal(self):
            return self.board.outcome(self.move) != None

        def is_leaf(self):
            return self.children == []

    def __init__(self, board):
        self.root_node = self.Node(None, board, (-2,-2))
        self.c = 1.4142
    
    def uct(self, parent, node, player_num):
        if node.visit_count == 0:
            return float("inf")
        else:
            return (player_num * node.total_value/node.visit_count) + self.c * math.sqrt(math.log(parent.visit_count) / node.visit_count)
    
    def select(self, node):
        if node.is_leaf():
            return node
        else:
            chosen_node = max(node.children, key=lambda child: self.uct(node, child, node.board.turn))
            return self.select(chosen_node)

    def expand(self, node):
        legal_moves = node.board.empties
        for m in legal_moves:
            child_board = node.board.deepcopy()
            child_board.make_move(m)
            child_node = self.Node(node, child_board, m)
            node.children.append(child_node)

    def evaluate(self, node, playouts):
        outcome = node.board.outcome(node.move)
        if outcome != None:
            return outcome
        else:
            total_value = 0
            for _ in range (playouts):
                total_value += self.play_random_game(node.board.deepcopy())
            return total_value / playouts

    def play_random_game(self, board):
        game_on = True
        while game_on:
            move = random.choice(board.empties)
            board.make_move(move)
            value = board.outcome(move)
            if value != None:
                return value

    def backpropagate(self, node, evaluation):
        node.visit_count += 1
        node.total_value += evaluation
        if node.parent != None:
            self.backpropagate(node.parent, evaluation)

    def search(self, iterations, playouts=20):
        for _ in range(iterations):
            node = self.select(self.root_node)
            if node.visit_count == 0 or node.is_terminal():
                node_eval = self.evaluate(node, playouts)
                self.backpropagate(node, node_eval)
            else:
                self.expand(node)
                child_node = random.choice(node.children)
                child_node_eval = self.evaluate(child_node, playouts)
                self.backpropagate(child_node, child_node_eval)
        move_probs = [[0 for _ in range(self.root_node.board.width)] for _ in range(self.root_node.board.height)]
        for child_node in self.root_node.children:
            prob = child_node.visit_count / self.root_node.visit_count
            move_probs[child_node.move[1]][child_node.move[0]] = prob
        return move_probs