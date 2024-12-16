from utility import *


def add_bonus(mcts, move, bonus):
    """
    The bonus value affects initial move selection but will be overridden
    if it is not one of the best moves.
    """
    node = mcts.root_node
    node.visit_count = 1
    legal_moves = node.board.empties
    for m in legal_moves:
        child_board = node.board.deepcopy()
        child_board.make_move(m)
        child_node = mcts.Node(node, child_board, m)
        node.children.append(child_node)
        if m == move:
            child_node.total_value = bonus
            child_node.visit_count = 1


board = [[ 1,  0,  0],
         [ 0,  0,  0],
         [ 0, -1,  0]]
empties = [(x, y) for y in range(3) for x in range(3) if board[y][x] == 0]
board = Board(3, 3, 3, 1, board, empties)

# Example 1: No bonus. Always converges on (0, 2) since it
# has the highest initial evaluation from random playouts
# out of the winning moves ((0, 2), (1, 1), and (2, 0)).
mcts = MCTS(board)
move_probs = mcts.search(1000)
print()
for row in move_probs:
    print([round(p, 3) for p in row])

# Example 2: Bonus on one of the winning moves
mcts = MCTS(board)
add_bonus(mcts, (2, 0), 25)
move_probs = mcts.search(1000)
print()
for row in move_probs:
    print([round(p, 3) for p in row])

# Example 3: Bonus on one of the non-winning moves.
# MCTS always self corrects to the best move.
mcts = MCTS(board)
add_bonus(mcts, (1, 0), 50)
move_probs = mcts.search(1000)
print()
for row in move_probs:
    print([round(p, 3) for p in row])