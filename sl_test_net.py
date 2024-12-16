import torch
from utility import *
from sl_train_net import TTTNet


def net_vs_human(board, net):
    while board.outcome != None:
        player = 'X' if board.turn==1 else 'O'
        print(f"\n{'='*7} {player} player's turn {'='*7}")
        print(f"\nBoard:\n{board}")
        
        if board.turn == 1:  # Network's turn (X)
            board_tensor = torch.tensor([board.state[y][x] for x in range(3) for y in range(3)], dtype=torch.float)
            output = net(board_tensor)
            print("\nMove probabilities:")
            for y in range(3):
                print([f"{output[y*3 + x]:.3f}" for x in range(3)])
            move = torch.argmax(output).item()
            move = (move % 3, move // 3)
        else:  # Human's turn (O)
            try:
                move = tuple(map(int, input("Enter move as x,y: ").split(',')))
            except:
                print("Invalid move, try again")
                continue      
        board.make_move(move)
    
    print(f"\nGame over!")
    print(f"\nFinal board:\n{board}")


# Play against the network
net = TTTNet()
net.load_state_dict(torch.load('sl_models/sl_demo_model.pth'))

net_vs_human(Board(3, 3, 3, -1), net)
