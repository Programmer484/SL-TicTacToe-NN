import pickle
import random
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from utility import *


class TTTNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=36, output_size=9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def calculate_accuracy(data_set, net, acceptable_prob_diff=0.05, return_fails=False):
    """
    A prediction is considered correct if the network's highest probability move
    is within the acceptable_prob_diff of the target's highest probability move.

    If return_fails is True, the function will return a dictionary of failed predictions sorted by game length
    """
    correct = 0
    if return_fails:
        fails = {i: [] for i in range(9)}
    for board, move_probs in data_set:
        output = net(board)
        best_move = torch.argmax(move_probs)
        best_moves = (move_probs >= move_probs[best_move] - acceptable_prob_diff).nonzero().flatten()
        if torch.argmax(output) in best_moves:
            correct += 1
        elif return_fails:
            game_length = (board != 0).sum().item()
            # Format tensors for easier reading
            board = board.view(3, 3).tolist()
            move_probs = move_probs.view(3, 3).tolist()
            output = output.view(3, 3).tolist()
            output = [[round(val, 3) for val in row] for row in output]
            fails[game_length].append((board, move_probs, output))
    if return_fails:
        return correct / len(data_set), fails
    return correct / len(data_set)
    

def calculate_loss(data_set, net, train=False):
    total_loss = 0
    for board, move_probs in data_set:
        output = net(board)
        loss = loss_metric(output, move_probs)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_set)


def print_fails(fails):
    for game_length, failed_predictions in fails.items():
        print(f"\n\n\nFails at length {game_length}:")
        for state, probs, out in failed_predictions:
            # Count pieces to determine player turn (-1 moves on even count, 1 moves on odd count)
            piece_count = sum(sum(1 for x in row if x != 0) for row in state)
            player = 1 if piece_count % 2 == 0 else -1
            print(f"\n{'='*7} Player to move: {player} {'='*7}")
            print("Board state:")
            for row in state:
                print(row)
            print("Expected probabilities:")
            for row in probs:
                print([round(val, 3) for val in row])
            print("Network output:")
            for row in out:
                print(row)


if __name__ == '__main__':
    # Prepare training data
    with open('sl_train_data/sl_train_data_mcts1000.pkl', 'rb') as f:
        all_data = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tensor_data = [(torch.tensor([board.state[y][x] for x in range(3) for y in range(3)], dtype=torch.float).to(device),
                    torch.tensor([move_probs[y][x] for x in range(3) for y in range(3)], dtype=torch.float).to(device))
                    for board, move_probs in all_data]

    random.shuffle(tensor_data)
    split_idx = int(len(tensor_data) * 0.8)
    train_data = tensor_data[:split_idx]
    test_data = tensor_data[split_idx:]

    # Initialize variables for training loop
    net = TTTNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.3)
    loss_metric = nn.MSELoss()
    
    # Train the network
    for epoch in tqdm.tqdm(range(20)):
        print(f"========= Epoch {epoch} =========")
        train_loss = calculate_loss(train_data, net, train=True)
        print(f"Train loss: {train_loss}")
        with torch.no_grad():
            test_loss = calculate_loss(test_data, net)
            print(f"Test loss: {test_loss}")
            train_accuracy = calculate_accuracy(train_data, net)
            print(f"Train accuracy: {train_accuracy}")
            test_accuracy = calculate_accuracy(test_data, net)
            print(f"Test accuracy: {test_accuracy}")
     
    # Save the trained model
    torch.save(net.state_dict(), 'sl_models/sl_model.pth')

    # # Print failed predictions
    # accuracy, fails = calculate_accuracy(train_data, net, 0.1, return_fails=True)
    # print_fails(fails)



