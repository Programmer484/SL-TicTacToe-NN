import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from game_logic import *


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


def calculate_loss(data_set, net, loss_metric, optimizer=None):
    total_loss = 0
    for board, move_probs in data_set:
        output = net(board)
        loss = loss_metric(output, move_probs)
        if optimizer: # Train the network if an optimizer is provided
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_set)


def calculate_accuracy(data_set, net, probability_threshold=0.1, return_failed_predictions=False):
    """
    A prediction is considered correct if the network's highest probability move
    is within the probability_threshold of the target's highest probability move.

    If return_failed_predictions is True, the function will return a dictionary of failed predictions sorted by game length
    """
    correct = 0
    if return_failed_predictions:
        fails = {i: [] for i in range(9)}
    for board, move_probs in data_set:
        output = net(board)
        best_move = torch.argmax(move_probs)
        best_moves = (move_probs >= move_probs[best_move] - probability_threshold).nonzero().flatten()
        if torch.argmax(output) in best_moves:
            correct += 1
        elif return_failed_predictions:
            game_length = (board != 0).sum().item()
            # Format tensors for easier reading
            board = board.view(3, 3).tolist()
            move_probs = move_probs.view(3, 3).tolist()
            output = output.view(3, 3).tolist()
            output = [[round(val, 3) for val in row] for row in output]
            fails[game_length].append((board, move_probs, output))
    if return_failed_predictions:
        return correct / len(data_set), fails
    return correct / len(data_set)


def save_model(model, hidden_size, train_test_split):
    train_pct = int(train_test_split * 100) # Convert to percentage for model path
    model_path = f'saved_models/model_{hidden_size}h_{train_pct}tr.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def train_model(train_test_split, network_hidden_size, epochs=25, learn_rate=0.1):
    """
    This function trains a neural network to predict optimal moves in Tic-Tac-Toe using
    supervised learning from MCTS-generated training data.
    Args:
        train_test_split (float): Ratio for splitting data into training/testing sets.
            Use 1.0 to use all data for training.
        network_hidden_size (int): Number of neurons in the hidden layer.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        learn_rate (float, optional): Learning rate for SGD optimizer. Defaults to 0.3.
    The function performs the following steps:
        1. Loads and preprocesses MCTS-generated training data
        2. Splits data into training and testing sets
        3. Trains the network using SGD optimizer and MSE loss
        4. Evaluates model performance on both training and test sets
        5. Saves the trained model
        6. Displays failed predictions for analysis
    """
    # Prepare training data
    with open('train_data/mcts200_iterations.pkl', 'rb') as f:
        all_data = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = [(torch.tensor(board.state, dtype=torch.float).flatten().to(device),
                    torch.tensor(move_probs, dtype=torch.float).flatten().to(device))
                    for board, move_probs in all_data]
    random.shuffle(tensor_data)
    split_idx = int(len(tensor_data) * train_test_split)
    train_data = tensor_data[:split_idx]
    test_data = tensor_data[split_idx:]

    # Initialize variables for training loop
    net = TTTNet(hidden_size=network_hidden_size).to(device)
    optimizer = optim.SGD(net.parameters(), lr=learn_rate)
    loss_metric = nn.MSELoss()
    
    # Train the network
    for epoch in tqdm.tqdm(range(epochs)):
        print(f"========= Epoch {epoch} =========")
        train_loss = calculate_loss(train_data, net, loss_metric, optimizer)
        print(f"Train loss: {train_loss}")
        with torch.no_grad():
            train_accuracy = calculate_accuracy(train_data, net)
            print(f"Train accuracy: {train_accuracy}")
            if train_test_split < 1.0: # Skip test set evaluation if using all data for training
                test_loss = calculate_loss(test_data, net, loss_metric)
                print(f"Test loss: {test_loss}")
                test_accuracy = calculate_accuracy(test_data, net)
                print(f"Test accuracy: {test_accuracy}")
     
    # Save the trained model
    save_model(net, network_hidden_size, train_test_split)

    # Save failed predictions
    if train_test_split < 1.0:
        _, fails = calculate_accuracy(train_data, net, return_failed_predictions=True)
    else: 
        _, fails = calculate_accuracy(train_data, net, return_failed_predictions=True)
    save_failed_predictions(fails)


def save_failed_predictions(fails):
    with open('failed_predictions.txt', 'w') as f:
        n = '\n'
        for game_length, predictions in fails.items():
            f.write(f"{n*5}{'*'*20} Fails at length {game_length}:")
            for state, probs, out in predictions:
                f.write(f"\n{'='*7} Board state: {'='*7}\n")
                for row in state:
                    row = ['X' if x == 1 else 'O' if x == -1 else '_' for x in row]
                    f.write(f"{str(row)}\n")
                f.write("\nExpected probabilities:\n")
                for row in probs:
                    f.write(f"{[round(val, 3) for val in row]}\n")
                f.write("\nNetwork output:\n")
                for row in out:
                    f.write(f"{row}\n")


if __name__ == '__main__':
    train_model(0.8, 36, epochs=25, learn_rate=0.1)