import torch
from torch import nn
import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    # haven't tested added yet
    class ChessDataset(Dataset):
        def __init__(self, fens, evaluations):
            self.fens = fens
            self.evaluations = evaluations

        def __len__(self):
            return len(self.fens)

        def __getitem__(self, idx):
            board = chess.Board(self.fens[idx])
            encoded_board = encode_board(board)
            evaluation = self.evaluations[idx]
            return torch.tensor(encoded_board, dtype=torch.float32), torch.tensor(evaluation, dtype=torch.float32)


def encode_board(board):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    encoded = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        encoded[piece_map[piece.symbol()], row, col] = 1

    return encoded
if __name__ == "__main__":

    # Load CSV
    df = pd.read_csv("chessData.csv", skiprows=range(1, 1000001), nrows=10000000)

    # Extract data
    fens = df["FEN"].values

    df["Evaluation"] = pd.to_numeric(df["Evaluation"], errors="coerce")

    # Drop rows with NaN in Evaluation
    df.dropna(subset=["Evaluation"], inplace=True)

    # Extract Evaluations Again
    fens = df["FEN"].values
    evaluations = df["Evaluation"].values

    from sklearn.model_selection import train_test_split

    # Assuming you have the fens and evaluations already extracted
    fens_train, fens_val, evals_train, evals_val = train_test_split(fens, evaluations, test_size=0.2, random_state=42)

    # Create the training and validation datasets'''
    train_dataset = ChessDataset(fens_train, evals_train)
    val_dataset = ChessDataset(fens_val, evals_val)

    # Create DataLoaders for both datasets
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # Encode and Convert to Tensors
    # Pre-allocate numpy array with the correct shape
    positions = np.zeros((len(fens), 12, 8, 8), dtype=np.float32)

    # Fill the array
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        positions[i] = encode_board(board)

    # Convert to PyTorch tensor
    X = torch.from_numpy(positions)
    y = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)


class ChessBot(nn.Module):
    def __init__(self):
        super(ChessBot, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12*8*8, 512),
            nn.BatchNorm1d(512), #added
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # added
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def train(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
    avg_loss = test_loss / len(dataloader)
    print(f"Test Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model = ChessBot()
    print("loading")
    model.load_state_dict(torch.load("chess_ai_model.pt"))
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Assuming a DataLoader is defined as train_dataloader and test_dataloader
    epochs = 10
    # Track training and validation loss
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        epoch_loss = train(train_dataloader, model, loss_function, optimizer)
        test(train_dataloader, model, loss_function)
        train_loss.append(epoch_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dataloader:  # You will need a validation DataLoader
                pred = model(X)
                loss = loss_function(pred, y.unsqueeze(1))
                running_val_loss += loss.item()
        val_loss.append(running_val_loss / len(val_dataloader))

    # Save model after training
    print("saving")
    torch.save(model.state_dict(), "chess_ai_model.pt")

    # Plot training and validation loss

    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
