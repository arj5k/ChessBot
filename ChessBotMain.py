import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import random



from torch.utils.data import DataLoader, Dataset

# Dataset Class
class ChessDataset(Dataset):
    def __init__(self, fens, evaluations):
        self.fens = fens
        self.evaluations = evaluations

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        # Encode board
        board = chess.Board(self.fens[idx])
        encoded_board = encode_board(board).flatten()  # Flatten to 1D

        # Process FEN features
        processed_fen = preprocess_fen(self.fens[idx])

        # Combine features
        combined_features = torch.cat([torch.tensor(encoded_board, dtype=torch.float32), processed_fen])

        # Load evaluation
        evaluation = self.evaluations[idx]

        return combined_features, torch.tensor(evaluation, dtype=torch.float32)

# FEN Preprocessing
def preprocess_fen(fen):
    parts = fen.split()
    _, turn, castling, en_passant, halfmove, fullmove = parts

    # 1. Turn
    turn_encoded = torch.tensor([1 if turn == 'w' else 0], dtype=torch.float32)

    # 2. Castling Rights
    castling_encoded = torch.tensor([
        int('K' in castling),
        int('Q' in castling),
        int('k' in castling),
        int('q' in castling)
    ], dtype=torch.float32)

    # 3. En Passant
    en_passant_encoded = torch.zeros(64, dtype=torch.float32)
    if en_passant != '-':
        file = ord(en_passant[0]) - ord('a')
        rank = int(en_passant[1]) - 1
        index = rank * 8 + file
        en_passant_encoded[index] = 1

    # 4. Halfmove Clock
    halfmove_normalized = torch.tensor([int(halfmove) / 50], dtype=torch.float32)

    # 5. Fullmove Number
    fullmove_normalized = torch.tensor([min(int(fullmove), 500) / 500], dtype=torch.float32)

    # Combine all features
    return torch.cat([turn_encoded, castling_encoded, en_passant_encoded, halfmove_normalized, fullmove_normalized])

# Board Encoding
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

# Main Execution
if __name__ == "__main__":
    # Load Data
    # Get the total number of rows in the CSV
    n_rows = sum(1 for _ in open("chessData.csv")) - 1  # Subtract 1 for the header
    sample_size = 1000000

    # Generate random row indices to sample
    random_indices = sorted(random.sample(range(n_rows), sample_size))

    # Use a generator to skip and read only the desired rows
    def load_random_rows(file_path, random_indices):
        chunk_size = 10000  # Number of rows to read at a time
        chunk_start = 0  # Tracks the starting index of the current chunk
        selected_rows = []

        # Iterate through the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Check which random indices fall in the current chunk
            for idx in random_indices:
                if chunk_start <= idx < chunk_start + chunk_size:
                    # Get the row relative to the chunk's start
                    relative_idx = idx - chunk_start
                    selected_rows.append(chunk.iloc[relative_idx])

            chunk_start += chunk_size
            # Stop early if we've collected all desired rows
            if len(selected_rows) >= sample_size:
                break

        return pd.DataFrame(selected_rows)


    df = load_random_rows("chessData.csv", random_indices)

    # Preprocess Evaluations
    df["Evaluation"] = pd.to_numeric(df["Evaluation"], errors="coerce")
    df.dropna(subset=["Evaluation"], inplace=True)

    # Extract Data
    fens = df["FEN"].values
    evaluations = df["Evaluation"].values

    # Train-Test Split
    fens_train, fens_val, evals_train, evals_val = train_test_split(
        fens, evaluations, test_size=0.2, random_state=42
    )

    # Create Datasets
    train_dataset = ChessDataset(fens_train, evals_train)
    val_dataset = ChessDataset(fens_val, evals_val)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust dimensions for residual connection
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


    def forward(self, x):
        identity = x
        if self.adjust_channels:
            identity = self.adjust_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add the residual connection
        return self.relu(out)


class ChessBot(nn.Module):
    def __init__(self):
        super(ChessBot, self).__init__()

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # Residual blocks replacing standard conv layers
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
        )

        # Adaptive pooling for feature compression
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for board representation
        self.fc_board = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        # FEN Processing
        self.fc_fen = nn.Sequential(
            nn.Linear(71, 64),  # Input: 71 FEN features
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        # Final MLP for evaluation
        self.fc_combined = nn.Sequential(
            nn.Linear(256 + 64, 256),  # Combine board and FEN features
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 1)  # Output: Single evaluation value
        )

    def forward(self, x_board, x_fen):
        #  Process board with convolutional layers
        x_board = self.initial_conv(x_board)
        x_board = self.res_blocks(x_board)
        x_board = self.global_pool(x_board)
        x_board = x_board.view(x_board.size(0), -1)  # Flatten to (batch_size, 256)
        x_board = self.fc_board(x_board)

        # Process FEN features
        x_fen = self.fc_fen(x_fen)

        # 🔗 Concatenate board & FEN features
        x_combined = torch.cat((x_board, x_fen), dim=1)

        # Final evaluation output
        return self.fc_combined(x_combined)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        # Assuming X is the concatenated tensor of both x_board and x_fen
        x_board = X[:, :12*8*8].view(-1, 12, 8, 8)  # Reshape back to (batch_size, 12, 8, 8)
        x_fen = X[:, 12*8*8:]  # Remaining features are for FEN
        pred = model(x_board, x_fen)
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
            # Assuming X is the concatenated tensor of both x_board and x_fen
            x_board = X[:, :12 * 8 * 8].view(-1, 12, 8, 8)  # Reshape back to (batch_size, 12, 8, 8)
            x_fen = X[:, 12 * 8 * 8:]  # Remaining features are for FEN
            pred = model(x_board, x_fen)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
    avg_loss = test_loss / len(dataloader)
    print(f"Test Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model = ChessBot()
    print("loading")
    #model.load_state_dict(torch.load("chess_ai_model.pt"))
    loss_function = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # Assuming a DataLoader is defined as train_dataloader and test_dataloader
    epochs = 30
    # Track training and validation loss
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training phase
        epoch_loss = train(train_dataloader, model, loss_function, optimizer)
        train_loss.append(epoch_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_loss2 = 0.0
        with torch.no_grad():
            for X, y in val_dataloader:
                # Assuming X is the concatenated tensor of both x_board and x_fen
                x_board = X[:, :12 * 8 * 8].view(-1, 12, 8, 8)  # Reshape back to (batch_size, 12, 8, 8)
                x_fen = X[:, 12 * 8 * 8:]  # Remaining features are for FEN
                pred = model(x_board, x_fen)
                loss = loss_function(pred, y.unsqueeze(1))
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_dataloader)
        val_loss.append(avg_val_loss)

        # Step the scheduler
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        #if current_lr != prev_lr:
        #    print(f"Learning rate changed to: {current_lr}")

        # Print results for the epoch
        print("AdamW:")
        print(f"Training Loss: {epoch_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")
        print("")

        if epoch % 5 == 0:
            print("saving")
            torch.save(model.state_dict(), "chess_ai_model.pt")

    # Save model after training
    print("saving")
    torch.save(model.state_dict(), "chess_ai_model.pt")

    # Plot training and validation loss

    plt.plot(train_loss, label="SGD Training Loss")
    plt.plot(val_loss, label="SGD Validation Loss")
    plt.legend()
    plt.show()



