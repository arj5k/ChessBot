import pygame
import torch
import chess
import copy
import math
#things to work on: also when editing evaluation must cap at 100000 based on this code
pygame.init()

screen = pygame.display.set_mode((740, 740))
clock = pygame.time.Clock()

print("importing chessbot")
from ChessBotMain import ChessBot, encode_board, preprocess_fen

def load_model(model_path="chess_ai_model.pt"):
    """Loads the trained chess AI model."""
    print("loading...")
    model = ChessBot()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("loaded")
    return model

def evaluate_position(fen, model):
    """Takes a FEN string, processes it, and returns an evaluation score."""
    board = chess.Board(fen)

    # Encode board state
    encoded_board = encode_board(board)

    # Ensure encoded_board is a tensor of shape (12, 8, 8)
    if not isinstance(encoded_board, torch.Tensor):
        encoded_board = torch.tensor(encoded_board, dtype=torch.float32)

    if encoded_board.shape != (12, 8, 8):
        raise ValueError(f"Unexpected shape for encoded_board: {encoded_board.shape}, expected (12, 8, 8)")

    # Encode FEN features
    encoded_fen = preprocess_fen(fen)

    # Ensure encoded_fen is a tensor of shape (71,)
    if not isinstance(encoded_fen, torch.Tensor):
        encoded_fen = torch.tensor(encoded_fen, dtype=torch.float32)

    if encoded_fen.shape != (71,):
        raise ValueError(f"Unexpected shape for encoded_fen: {encoded_fen.shape}, expected (71,)")

    # Add batch dimension
    input_board = encoded_board.unsqueeze(0)  # Shape: (1, 12, 8, 8)
    input_fen = encoded_fen.unsqueeze(0)  # Shape: (1, 71)

    print(f"encoded_board shape: {input_board.shape}")  # Debugging
    print(f"encoded_fen shape: {input_fen.shape}")

    # Evaluate the position
    with torch.no_grad():
        evaluation = model(input_board, input_fen)

    print(fen)
    return evaluation.item()

print("calling model")
model = load_model("chess_ai_model.pt")

def array_to_fen(board_array, turn, moved_piece, new_pos):
    """Convert a 2D board array to a simple FEN with default game state values."""
    fen_rows = []
    for row in board_array:
        empty_count = 0
        fen_row = ""
        for square in row:
            if square == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    castling = "-"
    if wKing[0].can_castle() and not (moved_piece == wKing[0]):
        if wRooks[1].can_castle() and not (moved_piece == wRooks[1]) and not (new_pos == [7, 7]) and chessBoard[7][7] == wRooks[1]:
            castling = castling + "K"
        if wRooks[0].can_castle() and not (moved_piece == wRooks[0]) and not (new_pos == [7, 0]) and chessBoard[7][0] == wRooks[0]:
            castling = castling + "Q"
    if bKing[0].can_castle() and not (moved_piece == bKing[0]):
        if bRooks[1].can_castle() and not (moved_piece == bRooks[1]) and not (new_pos == [0, 7]) and chessBoard[0][7] == bRooks[1]:
            castling = castling + "k"
        if bRooks[0].can_castle() and not (moved_piece == bRooks[0]) and not (new_pos == [0, 0]) and chessBoard[0][0] == bRooks[0]:
            castling = castling + "q"
    if not (len(castling) == 1): castling = castling[1:]

    passantSq = "-"
    if moved_piece.get_name() == "pawn" and abs(moved_piece.get_position()[0] - new_pos[0]) == 2:
        passantSq = chr(new_pos[1] + ord('a')) + "" + str(8 - (moved_piece.get_position()[0] + new_pos[0])//2)

    newHalfMove = halfMoveCounter
    if moved_piece.get_name() == "pawn" or chessBoard[new_pos[0]][new_pos[1]]:
        newHalfMove = 0
    else:
        newHalfMove += 1

    fullMove = math.floor(moveCounter / 2) + 1

    return "/".join(fen_rows) + f" {turn} {castling} {passantSq} {newHalfMove} {fullMove}"

def chessBoard_to_array(chess_Board):
    """Converts a chessBoard with Piece objects and None values into a FEN-compatible array."""
    piece_map = {
        "king": "K", "queen": "Q", "rook": "R", "bishop": "B", "night": "N", "pawn": "P"
    }

    board_array = []
    for row in chess_Board:
        new_row = []
        for piece in row:
            if piece is None:
                new_row.append(".")  # Empty square
            else:
                piece_char = piece_map.get(piece.get_name(), "?")  # Convert name to FEN character
                if piece.get_color() == "black":  # Assuming Piece has a color attribute
                    piece_char = piece_char.lower()  # Black pieces are lowercase
                new_row.append(piece_char)
        board_array.append(new_row)
    return board_array

side = True
if side:
    background_image = pygame.image.load("ChessBoard.png")
else:
    background_image = pygame.image.load("ReverseBoard.png")
background_image = pygame.transform.scale(background_image, (640, 640))
background_color = (48, 46, 43)
square_width = background_image.get_width() / 8

# Pre-load all piece images
PIECE_IMAGES = {}

def load_piece_image(color, name):
    key = f"{color[0]}{name[0]}.png"
    if key not in PIECE_IMAGES:
        img = pygame.image.load(key)
        PIECE_IMAGES[key] = pygame.transform.scale(img, (
            background_image.get_width() // 8, background_image.get_height() // 8))
    return PIECE_IMAGES[key]

WHITE_TIMELEFT = 600000
BLACK_TIMELEFT = 600000
frame_count = 0

current_player = 1
promotionState = False
promotionSquare = None
endState = 0
moveCounter = 1
halfMoveCounter = 0

WIDTH = 8
chessBoard = [[None for x in range(WIDTH)] for y in range(WIDTH)]

class Piece:
    def __init__(self, position, color):
        self.position = position
        self.color = color
    def get_position(self):
        return self.position
    def get_color(self):
        return self.color

class Pawn(Piece):
    def __init__(self, position, color, double_move, en_passantable, promotion):
        super().__init__(position, color)
        self.double_move = double_move
        self.en_passantable = en_passantable
        self.image = load_piece_image(color, self.get_name())
        self.promotion = promotion
    def get_image(self):
        return self.image
    def get_name(self):
        return "pawn"
    def can_move_two(self):
        return self.double_move
    def can_be_en_passanted(self):
        return self.en_passantable
    def update_state(self):
        if self.double_move:
            self.double_move = False
            if (self.position[0] == 3 or self.position[0] == 4):
                self.en_passantable = True
        elif self.en_passantable:
            self.en_passantable = False
    def get_promotion(self):
        if self.get_color() == "white":
            return self.get_position()[0] == 0
        else:
            return self.get_position()[0] == 7
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        if (self.double_move):
            if (self.get_color() == "black"):
                if (not (chessBoard[newPosition[0]+1][newPosition[1]] or chessBoard[newPosition[0]+2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            else:
                if (not (chessBoard[newPosition[0] - 1][newPosition[1]] or chessBoard[newPosition[0]-2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (self.get_color() == "black"):
            if (not (chessBoard[newPosition[0]+1][newPosition[1]])):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] + 1]) and chessBoard[newPosition[0] + 1][newPosition[1] + 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0] + 1][newPosition[1] - 1]) and chessBoard[newPosition[0] + 1][newPosition[1] - 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0]][newPosition[1] + 1]) and chessBoard[newPosition[0]][newPosition[1] + 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] + 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] + 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [1])):
                    legalMoves.append(newPosition1 + [1])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0]][newPosition[1] - 1]) and chessBoard[newPosition[0]][newPosition[1] - 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] - 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] - 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [1])):
                    legalMoves.append(newPosition1 + [1])
        else:
            if (not (chessBoard[newPosition[0] - 1][newPosition[1]])):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0] - 1][newPosition[1] + 1]) and chessBoard[newPosition[0] - 1][newPosition[1] + 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] - 1]) and chessBoard[newPosition[0] - 1][newPosition[1] - 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0]][newPosition[1] + 1]) and chessBoard[newPosition[0]][newPosition[1] + 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] + 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] + 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [1])):
                    legalMoves.append(newPosition1 + [1])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0]][newPosition[1] - 1]) and chessBoard[newPosition[0]][newPosition[1] - 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] - 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] - 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [1])):
                    legalMoves.append(newPosition1 + [1])
        return legalMoves

class Night(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "night"
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        vertical = newPosition[1]
        horizontal = newPosition[0]
        if (vertical > 1):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical - 2] and not (chessBoard[horizontal - 1][vertical - 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 2] and not (chessBoard[horizontal + 1][vertical - 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (vertical < 6):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 2] and not (chessBoard[horizontal - 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 2] and not (chessBoard[horizontal + 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (horizontal > 1):
            if (vertical > 0):
                if (chessBoard[horizontal - 2][vertical - 1] and not (chessBoard[horizontal - 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (vertical < 7):
                if (chessBoard[horizontal - 2][vertical + 1] and not (chessBoard[horizontal - 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (horizontal < 6):
            if (vertical > 0):
                if (chessBoard[horizontal + 2][vertical - 1] and not (chessBoard[horizontal + 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (vertical < 7):
                if (chessBoard[horizontal + 2][vertical + 1] and not (chessBoard[horizontal + 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        return legalMoves

class Bishop(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "bishop"
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] - 1
        while (vertical > -1 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = 100
        return legalMoves

class Rook(Piece):
    def __init__(self, position, color, castle):
        super().__init__(position, color)
        self.castle = castle
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "rook"
    def can_castle(self):
        return self.castle
    def update_state(self):
        if self.castle: self.castle = False
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        vertical = newPosition[1] - 1
        horizontal = newPosition[0]
        while (vertical > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = 100
        return legalMoves

class Queen(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "queen"
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] - 1
        while (vertical > -1 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = 100

        vertical = newPosition[1] - 1
        horizontal = newPosition[0]
        while (vertical > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = 100
        return legalMoves

class King(Piece):
    def __init__(self, position, color, castle):
        super().__init__(position, color)
        self.castle = castle
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "king"
    def can_castle(self):
        return self.castle
    def update_state(self):
        if self.castle: self.castle = False
    def return_legal_moves(self):
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        vertical = newPosition[1]
        horizontal = newPosition[0]
        if (vertical > 0):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical - 1] and not (chessBoard[horizontal - 1][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 1] and not (chessBoard[horizontal + 1][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (chessBoard[horizontal][vertical - 1] and not (chessBoard[horizontal][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal][vertical - 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (vertical < 7):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 1] and not (chessBoard[horizontal - 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 1] and not (chessBoard[horizontal + 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (chessBoard[horizontal][vertical + 1] and not (chessBoard[horizontal][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal][vertical + 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (horizontal > 0):
            if (chessBoard[horizontal - 1][vertical] and not (chessBoard[horizontal - 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal - 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (horizontal < 7):
            if (chessBoard[horizontal + 1][vertical] and not (chessBoard[horizontal + 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal + 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if self.castle:
            if (chessBoard[self.get_position()[0]][0] and chessBoard[self.get_position()[0]][0].get_name() == "rook" and chessBoard[self.get_position()[0]][0].can_castle()):
                if (not chessBoard[self.get_position()[0]][1] and not chessBoard[self.get_position()[0]][2] and not chessBoard[self.get_position()[0]][3]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = self.get_position()[0]
                    newPosition1[1] = 2
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [2])):
                        legalMoves.append(newPosition1 + [2])
            if (chessBoard[self.get_position()[0]][7] and chessBoard[self.get_position()[0]][7].get_name() == "rook" and chessBoard[self.get_position()[0]][7].can_castle()):
                if (not chessBoard[self.get_position()[0]][6] and not chessBoard[self.get_position()[0]][5]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = self.get_position()[0]
                    newPosition1[1] = 6
                    if (not checkChecked(self.get_position(), newPosition1, self.get_color(), [3])):
                        legalMoves.append(newPosition1 + [3])
        return legalMoves

def checkChecked(originalLocation, newLocation, color, move_data=None, dumbCastlingVariable=True):
    # Simulate new board
    newBoard = [row[:] for row in chessBoard]
    newBoard[newLocation[0]][newLocation[1]] = newBoard[originalLocation[0]][originalLocation[1]]
    if dumbCastlingVariable: newBoard[originalLocation[0]][originalLocation[1]] = None
    if not (move_data == None):
        if (move_data[0] == 1):
            if color == "white":
                newBoard[newLocation[0] + 1][newLocation[1]] = None
            else:
                newBoard[newLocation[0] - 1][newLocation[1]] = None
        elif (move_data[0] == 2):
            if (checkChecked(originalLocation, originalLocation, color, None, False) or checkChecked(originalLocation, [originalLocation[0], 3], color)):
                return True
        elif (move_data[0] == 3):
            if (checkChecked(originalLocation, originalLocation, color, None, False) or checkChecked(originalLocation, [originalLocation[0], 5], color)):
                return True
    if color == "white":
        if newBoard[newLocation[0]][newLocation[1]].get_name() == "king":
            newPosition = newLocation
        else: newPosition = wKing[0].get_position()
        # Pawn
        if (newPosition[1] != 7 and newPosition[0] != 0 and (newBoard[newPosition[0] - 1][newPosition[1] + 1]) and newBoard[newPosition[0] - 1][newPosition[1] + 1].get_color() == "black" and newBoard[newPosition[0] - 1][newPosition[1] + 1].get_name() == "pawn"):
            return True
        if (newPosition[1] != 0 and newPosition[0] != 0 and (newBoard[newPosition[0] - 1][newPosition[1] - 1]) and newBoard[newPosition[0] - 1][newPosition[1] - 1].get_color() == "black" and newBoard[newPosition[0] - 1][newPosition[1] - 1].get_name() == "pawn"):
            return True
    else:
        if newBoard[newLocation[0]][newLocation[1]].get_name() == "king":
            newPosition = newLocation
        else: newPosition = bKing[0].get_position()
        # Pawn
        if (newPosition[1] != 7 and newPosition[0] != 7 and (newBoard[newPosition[0] + 1][newPosition[1] + 1]) and newBoard[newPosition[0] + 1][newPosition[1] + 1].get_color() == "white" and newBoard[newPosition[0] + 1][newPosition[1] + 1].get_name() == "pawn"):
            return True
        if (newPosition[1] != 0 and newPosition[0] != 7 and (newBoard[newPosition[0] + 1][newPosition[1] - 1]) and newBoard[newPosition[0] + 1][newPosition[1] - 1].get_color() == "white" and newBoard[newPosition[0] + 1][newPosition[1] - 1].get_name() == "pawn"):
            return True
    # Diagonals
    vertical = newPosition[1] - 1
    horizontal = newPosition[0] - 1
    while (vertical > -1 and horizontal > -1):
        if not (newBoard[horizontal][vertical]):
            vertical -= 1
            horizontal -= 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "bishop" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            vertical = -100
    vertical = newPosition[1] + 1
    horizontal = newPosition[0] - 1
    while (vertical < 8 and horizontal > -1):
        if not (newBoard[horizontal][vertical]):
            vertical += 1
            horizontal -= 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "bishop" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            horizontal = -100
    vertical = newPosition[1] - 1
    horizontal = newPosition[0] + 1
    while (vertical > -1 and horizontal < 8):
        if not (newBoard[horizontal][vertical]):
            vertical -= 1
            horizontal += 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "bishop" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            vertical = -100
    vertical = newPosition[1] + 1
    horizontal = newPosition[0] + 1
    while (vertical < 8 and horizontal < 8):
        if not (newBoard[horizontal][vertical]):
            vertical += 1
            horizontal += 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "bishop" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            horizontal = 100
    # Files
    vertical = newPosition[1] - 1
    horizontal = newPosition[0]
    while (vertical > -1):
        if not (newBoard[horizontal][vertical]):
            vertical -= 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "rook" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            vertical = -100
    vertical = newPosition[1] + 1
    while (vertical < 8):
        if not (newBoard[horizontal][vertical]):
            vertical += 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "rook" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            vertical = 100
    vertical = newPosition[1]
    horizontal = newPosition[0] - 1
    while (horizontal > -1):
        if not (newBoard[horizontal][vertical]):
            horizontal -= 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "rook" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            horizontal = -100
    horizontal = newPosition[0] + 1
    while (horizontal < 8):
        if not (newBoard[horizontal][vertical]):
            horizontal += 1
        else:
            if (not (newBoard[horizontal][vertical].get_color() == color) and (newBoard[horizontal][vertical].get_name() == "rook" or newBoard[horizontal][vertical].get_name() == "queen")):
                return True
            horizontal = 100
    # Knights
    vertical = newPosition[1]
    horizontal = newPosition[0]
    if (vertical > 1):
        if (horizontal > 0):
            if (newBoard[horizontal - 1][vertical - 2] and not (newBoard[horizontal - 1][vertical - 2].get_color() == color) and newBoard[horizontal - 1][vertical - 2].get_name() == "night"):
                return True
        if (horizontal < 7):
            if (newBoard[horizontal + 1][vertical - 2] and not (newBoard[horizontal + 1][vertical - 2].get_color() == color) and newBoard[horizontal + 1][vertical - 2].get_name() == "night"):
                return True
    if (vertical < 6):
        if (horizontal > 0):
            if (newBoard[horizontal - 1][vertical + 2] and not (newBoard[horizontal - 1][vertical + 2].get_color() == color) and newBoard[horizontal - 1][vertical + 2].get_name() == "night"):
                return True
        if (horizontal < 7):
            if (newBoard[horizontal + 1][vertical + 2] and not (newBoard[horizontal + 1][vertical + 2].get_color() == color) and newBoard[horizontal + 1][vertical + 2].get_name() == "night"):
                return True
    if (horizontal > 1):
        if (vertical > 0):
            if (newBoard[horizontal - 2][vertical - 1] and not (newBoard[horizontal - 2][vertical - 1].get_color() == color) and newBoard[horizontal - 2][vertical - 1].get_name() == "night"):
                return True
        if (vertical < 7):
            if (newBoard[horizontal - 2][vertical + 1] and not (newBoard[horizontal - 2][vertical + 1].get_color() == color) and newBoard[horizontal - 2][vertical + 1].get_name() == "night"):
                return True
    if (horizontal < 6):
        if (vertical > 0):
            if (newBoard[horizontal + 2][vertical - 1] and not (newBoard[horizontal + 2][vertical - 1].get_color() == color) and newBoard[horizontal + 2][vertical - 1].get_name() == "night"):
                return True
        if (vertical < 7):
            if (newBoard[horizontal + 2][vertical + 1] and not (newBoard[horizontal + 2][vertical + 1].get_color() == color) and newBoard[horizontal + 2][vertical + 1].get_name() == "night"):
                return True
    # King
    vertical = newPosition[1]
    horizontal = newPosition[0]
    if (vertical > 0):
        if (horizontal > 0):
            if (newBoard[horizontal - 1][vertical - 1] and not (newBoard[horizontal - 1][vertical - 1].get_color() == color) and (newBoard[horizontal - 1][vertical - 1].get_name() == "king")):
                return True
        if (horizontal < 7):
            if (newBoard[horizontal + 1][vertical - 1] and not (newBoard[horizontal + 1][vertical - 1].get_color() == color) and (newBoard[horizontal + 1][vertical - 1].get_name() == "king")):
                return True
        if (newBoard[horizontal][vertical - 1] and not (newBoard[horizontal][vertical - 1].get_color() == color) and (newBoard[horizontal][vertical - 1].get_name() == "king")):
            return True
    if (vertical < 7):
        if (horizontal > 0):
            if (newBoard[horizontal - 1][vertical + 1] and not (newBoard[horizontal - 1][vertical + 1].get_color() == color) and (newBoard[horizontal - 1][vertical + 1].get_name() == "king")):
                return True
        if (horizontal < 7):
            if (newBoard[horizontal + 1][vertical + 1] and not (newBoard[horizontal + 1][vertical + 1].get_color() == color) and (newBoard[horizontal + 1][vertical + 1].get_name() == "king")):
                return True
        if (newBoard[horizontal][vertical + 1] and not (newBoard[horizontal][vertical + 1].get_color() == color) and (newBoard[horizontal][vertical + 1].get_name() == "king")):
            return True
    if (horizontal > 0):
        if (newBoard[horizontal - 1][vertical] and not (newBoard[horizontal - 1][vertical].get_color() == color) and (newBoard[horizontal - 1][vertical].get_name() == "king")):
            return True
    if (horizontal < 7):
        if (newBoard[horizontal + 1][vertical] and not (newBoard[horizontal + 1][vertical].get_color() == color) and (newBoard[horizontal + 1][vertical].get_name() == "king")):
            return True
    return False

def updateChessPiece(piece, newLocation, updating=False, move_data=None, computer=-1):
    chessBoard[piece.get_position()[0]][piece.get_position()[1]] = None
    chessBoard[newLocation[0]][newLocation[1]] = piece
    piece.position = newLocation
    if (updating and (piece.get_name() == "pawn" or piece.get_name() == "rook" or piece.get_name() == "king")):
        piece.update_state()
        if not (move_data == None) and not (move_data[0] == 0):
            if (move_data[0] == 1):
                if piece.get_color() == "white":
                    chessBoard[piece.get_position()[0] + 1][piece.get_position()[1]] = None
                else:
                    chessBoard[piece.get_position()[0] - 1][piece.get_position()[1]] = None
            elif (move_data[0] == 2):
                updateChessPiece(chessBoard[piece.get_position()[0]][0], [piece.get_position()[0], 3], True)
            elif (move_data[0] == 3):
                updateChessPiece(chessBoard[piece.get_position()[0]][7], [piece.get_position()[0], 5], True)
        if piece.get_name() == "pawn" and piece.get_promotion():
            if computer == -1:
                global promotionState
                promotionState = True
                global promotionSquare
                promotionSquare = newLocation
                global button
                button = board_pos[0]
                global current_player
                current_player = 1 if current_player == 2 else 2
            else:
                if computer == 0:
                    wPromotions.append(Queen(newLocation, "white"))
                    updateChessPiece(wPromotions[-1], newLocation)
                elif computer == 1:
                    wPromotions.append(Night(newLocation, "white"))
                    updateChessPiece(wPromotions[-1], newLocation)
                elif computer == 2:
                    wPromotions.append(Rook(newLocation, "white", False))
                    updateChessPiece(wPromotions[-1], newLocation)
                elif computer == 3:
                    wPromotions.append(Bishop(newLocation, "white"))
                    updateChessPiece(wPromotions[-1], newLocation)
                elif computer == 7:
                    bPromotions.append(Bishop(newLocation, "black"))
                    updateChessPiece(bPromotions[-1], newLocation)
                elif computer == 6:
                    bPromotions.append(Rook(newLocation, "black", False))
                    updateChessPiece(bPromotions[-1], newLocation)
                elif computer == 5:
                    bPromotions.append(Night(newLocation, "black"))
                    updateChessPiece(bPromotions[-1], newLocation)
                elif computer == 4:
                    bPromotions.append(Queen(newLocation, "black"))
                    updateChessPiece(bPromotions[-1], newLocation)

bPawns = [None] * 8
wPawns = [None] * 8
for z in range(WIDTH):
    bPawns[z] = Pawn([1, z], "black", True, False, False)
    wPawns[z] = Pawn([6, z], "white", True, False, False)

# Place rooks
bRooks = [None] * 2
wRooks = [None] * 2
bRooks[0] = Rook([0, 0], "black", True)
bRooks[1] = Rook([0, 7], "black", True)
wRooks[0] = Rook([7, 0], "white", True)
wRooks[1] = Rook([7, 7], "white", True)

# Place knights
bNights = [None] * 2
wNights = [None] * 2
bNights[0] = Night([0, 1], "black")
bNights[1] = Night([0, 6], "black")
wNights[0] = Night([7, 1], "white")
wNights[1] = Night([7, 6], "white")

# Place bishops
bBishops = [None] * 2
wBishops = [None] * 2
bBishops[0] = Bishop([0, 2], "black")
bBishops[1] = Bishop([0, 5], "black")
wBishops[0] = Bishop([7, 2], "white")
wBishops[1] = Bishop([7, 5], "white")

# Place queens
bQueen = [None] * 1
wQueen = [None] * 1
bQueen[0] = Queen([0, 3], "black")
wQueen[0] = Queen([7, 3], "white")

# Place kings
bKing = [None] * 1
wKing = [None] * 1
bKing[0] = King([0, 4], "black", True)
wKing[0] = King([7, 4], "white", True)

# Prepare additional promotion pieces
wPromotions = []
bPromotions = []

for element in bPawns: updateChessPiece(element, element.get_position())
for element in wPawns: updateChessPiece(element, element.get_position())
for element in bRooks: updateChessPiece(element, element.get_position())
for element in wRooks: updateChessPiece(element, element.get_position())
for element in bNights: updateChessPiece(element, element.get_position())
for element in wNights: updateChessPiece(element, element.get_position())
for element in bBishops: updateChessPiece(element, element.get_position())
for element in wBishops: updateChessPiece(element, element.get_position())
for element in bQueen: updateChessPiece(element, element.get_position())
for element in wQueen: updateChessPiece(element, element.get_position())
for element in bKing: updateChessPiece(element, element.get_position())
for element in wKing: updateChessPiece(element, element.get_position())

# Variables for piece selection and movement
selected_i = None
selected_j = None
selected_piece = None
dragging = False
initial_click_pos = None
legal_moves = None
moves_data = None
just_moved = False
button = -1
halfPress = False

def get_board_position(mouse_pos):
    """Convert mouse position to board indices"""
    if 50 <= mouse_pos[0] < 690 and 50 <= mouse_pos[1] < 690:
        if side:
            return (int((mouse_pos[1] - 50) // square_width),
                    int((mouse_pos[0] - 50) // square_width))
        else:
            return (7 - (int((mouse_pos[1] - 50) // square_width)),
                    (7 - (int((mouse_pos[0] - 50) // square_width))))
    return [-1,-1]

def draw_timer(screen, white_time, black_time):
    """
    Draw timers for both players with tenths of a second.
    white_time and black_time should be in milliseconds
    """
    # Font setup
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)

    # Timer box dimensions
    timer_width = 100
    timer_height = 40
    border_width = 2

    # Calculate positions
    board_right_edge = 50 + 650  # 50px initial offset + 650px board width

    # Black timer at top, aligned with board right edge
    black_x = board_right_edge - timer_width
    black_y = 5

    # White timer at bottom, aligned with board right edge
    white_x = board_right_edge - timer_width
    white_y = 695

    # Draw timer backgrounds with borders
    # Black timer background
    pygame.draw.rect(screen, (0, 0, 0), (black_x - border_width, black_y - border_width,
                                         timer_width + 2 * border_width, timer_height + 2 * border_width))
    pygame.draw.rect(screen, (255, 255, 255), (black_x, black_y, timer_width, timer_height))

    # White timer background
    pygame.draw.rect(screen, (0, 0, 0), (white_x - border_width, white_y - border_width,
                                         timer_width + 2 * border_width, timer_height + 2 * border_width))
    pygame.draw.rect(screen, (255, 255, 255), (white_x, white_y, timer_width, timer_height))

    # Convert milliseconds to minutes, seconds, and tenths
    def format_time(milliseconds):
        total_seconds = milliseconds / 1000
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        tenths = int((total_seconds * 10) % 10)  # Changed to tenths instead of hundredths
        return f"{minutes:02d}:{seconds:02d}.{tenths}"  # Only one digit for tenths

    # Render time text
    black_text = font.render(format_time(black_time), True, (0, 0, 0))
    white_text = font.render(format_time(white_time), True, (0, 0, 0))

    # Calculate text position to center it in the timer box
    def center_text(text, box_x, box_y, box_width, box_height):
        text_x = box_x + (box_width - text.get_width()) // 2
        text_y = box_y + (box_height - text.get_height()) // 2
        return text_x, text_y

    if side:
        # Position and draw the text
        black_text_pos = center_text(black_text, black_x, black_y, timer_width, timer_height)
        white_text_pos = center_text(white_text, white_x, white_y, timer_width, timer_height)
    else:
        black_text_pos = center_text(black_text, white_x, white_y, timer_width, timer_height)
        white_text_pos = center_text(white_text, black_x, black_y, timer_width, timer_height)

    screen.blit(black_text, black_text_pos)
    screen.blit(white_text, white_text_pos)

while True:
    frame_count+=1
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not promotionState:
                initial_click_pos = pygame.mouse.get_pos()
                board_pos = get_board_position(initial_click_pos)
                if board_pos == [-1, -1]:
                    if not (selected_piece and not dragging):
                        board_pos = None
                if board_pos:
                    i, j = board_pos
                    if selected_piece is None:
                        # First click - select piece
                        if chessBoard[i][j]:
                            selected_piece = chessBoard[i][j]
                            selected_i = i
                            selected_j = j
                            if endState == 0 and ((selected_piece.get_color() == "white" and current_player == 1 and side) or (selected_piece.get_color() == "black" and current_player == 2 and not side)):
                                legal_moves = selected_piece.return_legal_moves()
                                first_two = [point[:2] for point in legal_moves]  # Extracts the first two elements
                                last_one = [point[2:] for point in legal_moves]  # Extracts the last element as a sublist
                                legal_moves = first_two
                                moves_data = last_one
                            else:
                                legal_moves = []
                                moves_data = []

                    else:
                        # Second click - move piece if it's not a drag operation
                        if not dragging:
                            inefficientVariable = True
                            if (i != selected_i or j != selected_j):
                                # Move the piece
                                if [i,j] in legal_moves:
                                    current_player = 1 if current_player == 2 else 2
                                    if current_player == 2:
                                        for pwn in wPawns:
                                            if pwn.can_be_en_passanted():
                                                pwn.update_state()
                                    else:
                                        for pwn in bPawns:
                                            if pwn.can_be_en_passanted():
                                                pwn.update_state()
                                    if selected_piece.get_name() == "pawn" or chessBoard[i][j]:
                                        halfMoveCounter = 0
                                    else:
                                        halfMoveCounter += 1
                                    updateChessPiece(selected_piece, [i, j], True, moves_data[legal_moves.index([i,j])])
                                    moveCounter = moveCounter + 1
                                    just_moved = True
                                elif (not i == -1) and chessBoard[i][j]:
                                    inefficientVariable = False
                                    selected_piece = chessBoard[i][j]
                                    selected_i = i
                                    selected_j = j
                                    if endState == 0 and ((selected_piece.get_color() == "white" and current_player == 1 and side) or (selected_piece.get_color() == "black" and current_player == 2 and not side)):
                                        legal_moves = selected_piece.return_legal_moves()
                                        first_two = [point[:2] for point in legal_moves]  # Extracts the first two elements
                                        last_one = [point[2:] for point in legal_moves]  # Extracts the last element as a sublist
                                        legal_moves = first_two
                                        moves_data = last_one
                                    else:
                                        legal_moves = []
                                        moves_data = []
                            if (inefficientVariable):
                                selected_piece = None
                                selected_i = None
                                selected_j = None
                                legal_moves = None
                                moves_data = None
            else:
                halfPress = True

        elif event.type == pygame.MOUSEMOTION:
            if not promotionState:
                if selected_piece and not dragging:
                    # Check if mouse has moved enough to start dragging
                    current_pos = pygame.mouse.get_pos()
                    if initial_click_pos:
                        dx = current_pos[0] - initial_click_pos[0]
                        dy = current_pos[1] - initial_click_pos[1]
                        # If mouse has moved more than 5 pixels, start dragging
                        if (dx * dx + dy * dy) > 25:  # 5 pixels squared
                            dragging = True
            else:
                board_pos = get_board_position(pygame.mouse.get_pos())
                if promotionSquare[0] == 0:
                    if board_pos[1] == promotionSquare[1]:
                        if board_pos[0] < 4:
                            button = board_pos[0]
                        else:
                            button = -1
                    else:
                        button = -1
                else:
                    if board_pos[1] == promotionSquare[1]:
                        if board_pos[0] > 3:
                            button = board_pos[0]
                        else:
                            button = -1
                    else:
                        button = -1

        elif event.type == pygame.MOUSEBUTTONUP:
            if not promotionState:
                if dragging:
                    # Handle drag and drop movement
                    board_pos = get_board_position(pygame.mouse.get_pos())
                    if board_pos:
                        new_i, new_j = board_pos
                        if [new_i, new_j] in legal_moves:
                            current_player = 1 if current_player == 2 else 2
                            if current_player == 2:
                                for pwn in wPawns:
                                    if pwn.can_be_en_passanted():
                                        pwn.update_state()
                            else:
                                for pwn in bPawns:
                                    if pwn.can_be_en_passanted():
                                        pwn.update_state()
                            if selected_piece.get_name() == "pawn" or chessBoard[new_i][new_j]:
                                halfMoveCounter = 0
                            else:
                                halfMoveCounter += 1
                            updateChessPiece(selected_piece, [new_i, new_j], True, moves_data[legal_moves.index([new_i,new_j])])
                            moveCounter = moveCounter + 1
                            just_moved = True
                    selected_piece = None
                    selected_i = None
                    selected_j = None
                    dragging = False
                    legal_moves = None
                    moves_data = None
                initial_click_pos = None
            else:
                if (not (button == -1)) and halfPress:
                    promotionState = False
                    current_player = 1 if current_player == 2 else 2
                    if button == 0:
                        wPromotions.append(Queen(promotionSquare, "white"))
                        updateChessPiece(wPromotions[-1], promotionSquare)
                    elif button == 1:
                        wPromotions.append(Night(promotionSquare, "white"))
                        updateChessPiece(wPromotions[-1], promotionSquare)
                    elif button == 2:
                        wPromotions.append(Rook(promotionSquare, "white", False))
                        updateChessPiece(wPromotions[-1], promotionSquare)
                    elif button == 3:
                        wPromotions.append(Bishop(promotionSquare, "white"))
                        updateChessPiece(wPromotions[-1], promotionSquare)
                    elif button == 4:
                        bPromotions.append(Bishop(promotionSquare, "black"))
                        updateChessPiece(bPromotions[-1], promotionSquare)
                    elif button == 5:
                        bPromotions.append(Rook(promotionSquare, "black", False))
                        updateChessPiece(bPromotions[-1], promotionSquare)
                    elif button == 6:
                        bPromotions.append(Night(promotionSquare, "black"))
                        updateChessPiece(bPromotions[-1], promotionSquare)
                    elif button == 7:
                        bPromotions.append(Queen(promotionSquare, "black"))
                        updateChessPiece(bPromotions[-1], promotionSquare)
                halfPress = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Check if Enter is pressed
                bPawns = [None] * 8
                wPawns = [None] * 8
                for z in range(WIDTH):
                    bPawns[z] = Pawn([1, z], "black", True, False, False)
                    wPawns[z] = Pawn([6, z], "white", True, False, False)
                bRooks = [None] * 2
                wRooks = [None] * 2
                bRooks[0] = Rook([0, 0], "black", True)
                bRooks[1] = Rook([0, 7], "black", True)
                wRooks[0] = Rook([7, 0], "white", True)
                wRooks[1] = Rook([7, 7], "white", True)
                bNights = [None] * 2
                wNights = [None] * 2
                bNights[0] = Night([0, 1], "black")
                bNights[1] = Night([0, 6], "black")
                wNights[0] = Night([7, 1], "white")
                wNights[1] = Night([7, 6], "white")
                bBishops = [None] * 2
                wBishops = [None] * 2
                bBishops[0] = Bishop([0, 2], "black")
                bBishops[1] = Bishop([0, 5], "black")
                wBishops[0] = Bishop([7, 2], "white")
                wBishops[1] = Bishop([7, 5], "white")
                bQueen = [None] * 1
                wQueen = [None] * 1
                bQueen[0] = Queen([0, 3], "black")
                wQueen[0] = Queen([7, 3], "white")
                bKing = [None] * 1
                wKing = [None] * 1
                bKing[0] = King([0, 4], "black", True)
                wKing[0] = King([7, 4], "white", True)
                wPromotions = []
                bPromotions = []

                WHITE_TIMELEFT = 600000
                BLACK_TIMELEFT = 600000
                current_player = 1
                promotionState = False
                promotionSquare = None
                endState = 0
                chessBoard = [[None for x in range(WIDTH)] for y in range(WIDTH)]
                moveCounter = 1
                halfMoveCounter = 0

                for element in bPawns: updateChessPiece(element, element.get_position())
                for element in wPawns: updateChessPiece(element, element.get_position())
                for element in bRooks: updateChessPiece(element, element.get_position())
                for element in wRooks: updateChessPiece(element, element.get_position())
                for element in bNights: updateChessPiece(element, element.get_position())
                for element in wNights: updateChessPiece(element, element.get_position())
                for element in bBishops: updateChessPiece(element, element.get_position())
                for element in wBishops: updateChessPiece(element, element.get_position())
                for element in bQueen: updateChessPiece(element, element.get_position())
                for element in wQueen: updateChessPiece(element, element.get_position())
                for element in bKing: updateChessPiece(element, element.get_position())
                for element in wKing: updateChessPiece(element, element.get_position())

                selected_i = None
                selected_j = None
                selected_piece = None
                dragging = False
                initial_click_pos = None
                legal_moves = None
                moves_data = None
                just_moved = False
                button = -1
                halfPress = False

                side = not side
                if side:
                    background_image = pygame.image.load("ChessBoard.png")
                else:
                    background_image = pygame.image.load("ReverseBoard.png")
                background_image = pygame.transform.scale(background_image, (640, 640))
                continue

    # Do logical updates here.
    if just_moved and not promotionState and endState == 0: # Check to see if checkmate or stalemate
        isLegalMove = False
        for i in range(WIDTH):
            for j in range(WIDTH):
                if current_player == 1: # White's turn
                    if chessBoard[i][j] and chessBoard[i][j].get_color() == "white":
                        for square in chessBoard[i][j].return_legal_moves():
                            isLegalMove = True
                            break
                        if isLegalMove: break
                else:
                    if chessBoard[i][j] and chessBoard[i][j].get_color() == "black":
                        for square in chessBoard[i][j].return_legal_moves():
                            isLegalMove = True
                            break
                        if isLegalMove: break
            if isLegalMove: break
        if not isLegalMove:
            if current_player == 1:
                if checkChecked(wKing[0].get_position(), wKing[0].get_position(), "white", None, False):
                    endState = 2
                else:
                    endState = 3
            else:
                if checkChecked(bKing[0].get_position(), bKing[0].get_position(), "black", None, False):
                    endState = 1
                else:
                    endState = 3
        just_moved = False

    # Background setup
    screen.fill(background_color)
    screen.blit(background_image, (50, 50))

    # Print ending message
    if endState > 0:
        font = pygame.font.Font(None, 42)
        if endState == 3: text = font.render("Stalemate!", True, (255, 255, 255))
        elif side:
            if endState == 1: text = font.render("Player Wins!", True, (255, 255, 255))
            else: text = font.render("Chess Bot Wins!", True, (255, 255, 255))
        else:
            if endState == 2: text = font.render("Player Wins!", True, (255, 255, 255))
            else: text = font.render("Chess Bot Wins!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(740 // 2, 25))
        screen.blit(text, text_rect)

    if current_player == 2 and side and endState == 0 and frame_count%200==0:
        bestPiece = None
        bestMove = None
        bestMoveData = None
        bestEval = 100000
        for i in range(WIDTH):
            for j in range(WIDTH):
                if chessBoard[i][j] and chessBoard[i][j].get_color() == "black":
                    for square in chessBoard[i][j].return_legal_moves():
                        first_two = square[:2]  # Extracts the first two elements
                        last_one = square[2:]
                        newBoard = [row[:] for row in chessBoard]
                        newBoard[first_two[0]][first_two[1]] = newBoard[i][j]
                        newBoard[i][j] = None
                        promoting1 = False
                        if newBoard[first_two[0]][first_two[1]].get_name() == "pawn" and first_two[0] == 7:
                            promoting1 = True
                            for loop in range(3):
                                newBoard[first_two[0]][first_two[1]] = None
                                if loop == 0:
                                    bPromotions.append(Queen(first_two, "black"))
                                elif loop == 1:
                                    bPromotions.append(Night(first_two, "black"))
                                elif loop == 2:
                                    bPromotions.append(Rook(first_two, "black", False))
                                newBoard[first_two[0]][first_two[1]] = bPromotions[-1]
                                fen_board = chessBoard_to_array(newBoard)
                                fen_string = array_to_fen(fen_board, "w", chessBoard[i][j], first_two)
                                evaluation = evaluate_position(fen_string, model)
                                if evaluation < bestEval:
                                    bestEval = evaluation
                                    bestMove = first_two
                                    bestMoveData = [10 + loop]
                                    bestPiece = [i, j]
                                bPromotions.pop()
                            newBoard[first_two[0]][first_two[1]] = None
                            bPromotions.append(Bishop(first_two, "black"))
                            newBoard[first_two[0]][first_two[1]] = bPromotions[-1]
                        if (last_one[0] == 1):
                            newBoard[first_two[0] - 1][first_two[1]] = None
                        elif (last_one[0] == 2):
                            newBoard[chessBoard[i][j].get_position()[0]][3] = newBoard[chessBoard[i][j].get_position()[0]][0]
                            newBoard[chessBoard[i][j].get_position()[0]][0] = None
                        elif (last_one[0] == 3):
                            newBoard[chessBoard[i][j].get_position()[0]][5] = newBoard[chessBoard[i][j].get_position()[0]][7]
                            newBoard[chessBoard[i][j].get_position()[0]][7] = None
                        fen_board = chessBoard_to_array(newBoard)
                        fen_string = array_to_fen(fen_board, "w", chessBoard[i][j], first_two)
                        evaluation = evaluate_position(fen_string, model)
                        if evaluation < bestEval:
                            bestEval = evaluation
                            bestMove = first_two
                            bestMoveData = last_one
                            if promoting1: bestMoveData = [13]
                            bestPiece = [i,j]
                        if promoting1: bPromotions.pop()
        current_player = 1 if current_player == 2 else 2
        if current_player == 2:
            for pwn in wPawns:
                if pwn.can_be_en_passanted():
                    pwn.update_state()
        else:
            for pwn in bPawns:
                if pwn.can_be_en_passanted():
                    pwn.update_state()
        if chessBoard[bestPiece[0]][bestPiece[1]].get_name() == "pawn" or chessBoard[bestMove[0]][bestMove[1]]:
            halfMoveCounter = 0
        else:
            halfMoveCounter += 1
        if bestMoveData[0] < 10: updateChessPiece(chessBoard[bestPiece[0]][bestPiece[1]], bestMove, True, bestMoveData)
        else: updateChessPiece(chessBoard[bestPiece[0]][bestPiece[1]], bestMove, True, None, bestMoveData[0] - 6)
        moveCounter = moveCounter + 1
        just_moved = True
        if selected_piece:
            if not chessBoard[selected_i][selected_j] == selected_piece:
                selected_i = None
                selected_j = None
                selected_piece = None
                dragging = False
                initial_click_pos = None
                legal_moves = None
                moves_data = None
            elif (selected_piece.get_color() == "white" and current_player == 1) or (selected_piece.get_color() == "black" and current_player == 2):
                legal_moves = selected_piece.return_legal_moves()
                first_two = [point[:2] for point in legal_moves]  # Extracts the first two elements
                last_one = [point[2:] for point in legal_moves]  # Extracts the last element as a sublist
                legal_moves = first_two
                moves_data = last_one
    elif current_player == 1 and not side and endState == 0 and frame_count%200==0:
        bestPiece = None
        bestMove = None
        bestMoveData = None
        bestEval = -100000
        for i in range(WIDTH):
            for j in range(WIDTH):
                if chessBoard[i][j] and chessBoard[i][j].get_color() == "white":
                    for square in chessBoard[i][j].return_legal_moves():
                        first_two = square[:2]  # Extracts the first two elements
                        last_one = square[2:]
                        newBoard = [row[:] for row in chessBoard]
                        newBoard[first_two[0]][first_two[1]] = newBoard[i][j]
                        newBoard[i][j] = None
                        promoting1 = False
                        if newBoard[first_two[0]][first_two[1]].get_name() == "pawn" and first_two[0] == 0:
                            promoting1 = True
                            for loop in range(3):
                                newBoard[first_two[0]][first_two[1]] = None
                                if loop == 0:
                                    wPromotions.append(Queen(first_two, "white"))
                                elif loop == 1:
                                    wPromotions.append(Night(first_two, "white"))
                                elif loop == 2:
                                    wPromotions.append(Rook(first_two, "white", False))
                                newBoard[first_two[0]][first_two[1]] = wPromotions[-1]
                                fen_board = chessBoard_to_array(newBoard)
                                fen_string = array_to_fen(fen_board, "b", chessBoard[i][j], first_two)
                                evaluation = evaluate_position(fen_string, model)
                                if evaluation > bestEval:
                                    bestEval = evaluation
                                    bestMove = first_two
                                    bestMoveData = [10 + loop]
                                    bestPiece = [i, j]
                                wPromotions.pop()
                            newBoard[first_two[0]][first_two[1]] = None
                            wPromotions.append(Bishop(first_two, "white"))
                            newBoard[first_two[0]][first_two[1]] = wPromotions[-1]
                        if (last_one[0] == 1):
                            newBoard[first_two[0] + 1][first_two[1]] = None
                        elif (last_one[0] == 2):
                            newBoard[chessBoard[i][j].get_position()[0]][3] = newBoard[chessBoard[i][j].get_position()[0]][0]
                            newBoard[chessBoard[i][j].get_position()[0]][0] = None
                        elif (last_one[0] == 3):
                            newBoard[chessBoard[i][j].get_position()[0]][5] = newBoard[chessBoard[i][j].get_position()[0]][7]
                            newBoard[chessBoard[i][j].get_position()[0]][7] = None
                        fen_board = chessBoard_to_array(newBoard)
                        fen_string = array_to_fen(fen_board, "b", chessBoard[i][j], first_two)
                        evaluation = evaluate_position(fen_string, model)
                        if evaluation > bestEval:
                            bestEval = evaluation
                            bestMove = first_two
                            bestMoveData = last_one
                            if promoting1: bestMoveData = [13]
                            bestPiece = [i, j]
                        if promoting1: wPromotions.pop()
        current_player = 1 if current_player == 2 else 2
        if current_player == 2:
            for pwn in wPawns:
                if pwn.can_be_en_passanted():
                    pwn.update_state()
        else:
            for pwn in bPawns:
                if pwn.can_be_en_passanted():
                    pwn.update_state()
        if chessBoard[bestPiece[0]][bestPiece[1]].get_name() == "pawn" or chessBoard[bestMove[0]][bestMove[1]]:
            halfMoveCounter = 0
        else:
            halfMoveCounter += 1
        if bestMoveData[0] < 10: updateChessPiece(chessBoard[bestPiece[0]][bestPiece[1]], bestMove, True, bestMoveData)
        else: updateChessPiece(chessBoard[bestPiece[0]][bestPiece[1]], bestMove, True, None, bestMoveData[0] - 10)
        moveCounter = moveCounter + 1
        just_moved = True
        if selected_piece:
            if not chessBoard[selected_i][selected_j] == selected_piece:
                selected_i = None
                selected_j = None
                selected_piece = None
                dragging = False
                initial_click_pos = None
                legal_moves = None
                moves_data = None
            elif (selected_piece.get_color() == "white" and current_player == 1) or (selected_piece.get_color() == "black" and current_player == 2):
                legal_moves = selected_piece.return_legal_moves()
                first_two = [point[:2] for point in legal_moves]  # Extracts the first two elements
                last_one = [point[2:] for point in legal_moves]  # Extracts the last element as a sublist
                legal_moves = first_two
                moves_data = last_one

    # Highlight selected square if a piece is selected
    if selected_piece:
        if side:
            if selected_i == 7 and selected_j == 7: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + selected_j * square_width, 50 + selected_i * square_width, square_width, square_width))
            elif selected_i == 7: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + selected_j * square_width, 50 + selected_i * square_width, square_width + 1, square_width))
            elif selected_j == 7: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + selected_j * square_width, 50 + selected_i * square_width, square_width, square_width + 1))
            else: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + selected_j * square_width, 50 + selected_i * square_width, square_width + 1, square_width + 1))
        else:
            if selected_i == 0 and selected_j == 0: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - selected_j) * square_width, 50 + (7 - selected_i) * square_width, square_width, square_width))
            elif selected_i == 0: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - selected_j) * square_width, 50 + (7 - selected_i) * square_width, square_width + 1, square_width))
            elif selected_j == 0: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - selected_j) * square_width, 50 + (7 - selected_i) * square_width, square_width, square_width + 1))
            else: pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - selected_j) * square_width, 50 + (7 - selected_i) * square_width, square_width + 1, square_width + 1))

    if current_player == 1:  # White's turn
        if endState == 0: WHITE_TIMELEFT -= clock.get_time()
        if endState == 0 and WHITE_TIMELEFT < 100:
            WHITE_TIMELEFT = 0
            endState = 2
            legal_moves = []
            moves_data = []
    else:  # Black's turn
        if endState == 0: BLACK_TIMELEFT -= clock.get_time()
        if endState == 0 and BLACK_TIMELEFT < 100:
            BLACK_TIMELEFT = 0
            endState = 1
            legal_moves = []
            moves_data = []

    draw_timer(screen, WHITE_TIMELEFT, BLACK_TIMELEFT)

    # Render the graphics here.
    for i in range(WIDTH):
        for j in range(WIDTH):
            if chessBoard[i][j]:
                if (not (dragging and i == selected_i and j == selected_j)):
                    piece = chessBoard[i][j]
                    if (side):
                        x_val = 50 + j * square_width
                        y_val = 50 + i * square_width
                        screen.blit(piece.get_image(), (x_val, y_val))
                    else:
                        x_val = 50 + (7-j) * square_width
                        y_val = 50 + (7-i) * square_width
                        screen.blit(piece.get_image(), (x_val, y_val))

    # Draw dragged piece only if actually dragging
    if dragging and selected_piece:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen.blit(selected_piece.get_image(),
                    (mouse_x - background_image.get_width() // 16,
                     mouse_y - background_image.get_width() // 16))

    if selected_piece:
        for square in legal_moves:
            circle_surface = pygame.Surface((square_width, square_width), pygame.SRCALPHA)
            circle_surface.set_alpha(85)
            if not chessBoard[square[0]][square[1]]:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), 13)
            else:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), square_width / 2,6)
            if (side): screen.blit(circle_surface, (50 + square[1] * square_width, 50 + square[0] * square_width))
            else: screen.blit(circle_surface, (50 + (7 - square[1]) * square_width, 50 + (7 - square[0]) * square_width))

    if promotionState:
        if promotionSquare[0] == 0:
            promB = load_piece_image("w", "b")
            promN = load_piece_image("w", "n")
            promQ = load_piece_image("w", "q")
            promR = load_piece_image("w", "r")
            if side:
                x_val = 50 + promotionSquare[1] * square_width
                y_val = 50 + promotionSquare[0] * square_width
            else:
                x_val = 50 + (7 - promotionSquare[1]) * square_width
                y_val = 50 + (7 - promotionSquare[0]) * square_width
            shadow_surface = pygame.Surface((square_width + 10, 4 * square_width + 5), pygame.SRCALPHA)
            shadow_surface.fill((50, 50, 50, 128))
            if side:
                screen.blit(shadow_surface, (x_val - 5, y_val))
                pygame.draw.rect(screen, (255, 255, 255), (x_val, y_val, square_width, 4 * square_width))
            else:
                screen.blit(shadow_surface, (x_val - 5, y_val - 3 * square_width - 5))
                pygame.draw.rect(screen, (255, 255, 255), (x_val, y_val - 3 * square_width, square_width, 4 * square_width))
            if not (button == -1):
                if side:
                    pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + promotionSquare[1] * square_width, 50 + button * square_width, square_width, square_width))
                else:
                    pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - promotionSquare[1]) * square_width, 50 + (7 - button) * square_width, square_width, square_width))
            screen.blit(promQ, (x_val, y_val))
            if side: y_val += square_width
            else: y_val -= square_width
            screen.blit(promN, (x_val, y_val))
            if side: y_val += square_width
            else: y_val -= square_width
            screen.blit(promR, (x_val, y_val))
            if side: y_val += square_width
            else: y_val -= square_width
            screen.blit(promB, (x_val, y_val))
        else:
            promB = load_piece_image("b", "b")
            promN = load_piece_image("b", "n")
            promQ = load_piece_image("b", "q")
            promR = load_piece_image("b", "r")
            if side:
                x_val = 50 + promotionSquare[1] * square_width
                y_val = 50 + promotionSquare[0] * square_width
            else:
                x_val = 50 + (7 - promotionSquare[1]) * square_width
                y_val = 50 + (7 - promotionSquare[0]) * square_width
            shadow_surface = pygame.Surface((square_width + 10, 4 * square_width + 5), pygame.SRCALPHA)
            shadow_surface.fill((50, 50, 50, 128))
            if side:
                screen.blit(shadow_surface, (x_val - 5, y_val - 3 * square_width - 5))
                pygame.draw.rect(screen, (255, 255, 255), (x_val, y_val - 3 * square_width, square_width, 4 * square_width))
            else:
                screen.blit(shadow_surface, (x_val - 5, y_val))
                pygame.draw.rect(screen, (255, 255, 255), (x_val, y_val, square_width, 4 * square_width))
            if not (button == -1):
                if side:
                    pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + promotionSquare[1] * square_width, 50 + button * square_width, square_width, square_width))
                else:
                    pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - promotionSquare[1]) * square_width, 50 + (7 - button) * square_width, square_width, square_width))
            screen.blit(promQ, (x_val, y_val))
            if side: y_val -= square_width
            else: y_val += square_width
            screen.blit(promN, (x_val, y_val))
            if side: y_val -= square_width
            else: y_val += square_width
            screen.blit(promR, (x_val, y_val))
            if side: y_val -= square_width
            else: y_val += square_width
            screen.blit(promB, (x_val, y_val))

    pygame.display.flip()
    clock.tick(60)
