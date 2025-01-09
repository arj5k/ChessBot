from tempfile import tempdir
import pygame
import torch
import chess
import copy
import time
import threading
print("importing chessbot")
from ChessBotMain import ChessBot, encode_board
print('imported')
def load_model(model_path="chess_ai_model.pt"):
     print("loading...")
     model = ChessBot()
     model.load_state_dict(torch.load(model_path))
     model.eval()  # Set the model to evaluation mode
     print("loaded")
     return model


def evaluate_position(fen, model):
     #Convert the FEN to an encoded board
     board = chess.Board(fen)
     encoded_board = encode_board(board)

    # Convert to a PyTorch tensor and add a batch dimension
     input_tensor = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)

    # Evaluate the position
     with torch.no_grad():
         evaluation = model(input_tensor)
     return evaluation.item()

print("callind model")
model = load_model("chess_ai_model.pt")

pygame.init()

screen = pygame.display.set_mode((740, 740))
clock = pygame.time.Clock()

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

PLAYER = {
    1:"white",
    2:"black"
}

current_player = 1

turnClock = 1
halfClock = 0
turn = True
check = False
checkmate = False
#add an additional element to the grid that says if a piece attacks it, also to the legal moves or somewhere else that only allows a move to be legal if not in check, also make everything else cooperate, especially update the board array

WIDTH = 8
chessBoard = [[None for x in range(WIDTH)] for y in range(WIDTH)]
attacked = [[None for x in range(WIDTH)] for y in range(WIDTH)]

class Piece:
    def __init__(self, position, color):
        self.position = position
        self.color = color
    def get_position(self):
        return self.position
    def get_color(self):
        return self.color

class Pawn(Piece):
    def __init__(self, position, color, double_move, en_passantable):
        super().__init__(position, color)
        self.double_move = double_move
        self.en_passantable = en_passantable
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "pawn"
    def can_move_two(self):
        return self.double_move
    def can_be_en_passanted(self):
        return self.en_passantable
    def return_legal_moves(self):
        #if piece in way, puts in check, or out of bounds + (also have to add in the check check after determign everything else possible function) - add in this square as a possibility to be returned - THE WAY YOU WILL CODE THE CheckChecked function does not iterate through every piece, so you will have to add "in_check" boolean at the start of each return_legal_moves function everyttime - add in a seperate part for getting out of check legal moves thing probably, this may mean you have to implement aditional variables (like what is the piece giving check, is it a double check)
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        if (self.double_move):
            if (self.get_color() == "black"):
                if (not (chessBoard[newPosition[0]+1][newPosition[1]] or chessBoard[newPosition[0]+2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1) #add this to every other part
            else:
                if (not (chessBoard[newPosition[0] - 1][newPosition[1]] or chessBoard[newPosition[0]-2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)  # add this to every other part
        else:
            if (self.get_color() == "black"):
                if (not (chessBoard[newPosition[0]+1][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] + 1])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] + 1])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1]]) and chessBoard[newPosition[0] + 1][newPosition[1]].get_name() == "pawn" and chessBoard[newPosition[0] + 1][newPosition[1]].can_be_en_passanted() and chessBoard[newPosition[0] + 1][newPosition[1]].get_color() == "white"):
                    newPosition1 = copy.deepcopy(self.get_position()) #have to add in an addition check for en passant if that will put you in check
                    newPosition1[0] += 1
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1]]) and chessBoard[newPosition[0] - 1][newPosition[1]].get_name() == "pawn" and chessBoard[newPosition[0] - 1][newPosition[1]].can_be_en_passanted() and chessBoard[newPosition[0] - 1][newPosition[1]].get_color() == "white"):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
            else:
                if (not (chessBoard[newPosition[0] - 1][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] - 1])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] - 1])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1]]) and chessBoard[newPosition[0] + 1][newPosition[1]].get_name() == "pawn" and chessBoard[newPosition[0] + 1][newPosition[1]].can_be_en_passanted() and chessBoard[newPosition[0] + 1][newPosition[1]].get_color() == "black"):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
                if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1]]) and chessBoard[newPosition[0] - 1][newPosition[1]].get_name() == "pawn" and chessBoard[newPosition[0] - 1][newPosition[1]].can_be_en_passanted() and chessBoard[newPosition[0] - 1][newPosition[1]].get_color() == "black"):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
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
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 2] and not (chessBoard[horizontal + 1][vertical - 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
        if (vertical < 6):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 2] and not (chessBoard[horizontal - 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 2] and not (chessBoard[horizontal + 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
        if (horizontal > 1):
            if (vertical > 0):
                if (chessBoard[horizontal - 2][vertical - 1] and not (chessBoard[horizontal - 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
            if (vertical < 7):
                if (chessBoard[horizontal - 2][vertical + 1] and not (chessBoard[horizontal - 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
        if (horizontal < 6):
            if (vertical > 0):
                if (chessBoard[horizontal + 2][vertical - 1] and not (chessBoard[horizontal + 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
            if (vertical < 7):
                if (chessBoard[horizontal + 2][vertical + 1] and not (chessBoard[horizontal + 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = 100

        vertical = newPosition[1] - 1
        horizontal = newPosition[0]
        while (vertical > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1)
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1)
                horizontal = 100
        return legalMoves

class King(Piece):
    def __init__(self, position, color, castle, check):
        super().__init__(position, color)
        self.castle = castle
        self.check = check
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "king"
    def can_castle(self):
        return self.castle
    def in_check(self):
        return self.check
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
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 1] and not (chessBoard[horizontal + 1][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
            if (chessBoard[horizontal][vertical - 1] and not (chessBoard[horizontal][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
            elif (not chessBoard[horizontal][vertical - 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
        if (vertical < 7):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 1] and not (chessBoard[horizontal - 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal - 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 1] and not (chessBoard[horizontal + 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
                elif (not chessBoard[horizontal + 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1)
            if (chessBoard[horizontal][vertical + 1] and not (chessBoard[horizontal][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
            elif (not chessBoard[horizontal][vertical + 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
        if (horizontal > 0):
            if (chessBoard[horizontal - 1][vertical] and not (chessBoard[horizontal - 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
            elif (not chessBoard[horizontal - 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
        if (horizontal < 7):
            if (chessBoard[horizontal + 1][vertical] and not (chessBoard[horizontal + 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
            elif (not chessBoard[horizontal + 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1)
        return legalMoves

def checkChecked(originalLocation, newLocation, isKing, color):
    #have to add in thing if it moves in the same direction so still blocks the check
    #severrely update this to catch further condiitons(en passant?)
    if (not isKing):
        if (color == "black"):
            kRow = bKing[0].get_position()[1]
            kCol = bKing[0].get_position()[0]
            if (kRow == originalLocation[1]):
                if (kCol < originalLocation[0]):
                    iterator = originalLocation[0] - 1
                    while (iterator > -1):
                        if (chessBoard[iterator][kRow] and not (iterator == originalLocation[0])):
                            if (chessBoard[iterator][kRow].get_name() == "rook" or chessBoard[iterator][kRow].get_name() == "queen") and chessBoard[iterator][kRow].get_color() == "white":
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[0] + 1
                    while (iterator < 8):
                        if (chessBoard[iterator][kRow] and not (iterator == originalLocation[0])):
                            if (chessBoard[iterator][kRow].get_name() == "rook" or chessBoard[iterator][kRow].get_name() == "queen") and chessBoard[iterator][kRow].get_color() == "white":
                                return True
                            return False
                        iterator += 1
                    return False
            elif (kCol == originalLocation[0]):
                if (kRow < originalLocation[1]):
                    iterator = originalLocation[1] - 1
                    while (iterator > -1):
                        if (chessBoard[kCol][iterator] and not (iterator == originalLocation[1])):
                            if (chessBoard[kCol][iterator].get_name() == "rook" or chessBoard[kCol][iterator].get_name() == "queen") and chessBoard[kCol][iterator].get_color() == "white":
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[1] + 1
                    while (iterator < 8):
                        if (chessBoard[kCol][iterator] and not (iterator == originalLocation[1])):
                            if (chessBoard[kCol][iterator].get_name() == "rook" or chessBoard[kCol][iterator].get_name() == "queen") and chessBoard[kCol][iterator].get_color() == "white":
                                return True
                            return False
                        iterator += 1
                    return False
            elif (abs(kRow - originalLocation[1]) == abs(kCol - originalLocation[0])):
                return False
            elif (kRow + kCol == originalLocation[1] + originalLocation[0]):
                return False
            else:
                return False
        else:
            kRow = wKing[0].get_position()[1]
            kCol = wKing[0].get_position()[0]
            if (kRow == originalLocation[1]):
                if (kCol < originalLocation[0]):
                    iterator = originalLocation[0] - 1
                    while (iterator > -1):
                        if (chessBoard[iterator][kRow] and not (iterator == originalLocation[0])):
                            if (chessBoard[iterator][kRow].get_name() == "rook" or chessBoard[iterator][
                                kRow].get_name() == "queen") and chessBoard[iterator][kRow].get_color() == "black":
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[0] + 1
                    while (iterator < 8):
                        if (chessBoard[iterator][kRow] and not (iterator == originalLocation[0])):
                            if (chessBoard[iterator][kRow].get_name() == "rook" or chessBoard[iterator][
                                kRow].get_name() == "queen") and chessBoard[iterator][kRow].get_color() == "black":
                                return True
                            return False
                        iterator += 1
                    return False
            elif (kCol == originalLocation[0]):
                if (kRow < originalLocation[1]):
                    iterator = originalLocation[1] - 1
                    while (iterator > -1):
                        if (chessBoard[kCol][iterator] and not (iterator == originalLocation[1])):
                            if (chessBoard[kCol][iterator].get_name() == "rook" or chessBoard[kCol][
                                iterator].get_name() == "queen") and chessBoard[kCol][iterator].get_color() == "black":
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[1] + 1
                    while (iterator < 8):
                        if (chessBoard[kCol][iterator] and not (iterator == originalLocation[1])):
                            if (chessBoard[kCol][iterator].get_name() == "rook" or chessBoard[kCol][
                                iterator].get_name() == "queen") and chessBoard[kCol][iterator].get_color() == "black":
                                return True
                            return False
                        iterator += 1
                    return False
            elif (abs(kRow - originalLocation[1]) == abs(kCol - originalLocation[0])):
                return False
            elif (kRow + kCol == originalLocation[1] + originalLocation[0]):
                return False
            else:
                return False
    else:
        return False #also have to change this

def updateChessPiece(piece, newLocation, board):
    board[piece.get_position()[0]][piece.get_position()[1]] = None
    board[newLocation[0]][newLocation[1]] = piece
    piece.position = newLocation

bPawns = [None] * 8
wPawns = [None] * 8
for z in range(WIDTH):
    bPawns[z] = Pawn([1, z], "black", True, False)
    wPawns[z] = Pawn([6, z], "white", True, False)

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
bKing[0] = King([0, 4], "black", True, False)
wKing[0] = King([7, 4], "white", True, False)

for element in bPawns: updateChessPiece(element, element.get_position(), chessBoard)
for element in wPawns: updateChessPiece(element, element.get_position(), chessBoard)
for element in bRooks: updateChessPiece(element, element.get_position(), chessBoard)
for element in wRooks: updateChessPiece(element, element.get_position(), chessBoard)
for element in bNights: updateChessPiece(element, element.get_position(), chessBoard)
for element in wNights: updateChessPiece(element, element.get_position(), chessBoard)
for element in bBishops: updateChessPiece(element, element.get_position(), chessBoard)
for element in wBishops: updateChessPiece(element, element.get_position(), chessBoard)
for element in bQueen: updateChessPiece(element, element.get_position(), chessBoard)
for element in wQueen: updateChessPiece(element, element.get_position(), chessBoard)
for element in bKing: updateChessPiece(element, element.get_position(), chessBoard)
for element in wKing: updateChessPiece(element, element.get_position(), chessBoard)

# Variables for piece selection and movement
selected_i = None
selected_j = None
selected_piece = None
dragging = False
initial_click_pos = None
legal_moves = None

def get_board_position(mouse_pos):
    """Convert mouse position to board indices"""
    if 50 <= mouse_pos[0] <= 700 and 50 <= mouse_pos[1] <= 700:
        if side:
            return (int((mouse_pos[1] - 50) // square_width),
                    int((mouse_pos[0] - 50) // square_width))
        else:
            return ((int(7 - (mouse_pos[1] - 50) // square_width)),
                    (int(7 - ((mouse_pos[0] - 50) // square_width))))
    return None
def make_fen(board, halfClock, turnClock, current_player):
    board_string = ""
    emptys = 0
    turn_string = "w" if current_player == 1 else "b"
    castle_string = "-"
    en_passant = "-"
    halfmoves = str(halfClock)
    moves = str(turnClock)
    for row in board:
        for game_piece in row:
            if game_piece != None:
                if emptys:
                    board_string += str(emptys)
                emptys = 0
                if game_piece.get_color() == "black":
                    match str(type(game_piece)):
                        case "<class '__main__.Pawn'>": board_string += "p"
                        case "<class '__main__.Night'>": board_string += "n"
                        case "<class '__main__.Bishop'>": board_string += "b"
                        case "<class '__main__.Rook'>": board_string += "r"
                        case "<class '__main__.Queen'>": board_string += "q"
                        case "<class '__main__.King'>": board_string += "k"
                else:
                    match str(type(game_piece)):
                        case "<class '__main__.Pawn'>": board_string += "P"
                        case "<class '__main__.Night'>": board_string += "N"
                        case "<class '__main__.Bishop'>": board_string += "B"
                        case "<class '__main__.Rook'>": board_string += "R"
                        case "<class '__main__.Queen'>": board_string += "Q"
                        case "<class '__main__.King'>": board_string += "K"
            else:
                emptys+=1
        if emptys:
            board_string += str(emptys)
        emptys = 0
        if board.index(row) != WIDTH-1:
            board_string +="/"
    return board_string+" "+turn_string+" "+castle_string+" "+en_passant+" "+halfmoves+" "+moves
def copy_piece(piece):
    if isinstance(piece, Pawn):
        return Pawn(piece.get_position(), piece.get_color(), piece.can_move_two(), piece.can_be_en_passanted())
    elif isinstance(piece, Night):
        return Night(piece.get_position(), piece.get_color())
    elif isinstance(piece, Bishop):
        return Bishop(piece.get_position(), piece.get_color())
    elif isinstance(piece, Queen):
        return Queen(piece.get_position(), piece.get_color())
    elif isinstance(piece, Rook):
        return Rook(piece.get_position(), piece.get_color(), piece.can_castle())
    elif isinstance(piece, King):
        return King(piece.get_position(), piece.get_color(), piece.can_castle(), piece.in_check())

def deep_copy_board(board):
    return [[copy_piece(piece) for piece in row] for row in board]
def bot_move():
    global turnClock
    global halfClock
    global current_player
    best_eval = None
    best_piece = None
    best_move = None
    new_clock = turnClock
    new_hc = halfClock

    for i in range (0, WIDTH):
        for j in range (0, WIDTH):
            print("working")
            if chessBoard[i][j] != None and chessBoard[i][j].get_color() == "black":
                for square in chessBoard[i][j].return_legal_moves():
                    print("iterating")
                    new_move = deep_copy_board(chessBoard)
                    updateChessPiece(new_move[i][j], square, new_move)
                    temp_clock = turnClock + 1
                    temp_hc = halfClock + 1
                    if isinstance(chessBoard[i][j], Pawn) or chessBoard[square[0]][square[1]] != None:
                        temp_hc = 0
                    print(make_fen(new_move, temp_hc, temp_clock, 1))
                    test_eval = evaluate_position(make_fen(new_move, new_hc, new_clock, 1), model)
                    if ((best_eval==None) or test_eval<best_eval):
                        print("found better move")
                        best_eval = test_eval
                        best_piece = chessBoard[i][j]
                        best_move = square
                        new_clock = temp_clock
                        new_hc = temp_hc
    time.sleep(0.5)
    updateChessPiece(best_piece, best_move, chessBoard)
    current_player = 1
    halfClock = new_hc
    turnClock = new_clock

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
    white_y = 705

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

    # Position and draw the text
    black_text_pos = center_text(black_text, black_x, black_y, timer_width, timer_height)
    white_text_pos = center_text(white_text, white_x, white_y, timer_width, timer_height)

    screen.blit(black_text, black_text_pos)
    screen.blit(white_text, white_text_pos)

while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        elif event.type == pygame.MOUSEBUTTONDOWN:
            initial_click_pos = pygame.mouse.get_pos()
            board_pos = get_board_position(initial_click_pos)

            if board_pos:
                i, j = board_pos
                if selected_piece is None:
                    # First click - select piece
                    if chessBoard[i][j] and chessBoard[i][j].get_color() == PLAYER[current_player]:
                        selected_piece = chessBoard[i][j]
                        selected_i = i
                        selected_j = j
                        legal_moves = selected_piece.return_legal_moves()
                else:
                    # Second click - move piece if it's not a drag operation
                    if not dragging:
                        if (i != selected_i or j != selected_j)  and [i,j] in selected_piece.return_legal_moves():
                            # Move the piece
                            reset_clock = False
                            if chessBoard[i][j] or isinstance(selected_piece, Pawn):
                                reset_clock = True
                            updateChessPiece(selected_piece, [i,j], chessBoard)
                            #add update34h208hfou32hrfgou3ewhgiutehgoiuhgwoiurtghoiuwetgtrw
                            if current_player == 1:
                                current_player = 2
                                halfClock+=1
                            else:
                                current_player = 1
                                turnClock+=1
                                halfClock+=1
                            if reset_clock:
                                halfClock = 0
                        selected_piece = None
                        selected_i = None
                        selected_j = None
                        legal_moves = None



        elif event.type == pygame.MOUSEMOTION:
            if selected_piece and not dragging:
                # Check if mouse has moved enough to start dragging
                current_pos = pygame.mouse.get_pos()
                if initial_click_pos:
                    dx = current_pos[0] - initial_click_pos[0]
                    dy = current_pos[1] - initial_click_pos[1]
                    # If mouse has moved more than 5 pixels, start dragging
                    if (dx * dx + dy * dy) > 25:  # 5 pixels squared
                        dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                # Handle drag and drop movement
                board_pos = get_board_position(pygame.mouse.get_pos())
                if board_pos:
                    new_i, new_j = board_pos
                    if([new_i, new_j] in selected_piece.return_legal_moves()):
                        updateChessPiece(selected_piece, [new_i, new_j], chessBoard)
                        if current_player == 1:
                            current_player = 2
                        else:
                            current_player = 1
                            turnClock += 1
                selected_piece = None
                selected_i = None
                selected_j = None
                dragging = False
                legal_moves = None
            initial_click_pos = None

    # Do logical updates here.
    screen.fill(background_color)
    screen.blit(background_image, (50, 50))

    # Highlight selected square if a piece is selected
    if selected_piece:
        if side:
            pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + selected_j * square_width, 50 + selected_i * square_width, square_width + 1, square_width + 1))
        else:
            pygame.draw.rect(screen, (255, 255, 197), pygame.Rect(50 + (7 - selected_j) * square_width, 50 + (7 - selected_i) * square_width, square_width + 1, square_width + 1))

    if current_player == 1:  # White's turn
        WHITE_TIMELEFT -= clock.get_time()
        if WHITE_TIMELEFT < 0:
            WHITE_TIMELEFT = 0
    else:  # Black's turn
        BLACK_TIMELEFT -= clock.get_time()
        if BLACK_TIMELEFT < 0:
            BLACK_TIMELEFT = 0


    draw_timer(screen, WHITE_TIMELEFT, BLACK_TIMELEFT)

    # Render the graphics here.
    for i in range(WIDTH):
        for j in range(WIDTH):
            if chessBoard[i][j]:
                if (not (dragging and i == selected_i and j == selected_j)):
                    piece = chessBoard[i][j]
                    x_val = 50 + j * square_width
                    y_val = 50 + i * square_width
                    if (side):
                        screen.blit(piece.get_image(), (x_val, y_val))
                    else:
                        screen.blit(piece.get_image(), (
                            screen.get_width() - x_val - square_width,
                            screen.get_width() - y_val - square_width))

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
            sq_copy = []
            if side:
                sq_copy.append(square[0])
                sq_copy.append(square[1])
            else:
                sq_copy.append(square[1])
                sq_copy.append(square[0])
            if not chessBoard[sq_copy[0]][sq_copy[1]]:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), 13)
            else:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), square_width / 2,
                                   6)
            screen.blit(circle_surface, (50 + sq_copy[1] * square_width, 50 + sq_copy[0] * square_width))

    if(current_player == 2):
        t1 = threading.Thread(target=bot_move)
        t1.start()
        current_player=1
    pygame.display.flip()
    clock.tick(60)
