from tempfile import tempdir
import pygame
import copy

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
    1:"WHITE",
    2:"BLACK"
}

current_player = 1
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
        #add promotion thing here as well
    def return_legal_moves(self): #enpassant doesnt work also you can take your own piece, also why does it signal you can attack outside without error (this itself isnt a problem but seems almost as if signals an error)
        #if piece in way, puts in check, or out of bounds + (also have to add in the check check after determign everything else possible function) - add in this square as a possibility to be returned - THE WAY YOU WILL CODE THE CheckChecked function does not iterate through every piece, so you will have to add "in_check" boolean at the start of each return_legal_moves function everyttime - add in a seperate part for getting out of check legal moves thing probably, this may mean you have to implement aditional variables (like what is the piece giving check, is it a double check)
        legalMoves = []
        newPosition = copy.deepcopy(self.get_position())
        if (self.double_move):
            if (self.get_color() == "black"):
                if (not (chessBoard[newPosition[0]+1][newPosition[1]] or chessBoard[newPosition[0]+2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1 + [0])
            else:
                if (not (chessBoard[newPosition[0] - 1][newPosition[1]] or chessBoard[newPosition[0]-2][newPosition[1]])):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1 + [0])
        if (self.get_color() == "black"):#first check to see if last square to promote bro
            if (not (chessBoard[newPosition[0]+1][newPosition[1]])):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] + 1]) and chessBoard[newPosition[0] + 1][newPosition[1] + 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0] + 1][newPosition[1] - 1]) and chessBoard[newPosition[0] + 1][newPosition[1] - 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0]][newPosition[1] + 1]) and chessBoard[newPosition[0]][newPosition[1] + 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] + 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] + 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position()) #have to add in an addition check for en passant if that will put you in check
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1 + [1])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0]][newPosition[1] - 1]) and chessBoard[newPosition[0]][newPosition[1] - 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] - 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] - 1].get_color() == "white"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1 + [1])
        else:
            if (not (chessBoard[newPosition[0] - 1][newPosition[1]])):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0] - 1][newPosition[1] + 1]) and chessBoard[newPosition[0] - 1][newPosition[1] + 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] - 1]) and chessBoard[newPosition[0] - 1][newPosition[1] - 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1 + [0])
            if (newPosition[1] != 7 and (chessBoard[newPosition[0]][newPosition[1] + 1]) and chessBoard[newPosition[0]][newPosition[1] + 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] + 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] + 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1 + [1])
            if (newPosition[1] != 0 and (chessBoard[newPosition[0]][newPosition[1] - 1]) and chessBoard[newPosition[0]][newPosition[1] - 1].get_name() == "pawn" and chessBoard[newPosition[0]][newPosition[1] - 1].can_be_en_passanted() and chessBoard[newPosition[0]][newPosition[1] - 1].get_color() == "black"):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] -= 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
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
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 2] and not (chessBoard[horizontal + 1][vertical - 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical - 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (vertical < 6):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 2] and not (chessBoard[horizontal - 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 2] and not (chessBoard[horizontal + 1][vertical + 2].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical + 2]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 1
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (horizontal > 1):
            if (vertical > 0):
                if (chessBoard[horizontal - 2][vertical - 1] and not (chessBoard[horizontal - 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (vertical < 7):
                if (chessBoard[horizontal - 2][vertical + 1] and not (chessBoard[horizontal - 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] -= 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
        if (horizontal < 6):
            if (vertical > 0):
                if (chessBoard[horizontal + 2][vertical - 1] and not (chessBoard[horizontal + 2][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 2][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] -= 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (vertical < 7):
                if (chessBoard[horizontal + 2][vertical + 1] and not (chessBoard[horizontal + 2][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 2][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] += 2
                    newPosition1[1] += 1
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
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
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] - 1
        while (vertical < 8 and horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        vertical = newPosition[1] - 1
        horizontal = newPosition[0] + 1
        while (vertical > -1 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        horizontal = newPosition[0] + 1
        while (vertical < 8 and horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = 100

        vertical = newPosition[1] - 1
        horizontal = newPosition[0]
        while (vertical > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = -100
        vertical = newPosition[1] + 1
        while (vertical < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                vertical += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                vertical = 100
        vertical = newPosition[1]
        horizontal = newPosition[0] - 1
        while (horizontal > -1):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal -= 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                horizontal = -100
        horizontal = newPosition[0] + 1
        while (horizontal < 8):
            if not (chessBoard[horizontal][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
                horizontal += 1
            else:
                if (not (chessBoard[horizontal][vertical].get_color() == self.get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal
                    newPosition1[1] = vertical
                    if (not checkChecked(self.get_position(), newPosition1, False, self.get_color())):
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
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical - 1] and not (chessBoard[horizontal + 1][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical - 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical - 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (chessBoard[horizontal][vertical - 1] and not (chessBoard[horizontal][vertical - 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal][vertical - 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical - 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (vertical < 7):
            if (horizontal > 0):
                if (chessBoard[horizontal - 1][vertical + 1] and not (chessBoard[horizontal - 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal - 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal - 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (horizontal < 7):
                if (chessBoard[horizontal + 1][vertical + 1] and not (chessBoard[horizontal + 1][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
                elif (not chessBoard[horizontal + 1][vertical + 1]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = horizontal + 1
                    newPosition1[1] = vertical + 1
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [0])
            if (chessBoard[horizontal][vertical + 1] and not (chessBoard[horizontal][vertical + 1].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal][vertical + 1]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal
                newPosition1[1] = vertical + 1
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (horizontal > 0):
            if (chessBoard[horizontal - 1][vertical] and not (chessBoard[horizontal - 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal - 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal - 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if (horizontal < 7):
            if (chessBoard[horizontal + 1][vertical] and not (chessBoard[horizontal + 1][vertical].get_color() == chessBoard[horizontal][vertical].get_color())):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
            elif (not chessBoard[horizontal + 1][vertical]):
                newPosition1 = copy.deepcopy(self.get_position())
                newPosition1[0] = horizontal + 1
                newPosition1[1] = vertical
                if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                    legalMoves.append(newPosition1 + [0])
        if self.castle:
            if (chessBoard[self.get_position()[0]][0] and chessBoard[self.get_position()[0]][0].get_name() == "rook" and chessBoard[self.get_position()[0]][0].can_castle()):
                if (not chessBoard[self.get_position()[0]][1] and not chessBoard[self.get_position()[0]][2] and not chessBoard[self.get_position()[0]][3]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = self.get_position()[0]
                    newPosition1[1] = 2
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [2])
            if (chessBoard[self.get_position()[0]][7] and chessBoard[self.get_position()[0]][7].get_name() == "rook" and chessBoard[self.get_position()[0]][7].can_castle()):
                if (not chessBoard[self.get_position()[0]][6] and not chessBoard[self.get_position()[0]][5]):
                    newPosition1 = copy.deepcopy(self.get_position())
                    newPosition1[0] = self.get_position()[0]
                    newPosition1[1] = 6
                    if (not checkChecked(self.get_position(), newPosition1, True, self.get_color())):
                        legalMoves.append(newPosition1 + [3])
        return legalMoves

def checkChecked(originalLocation, newLocation, color, move_data=None):
    # #simulate new board
    # newBoard = copy.deepcopy(chessBoard)
    # newBoard[newLocation[0]][newLocation[1]] = newBoard[originalLocation[0]][originalLocation[1]] #check to see if this works (referecnes and stuff)
    # newBoard[originalLocation[0]][originalLocation[1]] = None
    # if not (move_data == None) and not (move_data[0] == 0):
    #     if (move_data[0] == 1):
    #         if color == "white":
    #             newBoard[newLocation[0] + 1][newLocation[1]] = None
    #         else:
    #             newBoard[newLocation[0] - 1][newLocation[1]] = None
    #     elif (move_data[0] == 2):
    #         #special stuff here (castle through check)
    #     elif (move_data[0] == 3):
    #         #special stuff here, also WONT have to add an extra movedata here for promotion because this doesnt affect checks and stuff
    # if color == "white":
    #     wKing
    # else:
    return False



def updateChessPiece(piece, newLocation, updating=False, move_data=None):
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

def get_board_position(mouse_pos):
    """Convert mouse position to board indices"""
    if 50 <= mouse_pos[0] < 690 and 50 <= mouse_pos[1] < 690:
        if side:
            return (int((mouse_pos[1] - 50) // square_width),
                    int((mouse_pos[0] - 50) // square_width))
        else:
            return ((int(7 - (mouse_pos[1] - 50) // square_width)),
                    (int(7 - ((mouse_pos[0] - 50) // square_width))))
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
                        if (selected_piece.get_color() == "white" and current_player == 1) or (selected_piece.get_color() == "black" and current_player == 2):
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
                                updateChessPiece(selected_piece, [i, j], True, moves_data[legal_moves.index([i,j])])
                            elif (not i == -1) and chessBoard[i][j]:
                                inefficientVariable = False
                                selected_piece = chessBoard[i][j]
                                selected_i = i
                                selected_j = j
                                if (selected_piece.get_color() == "white" and current_player == 1) or (selected_piece.get_color() == "black" and current_player == 2):
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
                        updateChessPiece(selected_piece, [new_i, new_j], True, moves_data[legal_moves.index([new_i,new_j])])
                selected_piece = None
                selected_i = None
                selected_j = None
                dragging = False
                legal_moves = None
                moves_data = None
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
            if not side:
                temp = square[0]
                square[0] = square[1]
                square[1] = temp
            if not chessBoard[square[0]][square[1]]:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), 13)
            else:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), square_width / 2,
                                   6)
            screen.blit(circle_surface, (50 + square[1] * square_width, 50 + square[0] * square_width))

    pygame.display.flip()
    clock.tick(60)
