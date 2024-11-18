from tempfile import tempdir

import pygame

pygame.init()

screen = pygame.display.set_mode((740, 740))
clock = pygame.time.Clock()

side = False
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

turn = True
check = False
checkmate = False

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
    def return_legal_moves(self):
        return [[2,1],[6,3],[4,5],[0,7]]

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
    # def return_legal_moves(self):

class Night(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "night"

class Bishop(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "bishop"

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

class Queen(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.image = load_piece_image(color, self.get_name())
    def get_image(self):
        return self.image
    def get_name(self):
        return "queen"

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
def updateChessPiece(piece, newLocation):
    chessBoard[piece.get_position()[0]][piece.get_position()[1]] = None
    chessBoard[newLocation[0]][newLocation[1]] = piece
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

def get_board_position(mouse_pos):
    """Convert mouse position to board indices"""
    if 50 <= mouse_pos[0] <= 700 and 50 <= mouse_pos[1] <= 700:
        if side:
            return (int((mouse_pos[1] - 50) // square_width),
                int((mouse_pos[0] - 50) // square_width))
        else:
            return ((int(7 - (mouse_pos[1] - 50) // square_width)),
            (int(7-((mouse_pos[0] - 50) // square_width))))
    return None

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
                print(i,"  ",j)
                if selected_piece is None:
                    # First click - select piece
                    if chessBoard[i][j]:
                        if (chessBoard[i][j].get_color() == "white" and current_player == 1) or (chessBoard[i][j].get_color()=="black" and current_player == 2):
                            selected_piece = chessBoard[i][j]
                            selected_i = i
                            selected_j = j
                else:
                    # Second click - move piece if it's not a drag operation
                    if not dragging:
                        if (i != selected_i or j != selected_j):
                            # Move the piece
                            chessBoard[i][j] = selected_piece
                            chessBoard[selected_i][selected_j] = None

                        if i == selected_i and j==selected_j:
                            selected_piece = None
                            selected_i = None
                            selected_j = None
                            break

                        current_player = 1 if current_player == 2 else 2
                        selected_piece = None
                        selected_i = None
                        selected_j = None
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
                        chessBoard[selected_i][selected_j] = None

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                # Handle drag and drop movement
                board_pos = get_board_position(pygame.mouse.get_pos())
                if board_pos:
                    new_i, new_j = board_pos
                    chessBoard[new_i][new_j] = selected_piece
                else:
                    # If dropped outside the board, return piece to original position
                    chessBoard[selected_i][selected_j] = selected_piece

                selected_piece = None
                selected_i = None
                selected_j = None
                dragging = False
                current_player = 1 if current_player == 2 else 2

            initial_click_pos = None

    # Do logical updates here
    screen.fill(background_color)
    screen.blit(background_image, (50, 50))

    # Highlight selected square if a piece is selected
    if selected_piece:
        if side:
            pygame.draw.rect(screen, (255, 255, 197),
                         pygame.Rect(50 + selected_j * square_width,
                                     50 + selected_i * square_width,
                                     square_width+1, square_width+1))
        else:
            pygame.draw.rect(screen, (255, 255, 197),
                         pygame.Rect(50 + (7-selected_j) * square_width,
                                     50 + (7-selected_i) * square_width,
                                     square_width+1, square_width+1))


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
                piece = chessBoard[i][j]
                x_val = 50 + j * square_width
                y_val = 50 + i * square_width
                if(side):
                    screen.blit(piece.get_image(), (x_val, y_val))
                else:
                    screen.blit(piece.get_image(), (screen.get_width()-x_val-square_width, screen.get_width()-y_val-square_width))

    # Draw dragged piece only if actually dragging
    if dragging and selected_piece:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen.blit(selected_piece.get_image(),
                    (mouse_x - background_image.get_width() // 16,
                     mouse_y - background_image.get_width() // 16))
    if selected_piece:
        for square in selected_piece.return_legal_moves():
            circle_surface = pygame.Surface((square_width,square_width), pygame.SRCALPHA)
            circle_surface.set_alpha(85)
            if not side:
                temp = square[0]
                square[0] = square[1]
                square[1] = temp
            if not chessBoard[square[0]][square[1]]:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), 13)
            else:
                pygame.draw.circle(circle_surface, (75, 75, 75), (square_width / 2, square_width / 2), square_width/2, 6)
            screen.blit(circle_surface, (50+square[1]*square_width,50+square[0]*square_width))


    pygame.display.flip()
    clock.tick(60)