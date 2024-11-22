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
    1: "WHITE",
    2: "BLACK"
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
        legalMoves = []
        newPosition = self.get_position()
        if (self.double_move):
            if (self.get_color() == "black"):
                # Black pawns should move down (+2)
                if (not (chessBoard[newPosition[0]][newPosition[1] - 1] or chessBoard[newPosition[0]][newPosition[1] - 2])):
                    newPosition1 = self.get_position().copy()
                    newPosition1[1] -= 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                        legalMoves.append(newPosition1)
            else:
                # White pawns should move up (-2)
                if (not (chessBoard[newPosition[0]][newPosition[1] + 1] or chessBoard[newPosition[0]][newPosition[1] + 2])):
                    newPosition1 = self.get_position().copy()
                    newPosition1[1] += 2
                    if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                        legalMoves.append(newPosition1)
        if (self.get_color() == "black"):
            if (not (chessBoard[newPosition[0]][newPosition[1] + 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] + 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] + 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1]]) and
                    chessBoard[newPosition[0] + 1][newPosition[1]].get_name() == "pawn" and
                    chessBoard[newPosition[0] + 1][newPosition[1]].can_be_en_passanted() and
                    chessBoard[newPosition[0] + 1][newPosition[1]].get_color() == "white"):
                newPosition1 = self.get_position().copy()
                newPosition1[0] += 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1]]) and
                    chessBoard[newPosition[0] - 1][newPosition[1]].get_name() == "pawn" and
                    chessBoard[newPosition[0] - 1][newPosition[1]].can_be_en_passanted() and
                    chessBoard[newPosition[0] - 1][newPosition[1]].get_color() == "white"):
                newPosition1 = self.get_position().copy()
                newPosition1[0] -= 1
                newPosition1[1] += 1
                if (not checkChecked(self.get_position(), newPosition1, False, "black")):
                    legalMoves.append(newPosition1)
        else:
            if (not (chessBoard[newPosition[0]][newPosition[1] - 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1] - 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1] - 1])):
                newPosition1 = self.get_position().copy()
                newPosition1[0] -= 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 7 and (chessBoard[newPosition[0] + 1][newPosition[1]]) and
                    chessBoard[newPosition[0] + 1][newPosition[1]].get_name() == "pawn" and
                    chessBoard[newPosition[0] + 1][newPosition[1]].can_be_en_passanted() and
                    chessBoard[newPosition[0] + 1][newPosition[1]].get_color() == "black"):
                newPosition1 = self.get_position().copy()
                newPosition1[0] += 1
                newPosition1[1] -= 1
                if (not checkChecked(self.get_position(), newPosition1, False, "white")):
                    legalMoves.append(newPosition1)
            if (newPosition[0] != 0 and (chessBoard[newPosition[0] - 1][newPosition[1]]) and
                    chessBoard[newPosition[0] - 1][newPosition[1]].get_name() == "pawn" and
                    chessBoard[newPosition[0] - 1][newPosition[1]].can_be_en_passanted() and
                    chessBoard[newPosition[0] - 1][newPosition[1]].get_color() == "black"):
                newPosition1 = self.get_position().copy()
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
        newPosition = self.get_position()
        vertical = newPosition[1]
        horizontal = newPosition[0]

        knight_moves = [
            (-1, -2), (1, -2), (-1, 2), (1, 2),
            (-2, -1), (-2, 1), (2, -1), (2, 1)
        ]

        for dx, dy in knight_moves:
            new_h, new_v = horizontal + dx, vertical + dy
            if 0 <= new_h < 8 and 0 <= new_v < 8:
                if not chessBoard[new_h][new_v] or chessBoard[new_h][new_v].get_color() != self.get_color():
                    newPosition1 = self.get_position().copy()
                    newPosition1[0], newPosition1[1] = new_h, new_v
                    if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
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
        newPosition = self.get_position()
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            x, y = newPosition[0] + dx, newPosition[1] + dy
            while 0 <= x < 8 and 0 <= y < 8:
                if not chessBoard[x][y]:
                    newPosition1 = self.get_position().copy()
                    newPosition1[0], newPosition1[1] = x, y
                    if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                        legalMoves.append(newPosition1)
                else:
                    if chessBoard[x][y].get_color() != self.get_color():
                        newPosition1 = self.get_position().copy()
                        newPosition1[0], newPosition1[1] = x, y
                        if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                            legalMoves.append(newPosition1)
                    break
                x, y = x + dx, y + dy

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
        newPosition = self.get_position()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            x, y = newPosition[0] + dx, newPosition[1] + dy
            while 0 <= x < 8 and 0 <= y < 8:
                if not chessBoard[x][y]:
                    newPosition1 = self.get_position().copy()
                    newPosition1[0], newPosition1[1] = x, y
                    if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                        legalMoves.append(newPosition1)
                else:
                    if chessBoard[x][y].get_color() != self.get_color():
                        newPosition1 = self.get_position().copy()
                        newPosition1[0], newPosition1[1] = x, y
                        if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                            legalMoves.append(newPosition1)
                    break
                x, y = x + dx, y + dy

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
        newPosition = self.get_position()
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        for dx, dy in directions:
            x, y = newPosition[0] + dx, newPosition[1] + dy
            while 0 <= x < 8 and 0 <= y < 8:
                if not chessBoard[x][y]:
                    newPosition1 = self.get_position().copy()
                    newPosition1[0], newPosition1[1] = x, y
                    if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                        legalMoves.append(newPosition1)
                else:
                    if chessBoard[x][y].get_color() != self.get_color():
                        newPosition1 = self.get_position().copy()
                        newPosition1[0], newPosition1[1] = x, y
                        if not checkChecked(self.get_position(), newPosition1, False, self.get_color()):
                            legalMoves.append(newPosition1)
                    break
                x, y = x + dx, y + dy

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
        newPosition = self.get_position()
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        for dx, dy in directions:
            new_x = newPosition[0] + dx
            new_y = newPosition[1] + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if not chessBoard[new_x][new_y] or chessBoard[new_x][new_y].get_color() != self.get_color():
                    newPosition1 = self.get_position().copy()
                    newPosition1[0], newPosition1[1] = new_x, new_y
                    if not checkChecked(self.get_position(), newPosition1, True, self.get_color()):
                        legalMoves.append(newPosition1)

        return legalMoves


def checkChecked(originalLocation, newLocation, isKing, color):
    if not isKing:
        if color == "black":
            kRow = bKing[0].get_position()[1]
            kCol = bKing[0].get_position()[0]
            if kRow == originalLocation[1]:
                if kCol < originalLocation[0]:
                    iterator = originalLocation[0] - 1
                    while iterator > -1:
                        if chessBoard[iterator][kRow] and not (iterator == originalLocation[0]):
                            if (chessBoard[iterator][kRow].get_name() in ["rook", "queen"] and
                                    chessBoard[iterator][kRow].get_color() == "white"):
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[0] + 1
                    while iterator < 8:
                        if chessBoard[iterator][kRow] and not (iterator == originalLocation[0]):
                            if (chessBoard[iterator][kRow].get_name() in ["rook", "queen"] and
                                    chessBoard[iterator][kRow].get_color() == "white"):
                                return True
                            return False
                        iterator += 1
                    return False
            elif kCol == originalLocation[0]:
                if kRow < originalLocation[1]:
                    iterator = originalLocation[1] - 1
                    while iterator > -1:
                        if chessBoard[kCol][iterator] and not (iterator == originalLocation[1]):
                            if (chessBoard[kCol][iterator].get_name() in ["rook", "queen"] and
                                    chessBoard[kCol][iterator].get_color() == "white"):
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[1] + 1
                    while iterator < 8:
                        if chessBoard[kCol][iterator] and not (iterator == originalLocation[1]):
                            if (chessBoard[kCol][iterator].get_name() in ["rook", "queen"] and
                                    chessBoard[kCol][iterator].get_color() == "white"):
                                return True
                            return False
                        iterator += 1
                    return False
            elif abs(kRow - originalLocation[1]) == abs(kCol - originalLocation[0]):
                return False
            elif kRow + kCol == originalLocation[1] + originalLocation[0]:
                return False
            else:
                return False
        else:
            kRow = wKing[0].get_position()[1]
            kCol = wKing[0].get_position()[0]
            if kRow == originalLocation[1]:
                if kCol < originalLocation[0]:
                    iterator = originalLocation[0] - 1
                    while iterator > -1:
                        if chessBoard[iterator][kRow] and not (iterator == originalLocation[0]):
                            if (chessBoard[iterator][kRow].get_name() in ["rook", "queen"] and
                                    chessBoard[iterator][kRow].get_color() == "black"):
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[0] + 1
                    while iterator < 8:
                        if chessBoard[iterator][kRow] and not (iterator == originalLocation[0]):
                            if (chessBoard[iterator][kRow].get_name() in ["rook", "queen"] and
                                    chessBoard[iterator][kRow].get_color() == "black"):
                                return True
                            return False
                        iterator += 1
                    return False
            elif kCol == originalLocation[0]:
                if kRow < originalLocation[1]:
                    iterator = originalLocation[1] - 1
                    while iterator > -1:
                        if chessBoard[kCol][iterator] and not (iterator == originalLocation[1]):
                            if (chessBoard[kCol][iterator].get_name() in ["rook", "queen"] and
                                    chessBoard[kCol][iterator].get_color() == "black"):
                                return True
                            return False
                        iterator -= 1
                    return False
                else:
                    iterator = originalLocation[1] + 1
                    while iterator < 8:
                        if chessBoard[kCol][iterator] and not (iterator == originalLocation[1]):
                            if (chessBoard[kCol][iterator].get_name() in ["rook", "queen"] and
                                    chessBoard[kCol][iterator].get_color() == "black"):
                                return True
                            return False
                        iterator += 1
                    return False
            elif abs(kRow - originalLocation[1]) == abs(kCol - originalLocation[0]):
                return False
            elif kRow + kCol == originalLocation[1] + originalLocation[0]:
                return False
            else:
                return False
    return False


def updateChessPiece(piece, newLocation):
    chessBoard[piece.get_position()[0]][piece.get_position()[1]] = None
    chessBoard[newLocation[0]][newLocation[1]] = piece
    piece.position = newLocation


# Initialize pieces
bPawns = [Pawn([1, z], "black", True, False) for z in range(WIDTH)]
wPawns = [Pawn([6, z], "white", True, False) for z in range(WIDTH)]

bRooks = [Rook([0, 0], "black", True), Rook([0, 7], "black", True)]
wRooks = [Rook([7, 0], "white", True), Rook([7, 7], "white", True)]

bNights = [Night([0, 1], "black"), Night([0, 6], "black")]
wNights = [Night([7, 1], "white"), Night([7, 6], "white")]

bBishops = [Bishop([0, 2], "black"), Bishop([0, 5], "black")]
wBishops = [Bishop([7, 2], "white"), Bishop([7, 5], "white")]

bQueen = [Queen([0, 3], "black")]
wQueen = [Queen([7, 3], "white")]

bKing = [King([0, 4], "black", True, False)]
wKing = [King([7, 4], "white", True, False)]

# Place all pieces on the board
for pieces in [bPawns, wPawns, bRooks, wRooks, bNights, wNights,
               bBishops, wBishops, bQueen, wQueen, bKing, wKing]:
    for piece in pieces:
        updateChessPiece(piece, piece.get_position())

# Game state variables
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
            return (int(7 - (mouse_pos[1] - 50) // square_width),
                    int(7 - (mouse_pos[0] - 50) // square_width))
    return None


def draw_timer(screen, white_time, black_time):
    """Draw timers for both players"""
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)

    timer_width = 100
    timer_height = 40
    border_width = 2
    board_right_edge = 50 + 650

    # Timer positions
    black_x = board_right_edge - timer_width
    black_y = 5
    white_x = board_right_edge - timer_width
    white_y = 705

    # Draw timer backgrounds
    for x, y in [(black_x, black_y), (white_x, white_y)]:
        pygame.draw.rect(screen, (0, 0, 0),
                         (x - border_width, y - border_width,
                          timer_width + 2 * border_width, timer_height + 2 * border_width))
        pygame.draw.rect(screen, (255, 255, 255),
                         (x, y, timer_width, timer_height))

    def format_time(milliseconds):
        total_seconds = milliseconds / 1000
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        tenths = int((total_seconds * 10) % 10)
        return f"{minutes:02d}:{seconds:02d}.{tenths}"

    # Render time text
    black_text = font.render(format_time(black_time), True, (0, 0, 0))
    white_text = font.render(format_time(white_time), True, (0, 0, 0))

    def center_text(text, box_x, box_y, box_width, box_height):
        return (box_x + (box_width - text.get_width()) // 2,
                box_y + (box_height - text.get_height()) // 2)

    screen.blit(black_text, center_text(black_text, black_x, black_y, timer_width, timer_height))
    screen.blit(white_text, center_text(white_text, white_x, white_y, timer_width, timer_height))


def is_valid_move(selected_piece, new_i, new_j):
    """Check if move is valid based on piece's legal moves"""
    target_pos = [new_i, new_j]
    return any(move == target_pos for move in selected_piece.return_legal_moves())


def draw_legal_moves(screen, selected_piece, square_width, side, chessBoard):
    if not selected_piece:
        return

    for square in selected_piece.return_legal_moves():
        # Create transparent surface for the indicator
        circle_surface = pygame.Surface((square_width, square_width), pygame.SRCALPHA)
        circle_surface.set_alpha(85)
        circle_pos = (square_width / 2, square_width / 2)

        # Handle board orientation
        display_square = square.copy()
        if not side:
            temp = display_square[0]
            display_square[1] = 7 - display_square[1]
            display_square[0] = 7 - temp

        # Draw different indicators for empty vs capturable squares
        if not chessBoard[square[0]][square[1]]:
            # Small filled circle for empty squares
            pygame.draw.circle(circle_surface, (75, 75, 75), circle_pos, 13)
        else:
            # Large hollow circle for capture squares
            pygame.draw.circle(circle_surface, (75, 75, 75), circle_pos, square_width / 2, 6)

        # Position and draw the indicator
        screen.blit(circle_surface,
                    (50 + display_square[1] * square_width,
                     50 + display_square[0] * square_width))

# Main game loop
while True:
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
                    if chessBoard[i][j]:
                        if ((chessBoard[i][j].get_color() == "white" and current_player == 1) or
                                (chessBoard[i][j].get_color() == "black" and current_player == 2)):
                            selected_piece = chessBoard[i][j]
                            selected_i = i
                            selected_j = j
                else:
                    if not dragging:
                        if (i != selected_i or j != selected_j):
                            if is_valid_move(selected_piece, i, j):
                                updateChessPiece(selected_piece, [i, j])
                                current_player = 1 if current_player == 2 else 2

                        if i == selected_i and j == selected_j:
                            selected_piece = None
                            selected_i = None
                            selected_j = None
                            continue

                        selected_piece = None
                        selected_i = None
                        selected_j = None

        elif event.type == pygame.MOUSEMOTION:
            if selected_piece and not dragging:
                current_pos = pygame.mouse.get_pos()
                if initial_click_pos:
                    dx = current_pos[0] - initial_click_pos[0]
                    dy = current_pos[1] - initial_click_pos[1]
                    if (dx * dx + dy * dy) > 25:
                        dragging = True
                        chessBoard[selected_i][selected_j] = None

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                board_pos = get_board_position(pygame.mouse.get_pos())
                if board_pos:
                    new_i, new_j = board_pos
                    if is_valid_move(selected_piece, new_i, new_j):
                        updateChessPiece(selected_piece, [new_i, new_j])
                        current_player = 1 if current_player == 2 else 2
                    else:
                        chessBoard[selected_i][selected_j] = selected_piece

                else:
                    chessBoard[selected_i][selected_j] = selected_piece

                selected_piece = None
                selected_i = None
                selected_j = None
                dragging = False

            initial_click_pos = None

    # Update timers
    if current_player == 1:
        WHITE_TIMELEFT -= clock.get_time()
        WHITE_TIMELEFT = max(0, WHITE_TIMELEFT)
    else:
        BLACK_TIMELEFT -= clock.get_time()
        BLACK_TIMELEFT = max(0, BLACK_TIMELEFT)

    # Draw game state
    screen.fill(background_color)
    screen.blit(background_image, (50, 50))

    if selected_piece:
        if side:
            highlight_rect = (50 + selected_j * square_width,
                              50 + selected_i * square_width,
                              square_width + 1, square_width + 1)
        else:
            highlight_rect = (50 + (7 - selected_j) * square_width,
                              50 + (7 - selected_i) * square_width,
                              square_width + 1, square_width + 1)
        pygame.draw.rect(screen, (255, 255, 197), highlight_rect)

    draw_timer(screen, WHITE_TIMELEFT, BLACK_TIMELEFT)

    # Draw pieces
    for i in range(WIDTH):
        for j in range(WIDTH):
            if chessBoard[i][j]:
                piece = chessBoard[i][j]
                if side:
                    x_val = 50 + j * square_width
                    y_val = 50 + i * square_width
                    screen.blit(piece.get_image(), (x_val, y_val))
                else:
                    x_val = 50 + (7-j) * square_width
                    y_val = 50 + (7-i) * square_width
                    screen.blit(piece.get_image(), (x_val, y_val))

    # Draw dragged piece
    if dragging and selected_piece:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen.blit(selected_piece.get_image(),
                   (mouse_x - background_image.get_width() // 16,
                    mouse_y - background_image.get_width() // 16))

    # Draw legal moves
    draw_legal_moves(screen, selected_piece, square_width, side, chessBoard)


    pygame.display.flip()
    clock.tick(60)