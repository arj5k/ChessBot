import pygame

pygame.init()

screen = pygame.display.set_mode((750, 750))
clock = pygame.time.Clock()

background_image = pygame.image.load("ChessBoard.png")
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

NAMES = {
    1: "pawn",
    2: "night",  # should be knight - edited so that the name starts with 'n'
    3: "bishop",
    4: "rook",
    5: "queen",
    6: "king"
}
VALUES = {
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 100000
    }
class Piece:
    def __init__(self, number, color):
        self.number = number
        self.name = NAMES[number]
        self.color = color
        self.value = VALUES[number]
        # Pre-load the piece's image
        self.image = load_piece_image(color, self.name)

    def get_name(self):
        return self.name
    def get_color(self):
        return self.color

    def get_number(self):
        return self.number

    def get_image(self):
        return self.image


WIDTH = 8
chessboard = [[None for x in range(WIDTH)] for y in range(WIDTH)]

# Place pawns
for z in range(WIDTH):
    chessboard[1][z] = Piece(1, "black")
    chessboard[6][z] = Piece(1, "white")

# Place rooks
chessboard[0][0] = Piece(4, "black")
chessboard[7][0] = Piece(4, "white")
chessboard[0][7] = Piece(4, "black")
chessboard[7][7] = Piece(4, "white")

# Place knights
chessboard[0][1] = Piece(2, "black")
chessboard[7][1] = Piece(2, "white")
chessboard[0][6] = Piece(2, "black")
chessboard[7][6] = Piece(2, "white")

# Place bishops
chessboard[0][2] = Piece(3, "black")
chessboard[7][2] = Piece(3, "white")
chessboard[0][5] = Piece(3, "black")
chessboard[7][5] = Piece(3, "white")

# Place queens
chessboard[0][3] = Piece(5, "black")
chessboard[7][3] = Piece(5, "white")

# Place kings
chessboard[0][4] = Piece(6, "black")
chessboard[7][4] = Piece(6, "white")

# Variables for piece selection and movement
selected_i = None
selected_j = None
selected_piece = None
dragging = False
initial_click_pos = None


def get_board_position(mouse_pos):
    """Convert mouse position to board indices"""
    if 50 <= mouse_pos[0] <= 700 and 50 <= mouse_pos[1] <= 700:
        return (int((mouse_pos[1] - 50) // square_width),
                int((mouse_pos[0] - 50) // square_width))
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
                if selected_piece is None:
                    # First click - select piece
                    if chessboard[i][j]:
                        selected_piece = chessboard[i][j]
                        selected_i = i
                        selected_j = j
                else:
                    # Second click - move piece if it's not a drag operation
                    if not dragging:
                        if (i != selected_i or j != selected_j):
                            # Move the piece
                            chessboard[i][j] = selected_piece
                            chessboard[selected_i][selected_j] = None
                        selected_piece = None
                        selected_i = None
                        selected_j = None
                        current_player = 1 if current_player == 2 else 2

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
                        chessboard[selected_i][selected_j] = None

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                # Handle drag and drop movement
                board_pos = get_board_position(pygame.mouse.get_pos())
                if board_pos:
                    new_i, new_j = board_pos
                    chessboard[new_i][new_j] = selected_piece
                else:
                    # If dropped outside the board, return piece to original position
                    chessboard[selected_i][selected_j] = selected_piece

                selected_piece = None
                selected_i = None
                selected_j = None
                dragging = False
                current_player = 1 if current_player == 2 else 2

            initial_click_pos = None

    # Do logical updates here.
    screen.fill(background_color)
    screen.blit(background_image, (50, 50))

    # Highlight selected square if a piece is selected
    if selected_piece:
        pygame.draw.rect(screen, (255, 255, 197),
                         pygame.Rect(50 + selected_j * square_width,
                                     50 + selected_i * square_width,
                                     square_width, square_width))

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
            if chessboard[i][j]:
                piece = chessboard[i][j]
                x_val = 50 + j * square_width
                y_val = 50 + i * square_width
                screen.blit(piece.get_image(), (x_val, y_val))

    # Draw dragged piece only if actually dragging
    if dragging and selected_piece:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen.blit(selected_piece.get_image(),
                    (mouse_x - background_image.get_width() // 16,
                     mouse_y - background_image.get_width() // 16))

    pygame.display.flip()
    clock.tick(60)