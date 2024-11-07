import pygame

pygame.init()

screen = pygame.display.set_mode((750,750))

clock = pygame.time.Clock()

background_image = pygame.image.load("ChessBoard.png")
background_image = pygame.transform.scale(background_image, (650, 650))
background_color = (48,46,43)


WHITE_TIMELEFT = 600000
BLACK_TIMELEFT = 600000

PLAYER = {
    1:"WHITE",
    2:"BLACK"
}

current_player = 1

NAMES = {
    1:"pawn",
    2:"night", #should be knight - edited so that the name starts with 'n'
    3:"bishop",
    4:"rook",
    5:"queen",
    6:"king"
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
    def __init__(self, name, color):
        self.name = NAMES[name]
        self.color = color
        self.value = VALUES[name]
    def get_name(self):
        return self.name
    def get_color(self):
        return self.color
WIDTH = 8
chessboard = [[None for x in range(WIDTH)] for y in range(WIDTH)]

for z in range(WIDTH):
    chessboard[1][z] = Piece(1, "black")
    chessboard[6][z] = Piece(1,"white")
# Set up chessboard (8x8 grid)
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


def draw_timer(screen, white_time, black_time):
    """
    Draw timers for both players.
    white_time and black_time should be in milliseconds
    """
    # Font setup
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 32)

    # Timer box dimensions
    timer_width = 120
    timer_height = 60
    border_width = 2

    # Calculate positions (assuming 750x750 screen)
    # Black timer at top right
    black_x = 750 - timer_width - 20  # 20px padding from right edge
    black_y = 20  # 20px padding from top

    # White timer at bottom right
    white_x = 750 - timer_width - 20
    white_y = 750 - timer_height - 20

    # Draw timer backgrounds with borders
    # Black timer background
    pygame.draw.rect(screen, (0, 0, 0), (black_x - border_width, black_y - border_width,
                                         timer_width + 2 * border_width, timer_height + 2 * border_width))
    pygame.draw.rect(screen, (255, 255, 255), (black_x, black_y, timer_width, timer_height))

    # White timer background
    pygame.draw.rect(screen, (0, 0, 0), (white_x - border_width, white_y - border_width,
                                         timer_width + 2 * border_width, timer_height + 2 * border_width))
    pygame.draw.rect(screen, (255, 255, 255), (white_x, white_y, timer_width, timer_height))

    # Convert milliseconds to minutes and seconds
    def format_time(milliseconds):
        total_seconds = milliseconds // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("hi")

    # Do logical updates here.
    screen.fill(background_color)  # Fill the display with a solid color
    screen.blit(background_image, (50, 50))

    if current_player == 1:  # White's turn
        WHITE_TIMELEFT -= clock.get_time()
    else:  # Black's turn
        BLACK_TIMELEFT -= clock.get_time()

    draw_timer(screen, WHITE_TIMELEFT, BLACK_TIMELEFT)

    for i in range(WIDTH):
        for j in range(WIDTH):
            if chessboard[i][j]:
                piece = chessboard[i][j]
                piece_color = piece.get_color()
                piece_name = piece.get_name()

                # Determine the image based on piece color and name
                piece_key = f"{piece_color[0]}{piece_name[0]}.png"  # e.g., 'wp.png'
                img = pygame.image.load(piece_key)
                img = pygame.transform.scale(img, (background_image.get_width() //8 , background_image.get_height()//8))

                # Calculate position on the screen
                x_val = 50 + j * (background_image.get_width() // WIDTH)
                y_val = 50 + i * (background_image.get_height() // WIDTH)

                # Draw the piece image on the board
                screen.blit(img, (x_val, y_val))

    pygame.display.flip()  # Refresh on-screen display
    clock.tick(60)
