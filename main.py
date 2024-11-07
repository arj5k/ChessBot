import pygame

pygame.init()

screen = pygame.display.set_mode((750,750))

clock = pygame.time.Clock()

background_image = pygame.image.load("ChessBoard.png")
background_image = pygame.transform.scale(background_image, (650, 650))
background_color = (48,46,43)


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

    # Render the graphics here.
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
