import pygame

pygame.init()

screen = pygame.display.set_mode((1280,720))

clock = pygame.time.Clock()

my_rect = pygame.Rect((0, 0, 100, 100))

NAMES = {
    1:"pawn",
    2:"knight",
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
WIDTH = 8
chessboard = [[0 for x in range(WIDTH)] for y in range(WIDTH)]

for z in range(WIDTH):
    chessboard[1][z] = Piece(1, "black")
    chessboard[6][z] = Piece(1,"white")
chessboard[0][0] = Piece(4, "black")
chessboard[7][0] = Piece(4, "white")
chessboard[0][1] = Piece(2,"black")
chessboard[7][1] = Piece(2,"white")
chessboard[0][2] = Piece(3, "black")
chessboard[7][2] = Piece(3, "white")
chessboard[0][3] = Piece(5,"black")
chessboard[7][3] = Piece(5,"white")
chessboard[0][4] = Piece(6,"black")
chessboard[7][4] = Piece(6,"white")
chessboard[0][5] = Piece(3, "black")
chessboard[7][5] = Piece(3,"white")
chessboard[0][6] = Piece(2,"black")
chessboard[7][6] = Piece(2,"white")
chessboard[0][7] = Piece(4, "black")
chessboard[7][7] = Piece(4, "black")


while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                my_rect.x = 0

    # Do logical updates here.
    my_rect.x += 4

    screen.fill('pink')  # Fill the display with a solid color

    # Render the graphics here.
    pygame.draw.rect(screen, (0, 255, 0), my_rect)

    pygame.display.flip()  # Refresh on-screen display
    clock.tick(60)         # wait until next frame (at 60 FPS)