import pygame

pygame.init()

screen = pygame.display.set_mode((650,650))

clock = pygame.time.Clock()

background_image = pygame.image.load("ChessBoard.png")
background_image = pygame.transform.scale(background_image, (650, 650))


while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("hi")

