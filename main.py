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


    # Do logical updates here.
    screen.blit(background_image, (0, 0))

    screen.fill('pink')  # Fill the display with a solid color
    screen.blit(background_image, (0, 0))

    # Render the graphics here.


    pygame.display.flip()  # Refresh on-screen display
    clock.tick(60)         # wait until next frame (at 60 FPS)