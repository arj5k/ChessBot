import pygame

pygame.init()

screen = pygame.display.set_mode((690,690))

clock = pygame.time.Clock()

background_image = pygame.image.load("ChessBoard.png")
background_image = pygame.transform.scale(background_image, (650, 650))
my_color = (48,46,43)


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
    screen.fill(my_color)  # Fill the display with a solid color
    screen.blit(background_image, (20, 20))

    # Render the graphics here.


    pygame.display.flip()  # Refresh on-screen display
    clock.tick(60)
