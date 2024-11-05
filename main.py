import pygame

pygame.init()

screen = pygame.display.set_mode((1280,720))

clock = pygame.time.Clock()

my_rect = pygame.Rect((0, 0, 100, 100))

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