import pygame
from physics import all_value_names

white = (255, 255, 255)
light_green = (100, 255, 100)
blue = (100, 100, 255)
black = (0, 0, 0)
X = 1500
Y = 1000
pygame.init()
display = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Rocket GUI')
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
CLOCK_SPEED = 60  # frames per second

x_pixels = set()
z_pixels = set()


def handle_display(state, iteration):
    display.fill(black)
    y = 0
    base_x = 250
    x_change_per_letter = 8
    for name in all_value_names:
        new_x = base_x - x_change_per_letter * len(name)
        text_str = name + ":" + str(round(state[name], 3))
        text = font.render(text_str, True, white, black)
        text_rect = text.get_rect()
        y += 40
        text_rect.center = (new_x, y)
        display.blit(text, text_rect)

    # iteration
    iteration_str = 'Iteration: ' + str(iteration)
    iteration_text = font.render(iteration_str, True, white, black)
    iteration_text_rect = iteration_text.get_rect()
    iteration_text_rect.center = (180, 40*(len(all_value_names)+1))
    display.blit(iteration_text, iteration_text_rect)

    for pixel in x_pixels:
        display.set_at(pixel, light_green)
    for pixel in z_pixels:
        display.set_at(pixel, light_green)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True

    pygame.display.update()


def launch(rocket, iteration):
    global x_pixels, z_pixels

    x_pixels = set()
    z_pixels = set()
    crashed = False
    user_exit = False
    while True:
        if crashed or user_exit:
            break
        state, crashed = rocket.fly()
        add_pixels(state)
        if crashed or user_exit:
            break
        user_exit = handle_display(state, iteration)
        clock.tick(CLOCK_SPEED)
    if user_exit:
        return

    rocket.has_crashed()


def add_pixels(state):
    global x_pixels, z_pixels

    x = int((X/2)) - int(state['x'] / 20)  # X/2 to get the first half of the screen
    z = X - int(state['z'] / 20)   # not dividing to get 2nd half of the screen
    y = Y - int(state['y'])  # this is / by nothing because y is very low
    x_pixels.add((x, y))
    z_pixels.add((z, y))

    # making it a 2x2 box
    x_pixels.add((x + 1, y))
    x_pixels.add((x, y + 1))
    x_pixels.add((x + 1, y + 1))

    z_pixels.add((z + 1, y))
    z_pixels.add((z, y + 1))
    z_pixels.add((z + 1, y + 1))

    # making it a 3x3 box
    x_pixels.add((x + 2, y))
    x_pixels.add((x + 2, y + 1))
    x_pixels.add((x, y + 2))
    x_pixels.add((x + 1, y + 2))
    x_pixels.add((x + 2, y + 2))

    z_pixels.add((z + 2, y))
    z_pixels.add((z + 2, y + 1))
    z_pixels.add((z, y + 2))
    z_pixels.add((z + 1, y + 2))
    z_pixels.add((z + 2, y + 2))
