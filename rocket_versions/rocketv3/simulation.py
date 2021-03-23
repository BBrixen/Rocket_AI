import pygame
from physics import all_value_names

white = (255, 255, 255)
light_green = (100, 255, 100)
blue = (100, 100, 255)
black = (0, 0, 0)
X = 1000
Y = 1000
pygame.init()
display = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Rocket GUI')
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
clock_speed = 60  # frames per second

pixels = []
old_pixels = []


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

    for pixel in old_pixels:
        display.set_at(pixel, blue)
    for pixel in pixels:
        display.set_at(pixel, light_green)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True

    pygame.display.update()


def launch(rocket, iteration):
    global pixels
    global old_pixels

    pixels = []
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
        clock.tick(clock_speed)

    # for pixel in pixels:
    #     old_pixels.append(pixel)
    rocket.has_crashed()


def add_pixels(state):
    global pixels

    x = X - int(state['x'] / 10)
    y = Y - int(state['y'])  # this is / by nothing because y is very low
    pixels.append((x, y))

    # making it a 2x2 box
    pixels.append((x+1, y))
    pixels.append((x, y+1))
    pixels.append((x+1, y+1))
    return pixels
