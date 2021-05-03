import pygame
from physics import bounds
import math
import sys

white = (255, 255, 255)
light_green = (100, 255, 100)
blue = (100, 100, 255)
black = (0, 0, 0)
orange = (252, 115, 3)  # fuel color
X = 1500
Y = 1000
pygame.init()
display = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Rocket GUI')
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
CLOCK_SPEED = 100  # frames per second

x_pixels = set()
z_pixels = set()


def handle_display(state, iteration, MULTIPLE_AI):
    LINE_LENGTH = 50

    display.fill(black)

    # iteration
    iteration_str = 'Iteration: ' + str(iteration)
    iteration_text = font.render(iteration_str, True, white, black)
    iteration_text_rect = iteration_text.get_rect()
    iteration_text_rect.center = (100, LINE_LENGTH)
    display.blit(iteration_text, iteration_text_rect)

    num_ai_str = 'one AI'
    if MULTIPLE_AI:
        num_ai_str = 'multiple AIs'
    num_ai_text = font.render(num_ai_str, True, white, black)
    num_ai_text_rect = num_ai_text.get_rect()
    num_ai_text_rect.center = (100, 4*LINE_LENGTH)
    display.blit(num_ai_text, num_ai_text_rect)

    # velocity in x:
    vel_x_text = font.render('x velocity:', True, white, black)
    vel_x_text_rect = vel_x_text.get_rect()
    vel_x_text_rect.center = (400, LINE_LENGTH)
    display.blit(vel_x_text, vel_x_text_rect)
    vel_center = (540, LINE_LENGTH)
    vel_line_length = abs(state['velocity']/bounds['max_velocity']) * LINE_LENGTH
    angle = state['x_direction'] * 2 * math.pi
    vel_x_component = -1*vel_line_length * math.cos(angle) + vel_center[0]
    vel_y_component = -1*vel_line_length * math.sin(angle) + vel_center[1]
    vel_end = (vel_x_component, vel_y_component)
    pygame.draw.circle(display, white, vel_center, LINE_LENGTH, width=1)
    pygame.draw.line(display, white, vel_center, vel_end)

    # velocity in z:
    vel_z_text = font.render('z velocity:', True, white, black)
    vel_z_text_rect = vel_z_text.get_rect()
    vel_z_text_rect.center = (770, LINE_LENGTH)
    display.blit(vel_z_text, vel_z_text_rect)
    vel_center = (920, LINE_LENGTH)
    angle = state['z_direction'] * 2 * math.pi
    vel_x_component = -1 * vel_line_length * math.cos(angle) + vel_center[0]
    vel_y_component = -1 * vel_line_length * math.sin(angle) + vel_center[1]
    vel_end = (vel_x_component, vel_y_component)
    pygame.draw.circle(display, white, vel_center, LINE_LENGTH, width=1)
    pygame.draw.line(display, white, vel_center, vel_end)

    # acceleration in x:
    acc_x_text = font.render('x acceleration:', True, white, black)
    acc_x_text_rect = acc_x_text.get_rect()
    acc_x_text_rect.center = (370, 4*LINE_LENGTH)
    display.blit(acc_x_text, acc_x_text_rect)
    acc_center = (550, 4*LINE_LENGTH)
    acc_line_length = abs(state['acceleration']/bounds['max_acceleration']) * LINE_LENGTH
    angle = state['x_tilt'] * 2 * math.pi
    acc_x_component = -1 * acc_line_length * math.cos(angle) + acc_center[0]
    acc_y_component = -1 * acc_line_length * math.sin(angle) + acc_center[1]
    acc_end = (acc_x_component, acc_y_component)
    pygame.draw.circle(display, white, acc_center, LINE_LENGTH, width=1)
    pygame.draw.line(display, white, acc_center, acc_end)

    # acceleration in z:
    acc_z_text = font.render('z acceleration:', True, white, black)
    acc_z_text_rect = acc_z_text.get_rect()
    acc_z_text_rect.center = (740, 4*LINE_LENGTH)
    display.blit(acc_z_text, acc_z_text_rect)
    acc_center = (920, 4*LINE_LENGTH)
    angle = state['z_tilt'] * 2 * math.pi
    acc_x_component = -1 * acc_line_length * math.cos(angle) + acc_center[0]
    acc_y_component = -1 * acc_line_length * math.sin(angle) + acc_center[1]
    acc_end = (acc_x_component, acc_y_component)
    pygame.draw.circle(display, white, acc_center, LINE_LENGTH, width=1)
    pygame.draw.line(display, white, acc_center, acc_end)

    # wind
    wind_text = font.render('wind:', True, white, black)
    wind_text_rect = wind_text.get_rect()
    wind_text_rect.center = (1100, LINE_LENGTH)
    display.blit(wind_text, wind_text_rect)
    wind_center = (1200, LINE_LENGTH)
    wind_line_length = abs(state['wind']) * LINE_LENGTH
    angle = state['wind_direction'] * 2 * math.pi
    wind_x_component = -1 * wind_line_length * math.cos(angle) + wind_center[0]
    wind_y_component = -1 * wind_line_length * math.sin(angle) + wind_center[1]
    wind_end = (wind_x_component, wind_y_component)
    pygame.draw.circle(display, white, wind_center, LINE_LENGTH, width=1)
    pygame.draw.line(display, white, wind_center, wind_end)

    # fuel
    fuel_text = font.render('fuel:', True, white, black)
    fuel_text_rect = fuel_text.get_rect()
    fuel_text_rect.center = (1100, 4*LINE_LENGTH)
    display.blit(fuel_text, fuel_text_rect)
    fuel_center = (1200, 3*LINE_LENGTH)
    fuel_size = (LINE_LENGTH/2, LINE_LENGTH*2)
    fuel_ratio = state['fuel']/bounds['max_fuel']
    fuel_height = int(2*LINE_LENGTH * fuel_ratio)
    actual_size = (LINE_LENGTH/2, fuel_height)
    outline_rect = pygame.Rect(fuel_center[0], fuel_center[1], fuel_size[0], fuel_size[1])
    starting_y = fuel_center[1] + (1 - fuel_ratio) * fuel_size[1]
    actual_rect = pygame.Rect(fuel_center[0], starting_y, actual_size[0], actual_size[1])
    pygame.draw.rect(display, white, outline_rect, width=1)
    pygame.draw.rect(display, orange, actual_rect)

    for pixel in x_pixels:
        display.set_at(pixel, light_green)
    for pixel in z_pixels:
        display.set_at(pixel, light_green)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True

    pygame.display.update()


def launch(rocket, iteration, MULTIPLE_AI):
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
        user_exit = handle_display(state, iteration, MULTIPLE_AI)
        # clock.tick(CLOCK_SPEED)  # remove this for no set fps
    if user_exit:
        sys.exit("User exited")

    rocket.has_crashed()


def add_pixels(state):
    global x_pixels, z_pixels

    x = int((X/2)) - int(state['x'] / 100)  # X/2 to get the first half of the screen
    z = X - int(state['z'] / 100)   # not dividing to get 2nd half of the screen
    y = Y - int(state['y'] / 5)
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
