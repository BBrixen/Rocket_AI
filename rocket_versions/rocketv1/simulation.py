# import numpy as np
# import random
import pygame

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
X = 1000
Y = 1000
pygame.init()
display = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Rocket GUI')
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
clock.tick(60)


def handle_display(state, iteration):
    display.fill(black)

    # height
    height_str = 'Height: ' + str(round(state['height'], 1))
    height_text = font.render(height_str, True, white, black)
    height_text_rect = height_text.get_rect()
    height_text_rect.center = (200, 40)
    display.blit(height_text, height_text_rect)

    # velocity
    velocity_str = 'Velocity: ' + str(round(state['velocity'], 1))
    velocity_text = font.render(velocity_str, True, white, black)
    velocity_text_rect = velocity_text.get_rect()
    velocity_text_rect.center = (185, 80)
    display.blit(velocity_text, velocity_text_rect)

    # acceleration
    acceleration_str = 'Acceleration: ' + str(round(state['acceleration'], 1))
    acceleration_text = font.render(acceleration_str, True, white, black)
    acceleration_text_rect = acceleration_text.get_rect()
    acceleration_text_rect.center = (150, 120)
    display.blit(acceleration_text, acceleration_text_rect)

    # fuel
    fuel_str = 'Fuel: ' + str(round(state['fuel'], 1))
    fuel_text = font.render(fuel_str, True, white, black)
    fuel_text_rect = fuel_text.get_rect()
    fuel_text_rect.center = (215, 160)
    display.blit(fuel_text, fuel_text_rect)

    # iteration
    iteration_str = 'Iteration: ' + str(iteration)
    iteration_text = font.render(iteration_str, True, white, black)
    iteration_text_rect = iteration_text.get_rect()
    iteration_text_rect.center = (180, 200)
    display.blit(iteration_text, iteration_text_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True

    pygame.display.update()


def launch(rocket, iteration):
    crashed = False
    user_exit = False
    while True:
        if crashed or user_exit:
            break
        state, crashed = rocket.fly()
        if crashed or user_exit:
            break
        user_exit = handle_display(state, iteration)

    rocket.has_crashed()
