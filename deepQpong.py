#!/usr/bin/python3
import pygame
import sys
import numpy as np
from onepong import *

FPS = 5
# colors
BACKGROUND_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
BALL_COLOR = (0, 0, 255)
RAD_COLOR = (255, 0, 0)

# This sets the WIDTH and HEIGHT of each state "pixel"
WIDTH = 10
HEIGHT = 10
MARGIN = 1
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [COLUMNS*(WIDTH+MARGIN) +MARGIN, ROWS*(HEIGHT + MARGIN) + MARGIN]

def main():
    player = True
    draw = True
    if len(sys.argv) > 1:
        if "-ai" in sys.argv:
            player = False
            draw = False
        if "-d" in sys.argv:
            draw = True

    play(1, player, draw)


def play(times, player, draw=False):
    for t in range(0, times):
        # Initialize board
        state = State(ROWS, COLUMNS)

        # Initialize pygame
        if draw:
            pygame.init()
            pygame.font.init() # you have to call this at the start,
                               # if you want to use this module.
            small_id_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
            screen = pygame.display.set_mode(WINDOW_SIZE)
            # Used to manage how fast the screen updates
            clock = pygame.time.Clock()

        # loop condition
        done = False

        # -------- Main Program Loop -----------
        while not done:
            done_draw = False
            if draw and player:
                done_draw = draw_and_play(state, screen, clock, small_id_font)
            elif draw:
                done_draw = draw_only(state, screen, clock, small_id_font)

            done_play = state.run()
            done = done_draw or done_play

        print("SCORE: ", state.points)
    pygame.quit()

def draw_and_play(s, screen, clock, font):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("abort")
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                print("abort")
                return True
            if event.key == pygame.K_LEFT:
                s.move_pad(Movement.PAD_L)
            if event.key == pygame.K_RIGHT:
                s.move_pad(Movement.PAD_R)
            #break
    actual_draw(s, screen, clock, font)

def actual_draw(s, screen, clock, font):
    # Set the screen background
    screen.fill(BACKGROUND_COLOR)
    # Draw the state
    for row in range(0,ROWS):
        for column in range(0, COLUMNS):
            color = WHITE
            if s._state[row][column] == 1:
                color = BALL_COLOR
            elif s._state[row][column] == 2:
                color = RAD_COLOR
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

    text_id = font.render(("Points: " + str(s.points)), False, (0, 0, 0))
    screen.blit(text_id, (0, 0))
    # Limit to 6 frames per second
    clock.tick(FPS)
    pygame.display.flip()
    return False

def draw_only(s, screen, clock, font):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("abort")
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                print("abort")
                return True

    actual_draw(s, screen, clock, font)


if __name__ == "__main__":
    main()
    # observation - move - prediction - reward
