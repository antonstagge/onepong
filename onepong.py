#!/usr/bin/python3
import pygame
import sys
import numpy as np
from enum import Enum
import time

# CONSTANTS
# state size
ROWS = 21
COLUMNS = 11
# pad size
PAD_SIZE = 3
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [280, 530]
# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
# This sets the WIDTH and HEIGHT of each state "pixel"
WIDTH = 20
HEIGHT = 20
# This sets the margin between each cell
MARGIN = 5


def main():
    player = True
    if len(sys.argv) > 1:
        print(sys.argv)
        if "-ai" in sys.argv:
            player = False

    s = State(ROWS, COLUMNS)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    # loop condition
    done = False
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # -------- Main Program Loop -----------
    while not done:

        done = s.move()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    s._move_pad(Movement.L)
                if event.key == pygame.K_RIGHT:
                    s._move_pad(Movement.R)
                break


        # Set the screen background
        screen.fill(BLACK)

        # Draw the state
        for row in range(0,ROWS):
            for column in range(0, COLUMNS):
                color = WHITE
                if s._state[row][column] == 1:
                    color = GREEN
                elif s._state[row][column] == 2:
                    color = RED
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        # Limit to 60 frames per second
        clock.tick(6)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    pygame.quit()

    print("SCORE: ", s.points)

class State(object):
    def __init__(self, r, c):
        self._state = np.zeros((r, c), dtype=int)
        self._position = (0, 0) # TODO (int(r/2),int(c/2))
        self._direction = Movement.LU #c TODO D
        self._state[self._position[0]][self._position[1]] = 1
        self._pad = 4
        for i in range(0, PAD_SIZE):
            self._state[self._state.shape[0]-1][self._pad + i] = 2
        self.points = 0

    def move(self):
        if self._is_reg_move():
            self._move_ball_reg()
            return False
        elif self._is_bounce_move():
            self._move_ball_bounce()
            return False
        else:
            # bottom bounce
            new_pos = self._get_new_pos()
            bottom_value = self._state[new_pos[0]][new_pos[1]]
            if bottom_value == 0:
                return True
            self.points += 1
            self._move_ball_bounce()

    def _get_new_pos(self):
        return (self._position[0] + self._direction[0], self._position[1] + self._direction[1])

    def _move_ball_reg(self):
        new_pos = self._get_new_pos()
        self._state[self._position[0]][self._position[1]] = 0
        self._state[new_pos[0]][new_pos[1]] = 1
        self._position = new_pos

    def _move_ball_bounce(self):
        new_pos = self._get_new_pos()
        new_move = (self._direction[0],self._direction[1])
        if (new_pos[1] < 0 or new_pos[1] >= self._state.shape[1]):
            new_move = (new_move[0], new_move[1]*-1)

        if (new_pos[0] < 0 or new_pos[0] >= self._state.shape[0]-1):
            new_move = (new_move[0]*-1, new_move[1])

        self._direction = new_move
        self._move_ball_reg()


    def _is_reg_move(self):
        new_pos = self._get_new_pos()
        return (new_pos[0] < self._state.shape[0] -1 and new_pos[0] >= 0
            and new_pos[1] < self._state.shape[1] and new_pos[1] >= 0)

    def _is_bounce_move(self):
        new_pos = self._get_new_pos()
        return ((new_pos[1] >= self._state.shape[1] or new_pos[1] < 0
            or new_pos[0] < 0) and new_pos[0] < self._state.shape[0]-1)

    def __str__(self):
        return (self._state.__str__())

    def _move_pad(self, movement):
        if movement == Movement.R:
            if self._pad < self._state.shape[1]-PAD_SIZE:
                self._state[self._state.shape[0]-1][self._pad] = 0
                self._pad += 1
                for i in range(0, PAD_SIZE):
                    self._state[self._state.shape[0]-1][self._pad + i] = 2
        elif movement == Movement.L:
            if self._pad > 0:
                self._state[self._state.shape[0]-1][self._pad + (PAD_SIZE-1)] = 0
                self._pad -= 1
                for i in range(0, PAD_SIZE):
                    self._state[self._state.shape[0]-1][self._pad + i] = 2

class Movement():
    U = (-1,0)
    UR = (-1, 1)
    R = (0,1)
    RD = (1, 1)
    D = (1,0)
    DL = (1, -1)
    L = (0,-1)
    LU = (-1,-1)




if __name__ == "__main__":
    main()
