#!/usr/bin/python3
import pygame
import sys
import numpy as np
from enum import Enum
import time

# CONSTANTS

ROWS = 40
COLUMNS = 100

PAD_SIZE = 8

EMPTY = 0
PAD = 2
BALL = 1

class State(object):
    def __init__(self, r, c):
        self._state = np.zeros((r, c), dtype=int)
        self._position = (int(r/2),int(c/2))
        self._direction = Movement.d_45
        self._state[self._position[0]][self._position[1]] = BALL
        self._pad = round(COLUMNS/2 - PAD_SIZE/2)
        for i in range(0, PAD_SIZE):
            self._state[ROWS-1][self._pad + i] = PAD
        self.points = 0

    def run(self):
        if self._is_reg_move():
            self._move_ball_reg()
            return False
        elif self._is_bounce_move():
            self._move_ball_bounce()
            return False
        elif self._is_pad_bounce():
            self._move_ball_pad_bounce()
            self.points += 1
        else:
            # Crash return done is true
            return True

    # Returns the balls next position
    def _get_new_pos(self):
        return (self._position[0] + self._direction[0], self._position[1] + self._direction[1])

    # Return true if the ball is inside the court
    def _is_reg_move(self):
        new_pos = self._get_new_pos()
        return (new_pos[0] < ROWS -1 and new_pos[0] >= 0
            and new_pos[1] < COLUMNS and new_pos[1] >= 0)

    # Move the ball to the next position inside the court
    def _move_ball_reg(self):
        new_pos = self._get_new_pos()
        self._state[self._position[0]][self._position[1]] = EMPTY
        self._state[new_pos[0]][new_pos[1]] = BALL
        self._position = new_pos

    # True if ball is about to bounce on PAD
    def _is_pad_bounce(self):
        row_idx = self._position[0] + 1
        if row_idx == ROWS-1:
            for i in range(0, 5):
                col_idx = self._position[1] + i - 2
                if col_idx >= 0 and col_idx < COLUMNS:
                    if self._state[row_idx][col_idx] == PAD:
                        return True
        row_idx += 1
        if row_idx == ROWS-1:
            for i in range(0, 5):
                col_idx = self._position[1] + i - 2
                if col_idx >= 0 and col_idx < COLUMNS:
                    if self._state[row_idx][col_idx] == PAD:
                        return True
        return False

    # Returns the type of bounce move
    def _is_bounce_move(self):
        new_pos = self._get_new_pos()
        return ((new_pos[1] >= COLUMNS or new_pos[1] < 0
            or new_pos[0] < 0) and new_pos[0] < ROWS-1)

    def _move_ball_bounce(self):
        print("bounce move")
        new_pos = self._get_new_pos()
        # Swap direction
        if (new_pos[1] < 0 or new_pos[1] >= COLUMNS):
            # hit sides
            print("direction hit side")
            self._direction = (self._direction[0], -1*self._direction[1])
        if (new_pos[0] < 0 or new_pos[0] > ROWS-1):
            print("direction hit roof or btn")
            self._direction = (-1*self._direction[0],self._direction[1])

        print("pos before")
        print(self._position)
        print(new_pos)
        if new_pos[0] == -2:
            # hit roof
            new_pos = (1, new_pos[1])
            print("hit roof")
            print(new_pos)
        else:
            new_pos = (self._position[0], new_pos[1])

        if new_pos[1] == -2:
            # hit left side
            new_pos = (new_pos[0], 1)
            print("hit left")
            print(new_pos)
        elif new_pos[1] == COLUMNS +1:
            new_pos = (new_pos[0], COLUMNS-3)
            print("hit right")
            print(new_pos)
        else:
            new_pos = (new_pos[0], self._position[1])

        print("done")
        print(new_pos)
        self._state[self._position[0]][self._position[1]] = EMPTY
        self._state[new_pos[0]][new_pos[1]] = BALL
        self._position = new_pos



    def __str__(self):
        return (self._state.__str__())

    def move_pad(self, movement):
        if movement == Movement.PAD_R:
            if self._pad < self._state.shape[1]-PAD_SIZE:
                self._state[self._state.shape[0]-1][self._pad] = EMPTY
                self._pad += 1
                for i in range(0, PAD_SIZE):
                    self._state[self._state.shape[0]-1][self._pad + i] = PAD
        elif movement == Movement.PAD_L:
            if self._pad > 0:
                self._state[self._state.shape[0]-1][self._pad + (PAD_SIZE-1)] = EMPTY
                self._pad -= 1
                for i in range(0, PAD_SIZE):
                    self._state[self._state.shape[0]-1][self._pad + i] = PAD

# 135 120  90  60  45
# 150              30
# 180               0
# 210             330
# 225 240 270 300 315
class Movement():
    d_135 = (-2, -2)
    d_120 = (-2, -1)
    d_90 = (-2, 0)
    d_60 = (-2, 1)
    d_45 = (-2, 2)
    d_150 = (-1, -2)
    d_30 = (-1, 2)
    d_180 = (0, -2)
    d_0 = (0, 2)
    d_210 = (1, -2)
    d_330 = (1, 2)
    d_225 = (2, -2)
    d_240 = (2, -1)
    d_270 = (2, 0)
    d_300 = (2, 1)
    d_315 = (2, 2)
    PAD_L = (0 ,-1)
    PAD_R = (0, 1)
