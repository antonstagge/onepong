#!/usr/bin/python3
import pygame

import numpy as np


# CONSTANTS

ROWS = 40
COLUMNS = 60

PAD_SIZE = 8

EMPTY = 1
PAD = 2
BALL = 3

FPS = 150
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

class State(object):
    def __init__(self):
        self._state = np.zeros((ROWS, COLUMNS), dtype=int)
        self._position = (int(ROWS/2),int(COLUMNS/2))
        start_dir_v = [Movement.d_240, Movement.d_270, Movement.d_300]
        self._direction = Movement.d_300#start_dir_v[np.random.randint(0,len(start_dir_v))]
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
            self._move_ball_bounce(True)
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
        new_pos = self._get_new_pos()
        return self._state[ROWS-1][new_pos[1]] == PAD

    # Returns the type of bounce move
    def _is_bounce_move(self):
        new_pos = self._get_new_pos()
        return ((new_pos[1] >= COLUMNS or new_pos[1] < 0
            or new_pos[0] < 0) and new_pos[0] < ROWS-1)

    def _move_ball_bounce(self, pad=False):
        new_pos = self._get_new_pos()
        first_new_pos = new_pos

        # change direction
        if new_pos[1] < 0 or new_pos[1] > COLUMNS-1:
            self._direction = (self._direction[0],-1*self._direction[1])
        if new_pos[0] < 0:
            self._direction = (-1*self._direction[0],self._direction[1])

        if pad and new_pos[0] > ROWS-2:
            dist = self._position[1] - self._pad
            if dist == 0 or dist == 1:
                self._direction = Movement.d_150
            elif dist == 2:
                self._direction = Movement.d_135
            elif dist == 3:
                self._direction = Movement.d_120
            elif dist == 4:
                self._direction = Movement.d_60
            elif dist == 5:
                self._direction = Movement.d_45
            elif dist == 6 or dist == 7:
                self._direction = Movement.d_30

        if self._position[1] == 1:
            # print("left away")
            new_pos = (new_pos[0], 1)
        elif self._position[1] == 0:
            # print("left now")
            new_pos = (new_pos[0], 2)
        elif self._position[1] == COLUMNS-2:
            # print("right away")
            new_pos = (new_pos[0], COLUMNS-2)
        elif self._position[1] == COLUMNS-1:
            # print("right now")
            new_pos = (new_pos[0], COLUMNS-3)

        if self._position[0] == 1:
            # print("roof away")
            new_pos = (1, new_pos[1])
        elif self._position[0] == 0:
            # print("roof now")
            new_pos = (2, new_pos[1])
        elif self._position[0] == ROWS-3:
            # print("btn away")
            new_pos = (ROWS-3, new_pos[1])
        elif self._position[0] == ROWS-2:
            # print("btn now")
            new_pos = (ROWS-4, new_pos[1])

        if pad:
            new_pos = (ROWS-2, first_new_pos[1])
        # move to new_pos
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
    PAD_STILL = -1

class PlayPong(object):
    def __init__(self, player, draw):
        self.player = player
        self.draw = draw
        self.done = False
        self.state = State()
        if draw:
            pygame.init()
            pygame.font.init() # you have to call this at the start,
                               # if you want to use this module.
            self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
            self.screen = pygame.display.set_mode(WINDOW_SIZE)
            # Used to manage how fast the screen updates
            self.clock = pygame.time.Clock()

    def play_one_pong(self, action = None):
        done_draw = False
        if self.draw and self.player:
            done_draw = draw_and_play(self.state, self.screen, self.clock, self.font)
        elif self.draw:
            done_draw = draw_only(self.state, self.screen, self.clock, self.font)

        if action is not None and not self.player:
            self.state.move_pad(action)

        done_play = self.state.run()
        self.done = done_draw or done_play

        return self.done

    def quit(self):
        pygame.quit()

    def just_show(self):
        actual_draw(self.state, self.screen, self.clock, self.font)

def draw_and_play(s, screen, clock, font):
    moved = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("end game")
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                print("abort")
                return True
            if event.key == pygame.K_LEFT:
                s.move_pad(Movement.PAD_L)
                moved = True
            elif event.key == pygame.K_RIGHT:
                s.move_pad(Movement.PAD_R)
                moved = True

    if not moved:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            s.move_pad(Movement.PAD_L)
        elif keys[pygame.K_RIGHT]:
            s.move_pad(Movement.PAD_R)

    actual_draw(s, screen, clock, font)

def actual_draw(s, screen, clock, font):
    # Set the screen background
    screen.fill(BACKGROUND_COLOR)
    # Draw the state
    for row in range(0,ROWS):
        for column in range(0, COLUMNS):
            color = WHITE
            if s._state[row][column] == BALL:
                color = BALL_COLOR
            elif s._state[row][column] == PAD:
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
            print("end game")
            return True
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        print("end game")
        return True

    actual_draw(s, screen, clock, font)
