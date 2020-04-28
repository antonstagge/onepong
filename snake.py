# All code written by Anton Stagge 2018
# Feel free to use/edit it to what ever extent but please
# keep the original authors name.
import pygame
import numpy as np
import random

# CONSTANTS
ROWS = 30
COLUMNS = 30

EMPTY = 0
SNAKE = 1
COIN = 2

FPS = 20
# colors
BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (255, 255, 255)
COIN_COLOR = (255, 0, 0)
# This sets the WIDTH and HEIGHT of each state "pixel"
WIDTH = 18
HEIGHT = 18
MARGIN = 1
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [COLUMNS*(WIDTH+MARGIN) + MARGIN, ROWS*(HEIGHT+MARGIN) + MARGIN]

# (y, x)
movements = ['L', 'U', 'R', 'D']
move_trans = {
    'L': [0, -1],
    'U': [-1, 0],
    'R': [0, 1],
    'D': [1, 0]
}


class State():
    """
    This class defines how the snake game works,
    how to move the ball and how to move the pad.
    """

    def __init__(self):
        self._state = np.zeros((ROWS, COLUMNS), dtype=int)
        self._pos_head = (int(ROWS/2), int(COLUMNS/2))
        self._state[self._pos_head[0]][self._pos_head[1]] = SNAKE
        self._pos_body = []
        self.set_coin_pos()
        self._dir = random.choice(movements)
        self.points = 0
        self._life = 200

    def run(self):
        """ run one iteration of the GAME """
        if self._is_not_wall() and self._is_not_snake() and self._life > 0:
            self._life -= 1
            last_spot = self._move_snake()

            if self._pos_head == self._coin:
                self.points += 1
                self._life += 100
                # if self.points % 2:
                # make snake longer
                self._pos_body.insert(0, last_spot)
                self.set_coin_pos()
            return False
        else:
            return True

    def set_coin_pos(self):
        self._coin = (random.randint(0, ROWS - 1),
                      random.randint(0, COLUMNS - 1))
        #self._coin = (1, 1)
        self._state[self._coin[0]][self._coin[1]] = COIN

    # Returns the snakes next position
    def _get_new_pos(self):
        """ Return the next position """
        direction = move_trans[self._dir]
        return (self._pos_head[0] + direction[0], self._pos_head[1] + direction[1])

    def _is_not_wall(self):
        """ True if the snake is still inside the map """
        new_pos = self._get_new_pos()
        return (new_pos[0] < ROWS and new_pos[0] >= 0
                and new_pos[1] < COLUMNS and new_pos[1] >= 0)

    def _is_not_snake(self):
        """ True if snake not crash with itself """
        new_pos = self._get_new_pos()
        return new_pos not in self._pos_body

    def _move_snake(self):
        """ Move the ball to the next position inside the map 
            Returns the tail of snake
        """
        new_pos = self._get_new_pos()
        last_pos = None
        if len(self._pos_body):
            last_pos = self._pos_body.pop(0)
        self._pos_body.append(self._pos_head)

        if last_pos:
            self._state[last_pos] = EMPTY
        self._state[new_pos] = SNAKE
        self._pos_head = new_pos
        return last_pos

    def __str__(self):
        return (self._state.__str__())

    def change_dir(self, m):
        if not np.abs(movements.index(m) - movements.index(self._dir)) % 2:
            # can not back
            return
        self._dir = m


class PlaySnake():
    """ Run a game of snake """

    def __init__(self, player=False, draw=False):
        self.player = player  # if False the AI is playing
        self.draw = draw
        self.done = False
        self.state = State()
        self.last_points = 0
        self.current_points = 0
        if draw:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
            self.screen = pygame.display.set_mode(WINDOW_SIZE)
            # Set the screen background
            self.screen.fill(BACKGROUND_COLOR)
            # Used to manage how fast the screen updates
            self.clock = pygame.time.Clock()

    def play_one_iteration(self, action=None):
        """
        Run one iteration of the game,
        if player you draw and take input
        if draw you only draw
        if action != None you take the action
        """
        done_draw = False
        done_play = False
        if self.player:
            done_play = self.play()
        else:
            # AI play
            if action is not None:
                m = movements[action]
                self.state.change_dir(m)

        done_run = self.state.run()

        if self.draw or self.player:
            done_draw = self.draw_state()

        self.done = done_draw or done_play or done_run
        self.last_points = self.current_points
        self.current_points = self.state.points
        return self.done

    def get_observation(self):
        """ Return the vector representation of a pong state
            which is the balls position direction and the pads position.
            - if there’s an immediate danger in the snake’s proximity (right, left and straight).
            - if the snake is moving up, down, left or right.
            - if the food is above, below, on the left or on the right.

            ex: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
                 L  U  R  D  l  u  r  d  o  u  l  r
        """
        obs = []

        # obstacle check
        org_dir = self.state._dir
        for m in movements:
            self.state._dir = m
            obs.append(int(self.state._is_not_wall()
                           and self.state._is_not_snake()))
        self.state._dir = org_dir
        # direction
        for m in movements:
            obs.append(int(m == org_dir))
        # coin check
        obs.append(
            int(self.state._pos_head[0] - self.state._coin[0] < 0))  # above
        obs.append(
            int(self.state._pos_head[0] - self.state._coin[0] > 0))  # under
        obs.append(
            int(self.state._pos_head[1] - self.state._coin[1] < 0))  # left
        obs.append(
            int(self.state._pos_head[1] - self.state._coin[1] > 0))  # right

        return obs

    def get_reward(self):
        reward = 0
        if self.current_points > self.last_points:
            reward = 1
        elif self.done:
            reward = -1
        return reward

    def play(self):
        """ Take input from keyboard for PAD then draw the game"""
        moved = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
                if event.key == pygame.K_LEFT:
                    self.state.change_dir('L')
                    moved = True
                elif event.key == pygame.K_RIGHT:
                    self.state.change_dir('R')
                    moved = True
                elif event.key == pygame.K_UP:
                    self.state.change_dir('U')
                    moved = True
                elif event.key == pygame.K_DOWN:
                    self.state.change_dir('D')
                    moved = True

        if not moved:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.state.change_dir('L')
            elif keys[pygame.K_RIGHT]:
                self.state.change_dir('R')
            elif keys[pygame.K_UP]:
                self.state.change_dir('U')
            elif keys[pygame.K_DOWN]:
                self.state.change_dir('D')

        return False

    def draw_state(self):
        """ Dont't take keyboard input for pad and just draw """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("end game")
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            print("end game")
            return True
        """ Draw the game """
        for row in range(0, ROWS):
            for column in range(0, COLUMNS):
                color = BACKGROUND_COLOR
                if self.state._state[row][column] == SNAKE:
                    color = SNAKE_COLOR
                elif self.state._state[row][column] == COIN:
                    color = COIN_COLOR
                pygame.draw.rect(self.screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        text_id = self.font.render(
            ("Points: %d - Life: %d" % (self.state.points, self.state._life)), False, (255, 255, 255))
        self.screen.blit(text_id, (0, 0))
        # Limit to 6 frames per second
        self.clock.tick(FPS)
        pygame.display.flip()
        return False
