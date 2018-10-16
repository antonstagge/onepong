import pygame
import numpy as np

DRAWING_AREA = 500
RADIUS = 12

def draw(screen, grow, inputs, hidden, outputs):
    if grow:
        screen_size = screen.get_size()
        screen = pygame.display.set_mode([screen_size[0] + DRAWING_AREA, screen_size[1]])
    # normalize
    # new_inputs = [0.0,0.0,0.0,0.0,0.0]
    # new_inputs[0] = inputs[0]/50
    # new_inputs[1] = inputs[1]/60
    # new_inputs[2] = abs(inputs[2])/2
    # new_inputs[3] = abs(inputs[3])/2
    # new_inputs[4] = inputs[4]/50
    # inputs = new_inputs

    #inputs = sigmoid(inputs)
    inputs = inputs/np.linalg.norm(inputs)
    outputs = softmax(outputs)

    in_positions = []
    hidden_positions = []
    out_positions = []
    screen_size = screen.get_size()
    y_pos = int(screen_size[1]/4)
    draw_nodes(screen, y_pos, inputs, in_positions)
    y_pos += y_pos
    draw_nodes(screen, y_pos, hidden, hidden_positions)
    y_pos += int(y_pos/2)
    draw_nodes(screen, y_pos, outputs, out_positions)

    for start in in_positions:
        for end in hidden_positions:
            draw_line(screen, start, end)

    for start in hidden_positions:
        for end in out_positions:
            draw_line(screen, start, end)

def draw_nodes(screen, y_pos, nodes, positions):
    x_pos = screen.get_size()[0] - DRAWING_AREA + first_x_pos(len(nodes))
    for x in nodes:
        color = 255*(x)
        if color < 0:
            color = 0
        elif color > 255:
            color = 255
        color = int(color)
        pygame.draw.circle(screen, (255,255,255), (x_pos, y_pos), RADIUS, 1)
        pygame.draw.circle(screen, (0,color,0), (x_pos, y_pos), RADIUS-1)
        positions.append((x_pos, y_pos))
        x_pos += 3*RADIUS

def draw_line(screen, start, end):
    direction = [0,0]
    direction[0] = start[0] - end[0]
    direction[1] = start[1] - end[1]
    direction = direction/np.linalg.norm(direction)
    direction[0] = direction[0]*RADIUS
    direction[1] = direction[1]*RADIUS
    new_start = [0,0]
    new_start[0] = start[0]-direction[0]
    new_start[1] = start[1]-direction[1]
    new_end = [0,0]
    new_end[0] = end[0]+direction[0]
    new_end[1] = end[1]+direction[1]
    pygame.draw.line(screen, (255,255,255), new_start, new_end)


def first_x_pos(number_of_nodes):
    return int((DRAWING_AREA - number_of_nodes*RADIUS*2 - (number_of_nodes-1)*RADIUS)/2) + RADIUS

def sigmoid(outputs):
    return 1.0/(1.0+np.exp(-outputs))

def softmax(outputs):
    normalisers = np.sum(np.exp(outputs))*np.ones((1,np.shape(outputs)[0]))
    return np.transpose(np.transpose(np.exp(outputs))/normalisers)