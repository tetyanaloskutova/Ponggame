# PONG emulator
import math
import random
import pygame, sys
import logging
import numpy as np
import shutil

from pygame.locals import *
from analysis import nn_function
from analysis import nonlin

pygame.init()
fps = pygame.time.Clock()

# colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# globals
BALL_RADIUS = 12
PAD_WIDTH = 60
PAD_HEIGHT = 8
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
WIDTH = PAD_WIDTH * 8
HEIGHT = WIDTH + PAD_HEIGHT

ball_pos = [0, 0]
paddle_vel = HALF_PAD_WIDTH
score = 0
wall_bounce = 0
win_lose_game = 0

# speed is important because the next location is calculated from the previous.
# For more accuracy, it would be required to calculate the initial vector and move along it.
speed = BALL_RADIUS
angle = 0
angle_radians = 0
logger = logging.getLogger('pong')
# False means training, True - playing
play = False

# what is going to be logged
last_x = 0
syn0 = [[]]

# canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Learning to pong!')


# Spawn a ball from the central position
# angle range 280:350 and 190:270
def ball_init():
    global ball_pos, speed, angle, angle_radians
    ball_pos = [WIDTH // 2, HEIGHT // 2 ]
    angle = random.randrange(280, 350)
    # to direct the ball in different direction
    dir = random.randint(0,1)
    if (dir == 0):
        angle -= 90

    angle_radians = math.radians(angle)

# define event handlers
def init():
    global logger, paddle_pos, paddle_vel, speed, angle, angle_radians, score

    logger = logging.getLogger('pong')
    hdlr = logging.FileHandler('pong.log', mode='w')
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)
    logger.error("win_lose,wall_bounce,score,paddle_pos,x,y")

    paddle_pos = [HALF_PAD_WIDTH, HEIGHT - HALF_PAD_HEIGHT]
    score = 0
    ball_init()

def outside_boundaries():
    global ball_pos
    if (int(ball_pos[0]) <= BALL_RADIUS or int(ball_pos[1]) <= BALL_RADIUS or int(ball_pos[0]) >= WIDTH - BALL_RADIUS):
        return True
    return False



# draw function of canvas
def draw(canvas):
    global syn0, last_x, logger, win_lose_game, wall_bounce, paddle_pos, paddle_vel, ball_pos, speed, angle, angle_radians, score

    # the playing field marking
    canvas.fill(BLUE)
    pygame.draw.line(canvas, WHITE, [0, HEIGHT // 2], [WIDTH, HEIGHT //2], 1)
    pygame.draw.line(canvas, WHITE, [0, HEIGHT - PAD_HEIGHT], [WIDTH, HEIGHT - PAD_HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 30, 1)


    if paddle_pos[0] <= HALF_PAD_WIDTH and paddle_vel < 0:
        paddle_vel = HALF_PAD_WIDTH
    elif paddle_pos[0] >= WIDTH - HALF_PAD_WIDTH and paddle_vel > 0:
        paddle_vel = -HALF_PAD_WIDTH

    paddle_pos[0] += paddle_vel


    # draw paddle and ball
    pygame.draw.circle(canvas, RED, ball_pos, BALL_RADIUS, 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle_pos[0] - HALF_PAD_WIDTH, paddle_pos[1] - HALF_PAD_HEIGHT],
                                        [paddle_pos[0] - HALF_PAD_WIDTH, paddle_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle_pos[0] + HALF_PAD_WIDTH, paddle_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle_pos[0] + HALF_PAD_WIDTH, paddle_pos[1] - HALF_PAD_HEIGHT]], 0)

    # ball collision check on left, top walls, and right wall
    const_angle = 90
    wall_bounce = 0
    if (int(ball_pos[0]) <= BALL_RADIUS or int(ball_pos[1]) <= BALL_RADIUS or int(ball_pos[0]) >= WIDTH + 3 - BALL_RADIUS):
        wall_bounce = 1
        if (ball_pos[1] <= BALL_RADIUS):
            last_x = ball_pos[0]
        if (angle > 180 and angle < 270 and int(ball_pos[1]) <= BALL_RADIUS):
            const_angle = -90
        if (angle > 270 and angle < 360 and int(ball_pos[0]) >= WIDTH - BALL_RADIUS):
            const_angle = -90
        if (angle > 90 and angle < 180 and (int(ball_pos[0]) >= WIDTH - BALL_RADIUS or int(ball_pos[0]) <= BALL_RADIUS)):
            const_angle = -90
        if (angle > 0 and angle < 90 and int(ball_pos[0]) >= WIDTH - BALL_RADIUS):
            print(str(angle))

        # if we've already trained and the ball is falling down, move the paddle to meet it
        if (play):
            test_arr = np.array([[last_x, ball_pos[0]]])
            test_arr = test_arr / 500
            test_res = 500 * nonlin(np.dot(test_arr, syn0))
            paddle_vel = 0
            paddle_pos[0] = int(test_res)
            print(test_res)

        # for +-straight angles and hits in the corner +-45, 135, 225, 315, just start over
        # because small deviations from these angles cause the ball to "spin" when calculating step vectors,
        # that is, these vectors end outside the boundaries and cause a trajectory incompatible with 90 deg angle rule
        if (angle == 278):
            angle = 90
            ball_init()
        elif (angle == 360) or (angle == 0):
            angle = 180
            ball_init()
        elif (angle == 90):
            angle = 270
            ball_init()
        elif (angle == 180):
            angle = 0
            ball_init()
        elif (angle == 45) or (angle == 135 ) or (angle == 225) or (angle == 315):
            angle += 180
            ball_init()
        else:
            angle = (angle + const_angle)
        if (angle >= 360):
            angle = angle - 360
        if (angle < 0):
            angle = 360 - angle

        # log when bounces from the wall
        logger.error(str(win_lose_game) + ", " + str(wall_bounce) + ", " + str(score) + ", " + str(
            paddle_pos[0]) + ", " + str(ball_pos[0]) + ", " + str(ball_pos[1]))
        angle_radians = math.radians(angle)
        # extra move to let the ball jump out of the conditional cycle
        count = 0
        while outside_boundaries():
            ball_pos[0] += int(speed * math.cos(angle_radians))
            ball_pos[1] += int(speed * math.sin(angle_radians))
            count +=1
            if count > 10:
                ball_init()

    ball_pos[0] += int(speed * math.cos(angle_radians))
    ball_pos[1] += int(speed * math.sin(angle_radians))

    const_angle = 90
    # ball collision check on paddle
    if int(ball_pos[1]) >= (HEIGHT + 1 - BALL_RADIUS - PAD_HEIGHT)  and int(ball_pos[0]) in range(paddle_pos[0] - HALF_PAD_WIDTH,
                                                                             paddle_pos[0] + HALF_PAD_WIDTH, 1):
        logger.error(str(win_lose_game) + ", " + str(wall_bounce) + ", " + str(score) + ", " + str(
            paddle_pos[0]) + ", " + str(ball_pos[0]) + ", " + str(ball_pos[1]))
        paddle_vel = HALF_PAD_WIDTH
        win_lose_game +=1
        const_angle = 90
        if (angle > 0 and angle < 90):
            const_angle = -90
        # for straight angles and hits in the corner +-45, 135, 225, 315, just bounce back
        if (angle > 80 and angle <100):
            angle = 270
        elif (angle > 40 and angle < 50) or (angle > 130 and angle < 140) or (angle > 220 and angle < 230) or (
            angle > 310 and angle < 320):
            angle += 180
        else:
            angle = (angle + const_angle)
        if (angle >= 360):
            angle = angle - 360
        if (angle < 0):
            angle = 360 + angle

        angle_radians = math.radians(angle)
        # extra move to let the ball jump out of the conditional cycle
        ball_pos[0] += int(speed * math.cos(angle_radians))
        ball_pos[1] += int(speed * math.sin(angle_radians))
        ball_pos[0] += int(speed * math.cos(angle_radians))
        ball_pos[1] += int(speed * math.sin(angle_radians))

        ball_pos[0] += int(speed * math.cos(angle_radians))
        ball_pos[1] += int(speed * math.sin(angle_radians))
        score += 1
        if (score > 10000):
            score = 0
            ball_init()
    elif int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS - PAD_HEIGHT:
        logger.error(str(win_lose_game) + ", " + str(wall_bounce) + ", " + str(score) + ", " + str(
            paddle_pos[0]) + ", " + str(ball_pos[0]) + ", " + str(ball_pos[1]))

        win_lose_game+=1
        ball_init()

    # update scores
    myfont = pygame.font.SysFont("Arial", 20)
    label = myfont.render("Score: " + str(score), 1, WHITE)
    canvas.blit(label, (WIDTH-100, HEIGHT-30))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "play":
        play = True
        shutil.copy2('pong.log', 'analysis.csv')
        syn0 = nn_function()
        if syn0 is None:
            print("Please run the program with 'train' argument ")
            exit(1)

init()


# game loop
while True:
    draw(window)

    for event in pygame.event.get():

        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fps.tick(100)
