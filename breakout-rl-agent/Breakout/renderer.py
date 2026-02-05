import pygame
import time
import os

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define game constants, multiplied by 20 for visuals
# Subtracting 2 for visual separation 

CELL_SIZE = 20
BALL_RADIUS = CELL_SIZE - 10
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
BRICK_WIDTH = (CELL_SIZE * 3) - 2  
BRICK_HEIGHT = CELL_SIZE - 2

class renderer:

    def __init__(self, env):
        self.env = env
        pygame.init()
        self.grid_width = env.grid_size[0] * CELL_SIZE
        self.grid_heigth = env.grid_size[1] * CELL_SIZE
        self.screen = pygame.display.set_mode((self.grid_width, self.grid_heigth))
        self.imgCounter = 0
        self.filelist = []
        if not os.path.exists("../.tmp"):
            os.mkdir("../.tmp")

        pygame.display.set_caption("Breakout")

    def draw(self, ball_position, paddle_position, bricks):
        self.screen.fill(BLACK)

        # Draw bricks
        for brick in bricks:
            x_brick = brick[0][0] * CELL_SIZE
            y_brick = brick[0][1] * CELL_SIZE
            pygame.draw.rect(self.screen, WHITE, pygame.Rect(x_brick, y_brick, BRICK_WIDTH, BRICK_HEIGHT))

        # Draw paddle
        pygame.draw.rect(self.screen, WHITE, (paddle_position[0] * CELL_SIZE, paddle_position[1] * CELL_SIZE, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.circle(self.screen, WHITE, (ball_position[0] * CELL_SIZE, ball_position[1] * CELL_SIZE), BALL_RADIUS)
    
        pygame.display.flip()

    def render(self):
        state = self.env._get_state()
        ball_position = state[0]
        paddle_position = state[2]
        bricks = state[4]
        self.draw(ball_position, paddle_position, bricks)
        self.savePNG()
        # time.sleep(1)

    def savePNG(self):
        filename = '.tmp/' + str(self.imgCounter) + '.png'
        pygame.image.save(self.screen, filename) 
        self.filelist.append(filename)
        self.imgCounter += 1
