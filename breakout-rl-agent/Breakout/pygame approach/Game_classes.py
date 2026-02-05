import pygame
from pygame.locals import *
import random

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define game constants
#multiplied by 20 for visuals
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 200
BALL_RADIUS = 20
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
BRICK_WIDTH = 60
BRICK_HEIGHT = 20

class Ball:
    def __init__(self):
        self.radius = BALL_RADIUS
        self.pos = [random.randint(self.radius, SCREEN_WIDTH - self.radius), SCREEN_HEIGHT // 2]
        self.vel = [random.randint(2, 4), random.randint(2, 4)]

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

    def reverse_velocity_x(self):
        self.vel[0] = -self.vel[0]

    def reverse_velocity_y(self):
        self.vel[1] = -self.vel[1]

class Paddle:
    def __init__(self):
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.pos = [(SCREEN_WIDTH - self.width) // 2, SCREEN_HEIGHT - self.height * 2]
        self.vel = 0

    def move_left(self):
        self.vel = -5

    def move_right(self):
        self.vel = 5

    def stop(self):
        self.vel = 0

    def update(self):
        self.pos[0] += self.vel
        if self.pos[0] < 0:
            self.pos[0] = 0
        elif self.pos[0] > SCREEN_WIDTH - self.width:
            self.pos[0] = SCREEN_WIDTH - self.width

    def get_direction(self, ball_pos):
        paddle_center = self.pos[0] + self.width / 2
        hit_position = (ball_pos[0] - self.pos[0]) / self.width
        #directions are not multiplied by 20
        if hit_position < 0.2:
            return -2, 1  # Move the ball to the left
        elif hit_position < 0.4:
            return -1, 1  # Move the ball slightly to the left
        elif hit_position < 0.6:
            return 0, 1  # Move the ball straight
        elif hit_position < 0.8:
            return 1, 1  # Move the ball slightly to the right
        else:
            return 2, 1  # Move the ball to the right

class Brick:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)

class BreakoutGame:
    def __init__(self):
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Breakout")

        # Set up the game clock
        self.clock = pygame.time.Clock()

        self.ball = Ball()
        self.paddle = Paddle()
        self.bricks = []
        for row in range(2):
            for col in range(SCREEN_WIDTH // BRICK_WIDTH):
                brick_x = col * BRICK_WIDTH
                brick_y = row * BRICK_HEIGHT
                brick = Brick(brick_x, brick_y)
                self.bricks.append(brick)

        self.game_over = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.game_over = True
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    self.paddle.move_left()
                elif event.key == K_RIGHT:
                    self.paddle.move_right()
            elif event.type == KEYUP:
                if event.key == K_LEFT or event.key == K_RIGHT:
                    self.paddle.stop()

    def update(self):
        self.paddle.update()
        self.ball.update()

        # Check for collisions
        if self.ball.pos[0] < self.ball.radius or self.ball.pos[0] > SCREEN_WIDTH - self.ball.radius:
            self.ball.reverse_velocity_x()
        if self.ball.pos[1] < self.ball.radius:
            self.ball.reverse_velocity_y()

        if self.ball.pos[1] > SCREEN_HEIGHT - self.ball.radius - self.paddle.height and self.paddle.pos[0] <= self.ball.pos[0] <= self.paddle.pos[0] + self.paddle.width:
            direction_x, direction_y = self.paddle.get_direction(self.ball.pos)
            self.ball.vel[0] = direction_x
            self.ball.vel[1] = direction_y
            self.ball.reverse_velocity_y()

        for brick in self.bricks:
            if brick.rect.collidepoint(self.ball.pos):
                self.bricks.remove(brick)
                self.ball.reverse_velocity_y()
                break

    def draw(self):
        self.screen.fill(BLACK)

        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, WHITE, brick.rect)

        # Draw paddle
        pygame.draw.rect(self.screen, WHITE, (self.paddle.pos[0], self.paddle.pos[1], self.paddle.width, self.paddle.height))

        # Draw ball
        pygame.draw.circle(self.screen, WHITE, (self.ball.pos[0], self.ball.pos[1]), self.ball.radius)

        pygame.display.flip()

    def run(self):
        while not self.game_over:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


