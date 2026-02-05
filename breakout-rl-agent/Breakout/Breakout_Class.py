import numpy as np
import random
from Breakout.renderer import renderer
import time

class Breakout:
    def __init__(self, grid_size=(15, 10), num_bricks=5, max_timesteps=10000, rendering=False, brick_layout="TopRow"):
        self.grid_size = grid_size
        self.num_bricks = num_bricks
        self.paddle_size = 5
        self.max_timesteps=max_timesteps
        self.timesteps= 0
        self.rendering = rendering
        self.brick_layout=brick_layout
        if rendering: 
            self.renderer = renderer(self)
        self.reset()

    def reset(self):
        self.paddle_position = [self.grid_size[0] // 2, self.grid_size[1] - 1] # place paddle in the center
        self.ball_position = [self.paddle_position[0] + self.paddle_size // 2, self.paddle_position[1] - 1] # place ball on top of the paddle
        self.ball_direction = [np.random.choice([-2, -1, 0, 1, 2]), -1] # initial direction of the ball
        self.paddle_speed = 0
        self.bricks = self._generate_bricks() 
        self.done = False
        self.timesteps=0
        return self._get_state()

#for exploring starts while training
    def random_reset(self):
        self.paddle_position = [np.random.choice(range(0,self.grid_size[0]-self.paddle_size)), self.grid_size[1] - 1] # place paddle at random possible position 
        self.ball_position = [np.random.choice(range(self.grid_size[0])), np.random.choice(range(self.grid_size[1]))] # place ball in random place in grid
        self.ball_direction = [np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-1,1])] # initial direction of the ball
        self.paddle_speed = np.random.choice([-2,-1,0,1,2])
        self.done=False

        # using this method a high and low number of bricks is equally likely 
        # higher chance of few brick leads than if we give 50/50 chance for each brick to exist 
        self.bricks = self._generate_bricks() 
        numOfBricks= random.randrange(len(self.bricks)) + 1


        while len(self.bricks) > numOfBricks:
            self.bricks.pop(random.randrange(len(self.bricks)))
 
        self.done = False
        self.timesteps=0
        return self._get_state()

    def ingame_reset(self):
        self.paddle_position = [self.grid_size[0] // 2, self.grid_size[1] - 1]# place paddle in the center
        self.ball_position = [self.paddle_position[0] + self.paddle_size // 2, self.paddle_position[1] - 1] # place ball on top of the paddle
        self.ball_direction = [np.random.choice([-2, -1, 0, 1, 2]), -1] # initial direction of the ball
        self.paddle_speed = 0
        self.bricks = self._generate_bricks() 
        self.done = False
        return self._get_state()

    def _generate_bricks(self):
        bricks = []
        # for i in range(0, self.grid_size[0], 3):  # Step size of 3 to make space for each brick
        #     brick = []
        #     for j in range(3):  # For each block of the brick
        #         if i + j < self.grid_size[0]:  # To prevent bricks going out of the grid
        #             brick.append([i + j, 0])  # Arrange the blocks of the brick horizontally
        #     bricks.append(brick)
        # return bricks

        if self.brick_layout=="TopRow":
            for k in range(0, self.num_bricks * 3 // self.grid_size[0] + 1): # Adding bricks for each row, step size of 3
                for i in range(0, self.grid_size[0], 3):  # Step size of 3 to make space for each brick
                    brick = []
                    for j in range(3):  # For each block of the brick
                        if i + j < self.grid_size[0] and len(bricks) < self.num_bricks: 
                                brick.append([i + j, k])  # Arrange the blocks of the brick horizontally in row k 
                    if len(bricks) < self.num_bricks:
                        bricks.append(brick)
            return bricks

        if self.brick_layout=="MiddleRow":
            for k in range(0, self.num_bricks * 3 // self.grid_size[0] + 1): # Adding bricks for each row, step size of 3
                for i in range(0, self.grid_size[0], 3):  # Step size of 3 to make space for each brick
                    brick = []
                    for j in range(3):  # For each block of the brick
                        if i + j < self.grid_size[0] and len(bricks) < self.num_bricks: 
                                brick.append([i + j, k+1])  # Arrange the blocks of the brick horizontally in row k 
                    if len(bricks) < self.num_bricks:
                        bricks.append(brick)
            return bricks

        if self.brick_layout == "ReversePyramid":
            margin = 0  # Margin from sides
            for k in range(0, self.num_bricks * 3 // self.grid_size[0] + 2):  # Adding bricks for each row
                for i in range(3*margin, self.grid_size[0] - (3*margin), 3):  # Step size of 3 to make space for each brick
                    brick = []
                    for j in range(3):  # For each block of the brick
                        if i + j < self.grid_size[0] - (3*margin) and len(bricks) < self.num_bricks: 
                            brick.append([i + j, k])  # Arrange the blocks of the brick horizontally in row k 
                    if len(brick) == 3 and len(bricks) < self.num_bricks:
                        bricks.append(brick)
                if len(bricks) >= self.num_bricks:  # stop creating bricks once we reach the desired number
                    break
                margin += 1  # Increase margin for next row
            return bricks




    def _get_state(self):
        stateTuple = (tuple(self.ball_position), tuple(self.ball_direction), tuple(self.paddle_position), self.paddle_speed, tuple(tuple(tuple(brick) for brick in bricks) for bricks in self.bricks))
        return stateTuple
    
    # public function needed for testing
    def get_state_public(self):
        stateTuple= self._get_state()
        return stateTuple

    def step(self, action):
        # Initialize reward
        reward = -1

        action = np.clip(action, -1, 1)
    
        # Update paddle speed, ensure the speed is within the range of [-2, 2]
        self.paddle_speed = np.clip(self.paddle_speed + action, -2, 2)
        
        # Update paddle position based on the speed
        self.paddle_position[0] += self.paddle_speed

        # Ensure the paddle stays within the grid
        self.paddle_position[0] = np.clip(self.paddle_position[0], 0, self.grid_size[0]-self.paddle_size)


        # Update ball position if within the boundaries
        if 0 <= self.ball_position[0] + self.ball_direction[0] <= self.grid_size[0]:
            self.ball_position[0] += self.ball_direction[0]
        else:
            self.ball_position[0] += self.ball_direction[0]
            self.ball_direction[0] *= -1 # Ball gets reflected in x-axis
        # elif case can ot happen due to being included in if (0<=x includes case 0==x)
        if 0 < self.ball_position[1] + self.ball_direction[1] <= self.grid_size[1]:
            self.ball_position[1] += self.ball_direction[1]
        elif self.ball_position[1] + self.ball_direction[1] == 0: # Ball hits the upper boundary
            self.ball_position[1] += self.ball_direction[1]
            self.ball_direction[1] *= -1 # Ball gets reflected in y-axis
        elif 0 > self.ball_position[1] + self.ball_direction[1]: # ball somehow outside screen
            self.ball_position[1] = 0
            self.ball_direction[1] = 1
            


        # Check if ball hits brick
        hit = False
        for brick in self.bricks:
            if self.ball_position in brick:
                self.bricks.remove(brick)
                self.ball_direction[1] *= -1 # Ball gets reflected
                #reward += 50 # reward for hitting a brick
                hit = True

        # Check if ball hits paddle
        if self.paddle_position[0] <= self.ball_position[0] < (self.paddle_position[0] + self.paddle_size) and self.ball_position[1] == self.paddle_position[1]:
            self.ball_direction[1] *= -1

            # Determine the part of the paddle the ball hit
            paddle_part = (self.ball_position[0] - self.paddle_position[0]) // (self.paddle_size // 5)
            paddle_parts = [-2, -1, 0, 1, 2]
            
            # Change horizontal direction based on where the ball hits the paddle
            self.ball_direction[0] = paddle_parts[paddle_part]


        # Check if ball goes past the paddle
        if self.ball_position[1] > self.paddle_position[1]:
            #reward += -200 # punishment for missing the ball
            self.ingame_reset()
        # Check if all bricks are destroyed
        if len(self.bricks) == 0:
            # reward = 0
            self.done = True # End of episode
            #reward += 1000 # reward for winning the game

        # count timesteps in this run
        self.timesteps += 1
        if self.timesteps >= self.max_timesteps: 
            self.done=True


        return self._get_state(), reward, self.done

    def render(self): 
        if self.rendering:
            self.renderer.render()

        