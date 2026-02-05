from Breakout.Breakout_Class import Breakout
import numpy as np
import time

#seems to work
'''
game = Breakout()
print(game.reset()) # Should print the initial state

for _ in range(100):
    state, reward, done = game.step(1) # Try moving the paddle to the right
    print(reward, done) 
'''

num_games = 1
game = Breakout(rendering=True)

for i in range(num_games):
    game.reset()
    done = False
    print("############################")
    while not done: # loop until the game is done
        choices = np.array([-1, 0, 1])
        action = np.random.choice(choices)
        state, reward, done = game.step(action)
        game.render()
        time.sleep(0.02)

        