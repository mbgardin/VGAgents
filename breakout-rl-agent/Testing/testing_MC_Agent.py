import numpy as np
import matplotlib.pyplot as plt
import time
import visualizer as visualizer

# function for calculating the average reward over number of episodes
def test_agent_average_Reward(agent, env, episodes):
    rewards = []
    timestep = 0
    executionTime =[]
    print("start Testing")

    startTime = time.time()
    for episode in range(episodes):
        total_reward = 0
        reward=0
        state= ()
        next_state=()
        done = False

        # Reset environment and get initial state
        state = env.reset()

        while not done:
            # Choose action based on the agent's policy
            action = agent.act(state)
            # Execute the action in the environment
            next_state, reward, done = env.step(action)
            # Record the transition
            # have to take the bricks list and make it an immutable item
            total_reward += reward
            state = next_state
            timestep += 1 
            
        # Store the reward
        rewards.append(total_reward)


        if (episode+1) % 100 == 0 or episode== 0:
            print(f'Episode {episode+1}/{episodes}: Reward {total_reward}')

        if (episode+1)== episodes:
            print("Testing done")

        #save time needed per episode
        endTime = time.time()
        executionTime.append(endTime-startTime)
        startTime= time.time()
    
    return rewards, executionTime


# test the agent given a certain starting State:
def test_agent_certain_start(agent, env, render=False, startingDirection= 0):
    rewards = []
    timestep = 0
    print(f'start testing for start {startingDirection}')



    # Record the transitions
    transitions = []
    total_reward = 0
    reward=0
    state= ()
    next_state=()
    done = False

    # Reset environment and get initial state

    state = env.reset()
    env.ball_direction = [startingDirection, -1] # initial direction of the ball
    state = env.get_state_public()

    while not done:
        # Choose action based on the agent's policy
        action = agent.act(state)
        if render:
            env.render()
        # Execute the action in the environment
        next_state, reward, done = env.step(action)
        # Record the transition
        transitions.append((state, action, reward))
        total_reward += reward
        state = next_state
        timestep += 1 
        
    # Store the reward
    rewards.append(total_reward)

    if render:
        env.render()


    return transitions

