import numpy as np
import matplotlib.pyplot as plt
import time
import visualizer as visualizer

def train_agent(agent, env, episodes, exploring_starts=True):
    rewards = []
    timestep = 0
    rewards_per_timestep = []
    executionTime =[]
    print("start Training")

    startTime = time.time()
    for episode in range(episodes):
        # Record the transitions
        transitions = []
        total_reward = 0
        reward=0
        state= ()
        next_state=()
        done = False

        # Reset environment and get initial state

        if (exploring_starts):
            # Exploring starts with random start state and random first action
            state = env.random_reset()
            # random action in timestep 0
            action = np.random.choice(agent.action_space)
            next_state, reward, done = env.step(action)
            # Record the transition
            transitions.append((state, action, reward))
            total_reward += reward
            state = next_state
            rewards_per_timestep.append([timestep, total_reward, episode]) # for plotting 
            timestep += 1 
        else:
            state = env.reset()


        while not done:
            # Choose action based on the agent's policy
            action = agent.act(state)

            # Execute the action in the environment
            next_state, reward, done = env.step(action)
            # Record the transition

            # have to take the bricks list and make it an immutable item

            transitions.append((state, action, reward))
            total_reward += reward
            state = next_state

            rewards_per_timestep.append([timestep, total_reward, episode]) # for plotting 
            timestep += 1 
            

        # After each episode, update the agent's Q values
        agent.update_Q(transitions, total_reward)

        # Store the reward
        rewards.append(total_reward)

        # Reduce the epsilon (Exploration vs. Exploitation trade-off)
        agent.update_epsilon(episode)

        if (episode+1) % 100 == 0 or episode== 0:
            print(f'Episode {episode+1}/{episodes}: Reward {total_reward}')

        if (episode+1)== episodes:
            print("Training done")

        #save time needed per episode
        endTime = time.time()
        executionTime.append(endTime-startTime)
        startTime= time.time()
    

    return rewards, executionTime


def plot_rewards(rewards, moving_avg_window=10,savePath=None, showPlot=False):
    #close old plots
    plt.close('all')
    # Plot raw rewards
    plt.plot(rewards, label='Raw rewards')

    # Also plot the moving average of rewards
    moving_avg_rewards = np.convolve(rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
    plt.plot(range(moving_avg_window-1, len(rewards)), moving_avg_rewards, label=f'Moving average ({moving_avg_window} episodes)')

    # Add labels and legend
    plt.title('Rewards per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Set y-axis limit
    plt.ylim([-2000, 30])
    if savePath is not None:
        #save plot at savePath
        path= 'Plots/Rewards_'+savePath+ '.png'
        plt.savefig(path)

    if showPlot:
        print("close plot to proceed")
        plt.show()


    