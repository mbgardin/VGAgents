import numpy as np
import pandas as pd
from Breakout.Breakout_Class import Breakout
from Agents.Monte_Carlo_Agent import MonteCarloAgent_ES, MonteCarloAgent_FV
from Training.train_MC_Agent import train_agent, plot_rewards
import time
import pickle
import json

# method="ES"
#method="FV"
maxTimesteps= 100000
numOfEpisodes= 1000
#brick_layout="TopRow"
#brick_layout="MiddleRow"
brick_layout="ReversePyramid"
num_bricks=9

saveKeyFindings =True
saveAgent= True
plotRewards= True

compTimes = []

brick_layouts = ["TopRow","MiddleRow","ReversePyramid"]
methods= ["ES", "FV"]
num_bricksList= [5,9]
#maxTimestepsList=[100,1000,10000,30000]
#numOfEpisodesList =[100,1000,10000,100000]

maxTimestepsList=[100,1000,10000]
numOfEpisodesList =[1000]
for num_bricks in num_bricksList:
    for brick_layout in brick_layouts:
        for method in methods:
            for maxTimesteps in maxTimestepsList:
                for numOfEpisodes in numOfEpisodesList:
                    # path to files that will be saved
                    TrainInfoFilePath=brick_layout+ '_NumBricks_' + str(num_bricks) + 'Method_' + method + '_Episodes_' + str(numOfEpisodes) + '_maxTimesteps_' + str(maxTimesteps)
                    print("Working on")
                    print(TrainInfoFilePath)

                    if method== "ES":
                        agent = MonteCarloAgent_ES()
                        exploringStarts=True
                    elif method == "FV":
                        agent = MonteCarloAgent_FV()
                        exploringStarts = False
                    else: 
                        raise TypeError("Method must be either ES or FV but is {}".format(method))

                    env = Breakout(max_timesteps=maxTimesteps, brick_layout=brick_layout, num_bricks=num_bricks)
                    startTime= time.time()
                    # Let's train the agent for 1000 episodes
                    rewards, exectuionTimes = train_agent(agent=agent, env=env, episodes=numOfEpisodes, exploring_starts=exploringStarts)
                    endTime= time.time()

                    overallTrainingTime = endTime-startTime
                    avgExecutionTime = sum(exectuionTimes)/len(exectuionTimes)


                    keyFindingsDict= {
                                    'BrickLayout': brick_layout,
                                    'numOfBricks': num_bricks,
                                    'numOfEpisodes': numOfEpisodes,
                                    'maxTimesteps': maxTimesteps,
                                    'OverallTrainingTime': overallTrainingTime,
                                    'avgExcutionTimePerEps' :  avgExecutionTime
                    }


                    if saveKeyFindings:
                        jsonPath= 'CompTimesAndAvgRewards/' +TrainInfoFilePath +'.json'
                        with open(jsonPath, "w") as outfile:
                            json.dump(keyFindingsDict, outfile)
                        compTimes.append([method, brick_layout, num_bricks, numOfEpisodes, maxTimesteps, overallTrainingTime, avgExecutionTime])


                    if plotRewards:
                        print("Plotting Rewards")
                        plot_rewards(rewards, moving_avg_window=10, savePath=TrainInfoFilePath)
                        print("Plotting Rewards done")


                    if saveAgent:
                        print("saving Agent")
                        AgentPath = 'TrainedAgents/' +TrainInfoFilePath +'.pkl'
                        with open(AgentPath, 'wb') as f:
                            pickle.dump(agent, f)
                        print("saving Agent done")

                    print("done")

df = pd.DataFrame(compTimes, columns=["method", "BrickLayout", "numOfBricks", "numOfEpisodes", "maxTimesteps", "overallTrainingTime", "avgExecutionTimePerEps"])
df.to_csv("compTimes.csv")