import numpy as np
from collections import defaultdict
import random



a= (1,2)

a[1]=np.clip(a ,10,11)
print(a)

brick_layouts = ["TopRow","MiddleRow","ReversePyramid"]
methods= ["ES","FV"]
maxTimestepsList=[100,1000,10000,30000]
numOfEpisodesList =[100,1000,10000,100000]

for brick_layout in brick_layouts:
    for methods in methods:
        for maxTimesteps in maxTimestepsList:
            for numEpisodes in numOfEpisodesList:
                print("")