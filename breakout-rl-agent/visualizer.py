import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

imgpath = "./img/"
sns.set_style("whitegrid")


def rewards_time(rewards_per_timestep):
    data = pd.DataFrame(rewards_per_timestep, columns=['step', 'reward', 'episode'])
    sns.scatterplot(x='step', y='reward', data=data, hue='episode', alpha=0.6)
    plt.savefig(imgpath + 'rewards_time.png', dpi=300)


def saveGIF(name): 
    filelist = []
    for file in os.listdir('.tmp'):
        filelist.append(file.replace(".png", ""))
    filelist = [int(file) for file in filelist]
    filelist.sort()
    filelist = ['.tmp/' + str(file) + ".png" for file in filelist]

    files = " ".join(filelist)

    # Using external library ImageMagick for automatic gif creation
    os.system(f'magick convert -size 600x400 -delay 5 -loop 0 {files} img/{name}.gif')
    # remove files

    for filename in os.listdir('.tmp'):   
        os.unlink(f'.tmp/{filename}')
    # os.rmdir('.tmp')