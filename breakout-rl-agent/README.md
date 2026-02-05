# Reinforcement-Learning-Breakout
Applying Reinforcement Learning for the videogame Breakout (1976) with own agent class.

## Setup
Environment:

```bash 
conda create -n breakout python=3.10
conda activate breakout
pip install -r requirements.txt
```

If you have issues installing pygame and seaborn, install it from the conda-forge repository: 

```bash
conda install -c conda-forge pygame
conda install -c conda-forge seaborn
```

We are using the external library [ImageMagick](https://imagemagick.org/) for gif creation.