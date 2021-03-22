# pvz_rl
Plants vs. Zombies remake and RL agents

## Requirements

The following python libraries are required: pytorch, gym, pygame (and shap to evaluate feature importance if wanted).

## Installation/Setup

The game engine we developed and the Open AI gym environment are both encapsulated in python libraries.
The easiest way to run our code is to use the command prompt (at least in windows or linux).

```
git clone https://github.com/inf581-pvz-anonymous/pvz_rl.git
cd pvz_rl
cd pvz
pip install -e .
cd ..
cd gym-pvz
pip install -e .
cd ..
```

Everything related to the game balance and FPS configuration is in the pvz library we developed. In particular, config.py, entities/zombies.WaveZombieSpawner.py and the main files containing balance data. For plant and zombie characteristics, check the files in the entities/plants/ and entities/zombies/ folders.

You can test your installation by running the following lines and watch our trained DDQN in action.
```
python game_render.py
```

## Usage example

Here we will guide you to train an agent from scratch. However, since the training can be fairly long depending on your machine (more than 30 minutes if the agent performs well), you can skip the training section (you then do not have to modify any of the code which will then use a pretrained agent from our agent zoo).

### Train an agent

To train a DDQN agent

```
python train_ddqn_agent.py
```

To train other agents, use the dedicated train scripts. You will be asked to save the agent under a certain name/path we will refer as NAME in the following commands.
Once the agent is trained, for DDQN agent, you can plot the learning curves with

```
python plot_training_ddqn.py NAME
```
In red is the performances of the agent evaluated by removing the epislon-greedy policy used for training and using a greedy policy instead (i.e. epsilon=0).

### Evaluate the agent
To evaluate your newly trained agent, you'll have to modify the script_evaluate.py file in the following way. Change the agent_type variable in accordance to the trained agent (e.g "DDQN") and then replace the loaded path by the chosen "NAME" in the corresponding if clause in the main function.

```
python script_evaluate.py
```
1000 plays will be run with your agent and you will then see in order: histogram distribution of the score, histogram distribution of the survived frames, histogram distribution of the actions used, histogram distribution of the plant used.


### Visualize a play
With the following, you can visualize a game of an agent.
```
python game_render.py
```
By default, this will show the behavior of the DDQN agent. You can modify agent_type in game_render.py to use some other saved agents or even your own agent (doing the exact same modifications as above)

To visualize a game with higher FPS (more fluid motions), change the FPS variable in pvz/pvz/config.py. This may have a slight impact on the behavior of some agents.

### Feature importance
You can compute SHAP values if you used a DDQN agent by setting the loaded path to the chosen "NAME" in script_feature_importance.py and then run:
```
python script_feature_importance.py
```
