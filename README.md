# pvz_rl
Plants vs. Zombies remake and RL agents

# TODO
## Requirements

The following python libraries are required: pytorch, gym, pygame (and shap to evaluate feature importance if wanted).

## Installation/Setup

The game engine we developped and the Open AI gym environment are both encapsulated in python libraries.
The easier way to run our code is to use the command prompt (at least in windows or linux).

```
git clone https://github.com/inf581-pvz-anonymous/pvz_rl.git
cd pvz_rl
cd pvz
pip install -e .
cd ..
cd gym-pvz
pip install -e .
```

Everything related to the game balance and FPS configuration is in the pvz library we developped. In particular, config.py, entities/zombies.WaveZombieSpawner.py and the main files containing balance data. For plant and zombie characteristics, check the files in the entities/plants/ and entities/zombies/ folders.

## Usage example

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

### Evaluate the agent
To evaluate your newly trained agent, you'll have to modify the script_evaluate.py file in the following way. Change the agent_type variable in accordance to the trained agent (e.g "DDQN") and then replace the loaded path by the chosen "NAME" in the corresponding if clause in the main function.

```
python script_evaluate.py
```

### Visualize a play

### Feature importance
