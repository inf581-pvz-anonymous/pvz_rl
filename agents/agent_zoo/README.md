# Pre-trained agents

## dfq5_epsexp : 
DDQN, illegal actions masked, observations related to zombies are a sum of hp per lane, early action asymmetry correction, exponential epsilon for epsilon-greedy behavior
Mean score 1892.04
Mean iterations 338.413

## dfq5_epsexp_znet_1.5x : 
DDQN, illegal actions masked, observations related to zombies the result of the same linear regression of hps on each lane, early action asymmetry correction, exponential epsilon for epsilon-greedy behavior. Trained on 150000 plays instead of 100000
Mean score 1659.08
Mean iterations 311.011

## dfq5_dqn : 
DQN, illegal actions masked, observations related to zombies are a sum of hp per lane, early action asymmetry correction, exponential epsilon for epsilon-greedy behavior
Mean score 1651.36
Mean iterations 306.288
