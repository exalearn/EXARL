# Changes/Additions to Agents 

## K. Cosburn:
a2c_vtrace.py (N.B. best not to use a2c without vtrace)

a2c_continuous.py (A2C for continuous action spaces, currently buggy and/or sensitive to hyperparameters/network architecture)

ddpg_lstm.py (DDPG with an LSTM network for actor and critic)

ddpg_per.py (DDPG with PER, works pretty well, alternative to Uchenna's version, uses prioritized_replay.py)

ddpg_vtrace (N.B. I'm not sure about this anymore since after Uchenna's updated action function, seems very sensitive to number of steps per episode, also: the loss functions for DDPG are different from those in IMPALA so I wasn't 100% sure how to implement, this might be an issue. Also, towards the end I was looking into D4PG but didn't have time to try it out. This agent is simialar to DDPG but stores and uses full trajectories for training, which could allow for easy implementation of the n-steps v-trace, but the loss function is still different than IMPALA so I am, again, not 100% sure of its "true" implementation.)

## U. Ezeobi:
td3.py

sac.py (under constructions/still buggy)

replay_buffer.py (has three classes for Uniform, Prioritized, and Hindsight experience replay)

## S. Chenna:

## M. Moraru
