# Workflows

## K. Cosburn:

Added sync and async workflows for A2C. Main difference is that the training occurs after the episode has completed in the case of both workflows. I have also gotten rid of the "yield" when collecting memories and added a function to clear the memory lists at the end of each episode.


## S. Chenna:

Changes made to the workflow vault.

1. async_learner.py : Included get_data_parallel and calc_target_f_parallel functions to accelerate the data generation pipeline of DQN agent. Set accelerate_datagen to True to enable acceleration.

2. mlasync_learner.py : Added multi-learner asynchronous implementation. Multiple learners perform distributed training on the trajectory data sent by the actors.

3. mlrma_learner_v1.py : Implemented rma_window_selector function to select the RMA window selection policy. Default policy is random where learners randomly selects an actor's RMA window. However, this results in multiple actors picking the same RMA window which may be detrimental to policy performance. Added a bin-based approach in which each learners is allocated the range of actor windows. Set random variable to False in rma_window_selector function to enable this behaviour.
