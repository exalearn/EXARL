# Workflows

## K. Cosburn:

Added sync and async workflows for A2C. Main difference is that the training occurs after the episode has completed in the case of both workflows. I have also gotten rid of the "yield" when collecting memories and added a function to clear the memory lists at the end of each episode.


## S. Chenna:

Changes made to the workflow vault.

1. async_learner.py : Included get_data_parallel and calc_target_f_parallel functions to accelerate the data generation pipeline of DQN agent. Set accelerate_datagen to True to enable acceleration.

2. mlasync_learner.py : Added multi-learner asynchronous implementation. Multiple learners perform distributed training on the trajectory data sent by the actors.

3. mlrma_learner_v1.py : Implemented rma_window_selector function to select the RMA window selection policy. Default policy is random where learners randomly selects an actor's RMA window. However, this results in multiple actors picking the same RMA window which may be detrimental to policy performance. Added a bin-based approach in which each learners is allocated the range of actor windows. Set random variable to False in rma_window_selector function to enable this behaviour.

## M. Moraru:
### non_blocking_async_learner_v2.py
Optimized version of the async workflow :
- The actor use non blocking MPI.isend() for sending batch data to learner.
- The actor get updated weights from an RMA window.
- The learner does not change (comparing to the async workflow).

### seed_learner.py
Workflow based on SEED architecture :
- The actor send a single observation to the learner (containing the current_state) and wait fo the next action to take.
- The learner recieve the observation --> remember(observation) --> generate training batch data --> inference action using the new policy --> send the action to the actor.

### seed_a2c_learner.py
Workflow based on SEED architecture compliant with the A2C agent.

### rma_queue_pop_all_learner.py
The main goal of this workflow is to offload the communication part:
- Before each training part the learner request a new batch of data from a queue.
- While data is transferred, the learner can perform a training step.

### mlrma_queue_learner.py
Workflow based on the RMA queue data structure (see diagrams/rma_queue.png and diagrams/rma_queue_2.png):
- Each group of actors is assigned to a specific learner.
- Learners that exhaust all ‘active’ actors assist other learners in fetching batch data.
- Use a “shared bitmap array” which indicates which actors are active. 
- Does not stop until there is no more active actors and all the queues are empty.

### mlrma_queue_short_learner.py
Similar to "mlrma_queue_learner", but the learners does not assist the others in fetching batch data. The program ends when there is at least one learner which can not get data from his group of actors (all actors from the group are done and their queues are empty). In this case the learner does not try to get data from other groups.
