ExaBsuite Environment
======

Integration Test Environments
-------
- A wrapper for `bsuite <https://github.com/deepmind/bsuite>`_ environments. ``ExaBsuite`` inherits from ``gym.Env``.

Has the following environments:

1. Simple Bandit (Noisy Rewards, Reward Scale)

2. MNIST Contextual Bandit (Noisy Rewards, Reward Scale)

3. Catch (Noisy Rewards, Reward Scale)

4. Cartpole (Noisy Rewards, Reward Scale)

5. Mountain Car (Noisy Rewards, Reward Scale)

6. Deep Sea (Stochastic)

7. Cartpole Swingup

8. Umbrella Length 

9. Umbrella Features

10. Discounting Chain

11. Memory Length

12. Memory Bits

Example of how bsuite usually works:

.. code-block:: python

  import bsuite
  from bsuite.utils import gym_wrapper
  SAVE_PATH_RAND = '/tmp/bsuite/rand'
  raw_env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_RAND, overwrite=True)
  env = gym_wrapper.GymFromDMEnv(raw_env)

  for episode in range(raw_env.bsuite_num_episodes):
    state = env.reset()
    done = False
    while not done:
      action = Agent(state)
      state, reward, done, info = env.step(action)
