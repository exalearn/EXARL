EXARL Marshmallow Environment
============

Creating custom agents
----------------------

- EXARL extends OpenAI gym's environment registration to agents
- Agents inherit from exarl.ExaAgent

Example:-

.. code-block:: python

   This environment is used to test an agents ability to explore.
    We present different types of underlying functions for the
    policy network to discover.  Each step, we are progressing
    the domain of the function by one i.e. f(t) where t is the
    current step within the episode.

    There are two types of functions (function_type):
        Polynomial
        Approximate unit function (step function)
            using to logistic functions (https://en.wikipedia.org/wiki/Logistic_function)

    Input (for functions):

        Polynomial:
            function - list of polynomial coefficents
            i.e. [1, 2, -4, .5] -> f(t) = 1*t^0 + 2*t^2 - 4*t^3 + .5*t^4

        Unit:
            L - Height of the step
            x0 - mid point of the rising edge
            x1 - mid point of the falling edge

        Pulse:
            L - List of heights of the step
            x0 - List of mid-points of the rising edge
            x1 - List of mid-points of the falling edge

    Observation space (gym.spaces.Box): 

        Polynomial: [x[0]*t^0, x[1]*t^1, ... x[n-1]*t^n-1]
            where x in the input function and t is the step

        Unit: [x0-t, x1-t] where x0 and x1 are input and 
            t is the step

        Pulse: [x0[0]-t, x0[1]-t, ... x0[n-1]-t, x1[0]-t, x1[1]-t, ... x1[n-1]-t] 
            where x0 and x1 are input and t is the step

    Action space:

        The action space can either be discrete or continueous
        in order to test different types of agents. To change
        set action_space to Box or Discrete in configuration.

        Discrete: 
            0 -> stops the episode
            1 -> Continues the episode

        Box:
            0 -> stops the episode
            0 < x >= 1 -> Scales the reward
    
    Reward:
        The reward is the value f(t) at step t

    Attributes
    ----------
    flat_fuct : list of floats
        Number used to set up observation space and store coeffiecents
    eval : function pointer
        The evaluation function used to generate the reward and
        next state for each type of function
    k : int
        Changes the slop of the logistic function used in unit function
    L : int or list of ints
        Max values of unit and pulse functions
    x0 : int or list of ints
        Mid-points of rising edges of unit and pulse functions
    x1 : int or list of ints
        Mid-points of falling edges of unit and pulse functions
    observation_space : gym.spaces.Box
        See observation space description
    action_space : gym.spaces
        See action space discription
    initial_state : gym.spaces.Box
        State to reset observation space back to
    current_step : int
        Current step within an episode
    
    Methods
    -------
    step(action)
        Takes a step by evaluating underlying function a current step index

    reset()
        Resets the observation space and step index

    poly(action_value)
        Takes the action as a single number (float or int) and
        evaulates the polynomial function (i.e. f(t) and 
        returns reward and next state

    unit(action_value)
        Takes the action as a single number (float or int) and
        evaulates the unit function (i.e. f(t) and 
        returns reward and next state

Agents must include the following functions:

.. code-block:: python

   get_weights()   # get target model weights
   set_weights()   # set target model weights
   train()         # train the agent
   update()        # update target model
   action()        # Next action based on current state
   load()          # load weights from memory
   save()          # save weights to memory
   monitor()       # monitor progress of learning

Register the agent in ``ExaRL/exarl/agents/__init__.py``

.. code-block:: python

   from .registration import register, make

   register(
      id='fooAgent-v0',
      entry_point='exarl.agents.agent_vault:FooAgent',
   )

The id variable will be passed to ``exarl.make()`` to call the agent.

The file ```ExaRL/exarl/agents/agent_vault/__init__.py``` should include

.. code-block:: python

   from agents.agent_vault.foo_agent import FooAgent

where ``ExaRL/exarl/agents/agent_vault/foo_agent.py`` is the file containing your agent
