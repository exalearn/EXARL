EXARL Agents
============

Creating custom agents
----------------------

- EXARL extends OpenAI gym's environment registration to agents
- Agents inherit from exarl.ExaAgent

Example:-

.. code-block:: python

   class agentName(exarl.ExaAgent):
      ...

Agents must include the following functions:

.. code-block:: python

   get_weights()   # get target model weights
   set_weights()   # set target model weights
   train()         # train the agent
   update()        # update target model
   action()        # Next action based on current state
   load()          # load weights from pickle file
   save()          # save weights to pickle file
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
