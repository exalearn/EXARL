EXARL Workflows
===============

Creating Custom Workflows
-------------------------

- EXARL also extends OpenAI gym's environment registration to workflows
- Workflows inherit from exarl.ExaWorkflow

Example:-

.. code-block:: python

   class workflowName(exarl.ExaWorkflow):
      ...

Workflows must include the following functions:

.. code-block:: python

   run()   # run the workflow

* Register the workflow in ``ExaRL/exarl/workflows/__init__.py``

.. code-block:: python

   from .registration import register, make

   register(
      id='fooWorkflow-v0',
      entry_point='exarl.workflows.workflow_vault:FooWorkflow',
   )

The id variable will be passed to exarl.make() to call the agent.

The file ```ExaRL/exarl/workflows/workflow_vault/__init__.py``` should include:

.. code-block:: python

   from exarl.workflows.workflow_vault.foo_workflow import FooWorkflow

where ``ExaRL/exarl/workflows/workflow_vault/foo_workflow.py`` is the file containing your workflow.