EXARL Utils
===========

Debugging, Timing, and Profiling
--------------------------------
- Function decorators are provided for debugging, timing, and profiling EXARL.
- Debugger captures the function signature and return values.
- Timer prints execution time in seconds.
- Either ``line_profiler`` or ``memory_profiler`` can be used for profiling the code.
   - Profiler can be selected in ``learner_cfg.json`` or using the command line argument ``--profile``.
   - Options for profiling are ``line``, ``mem``, or ``none``.

Function decorators can be used as shown below:

.. code-block:: python

   from utils.profile import *

   @DEBUG
   def my_func(*args, **kwargs):
      ...

   @TIMER
   def my_func(*args, **kwargs):
      ...

   @PROFILE
   def my_func(*args, **kwargs):
      ...

Profiling results are written to: ``results_dir + '/Profile/<line/memory>_profile.txt``.
