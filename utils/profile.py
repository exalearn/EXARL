# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import atexit
import utils.candleDriver as cd
import os
import functools
import time
from mpi4py import MPI

global_comm = MPI.COMM_WORLD

prof = cd.run_params['profile']
results_dir = cd.run_params['output_dir'] + '/'
if not os.path.exists(results_dir + '/Profile'):
    if (global_comm.rank == 0):
        os.makedirs(results_dir + '/Profile')


def PROFILE(func):
    """Invokes line_profiler and memory_profiler"""
    # Line profiler
    if prof == 'line':
        import line_profiler
        profile = line_profiler.LineProfiler()

        @functools.wraps(func)
        def wrapper_profile(*args, **kwargs):
            new_func = profile(func)
            return new_func(*args, **kwargs)

        # Write line profiler output to file
        def write_profile_to_file():
            if prof == 'line':
                with open(results_dir + '/Profile/line_profile.txt', 'w') as file:
                    profile.print_stats(stream=file)
        atexit.register(write_profile_to_file)

    # Memory profiler
    elif prof == 'mem':
        from memory_profiler import profile
        file = open(results_dir + '/Profile/mem_profile.txt', 'w')

        @functools.wraps(func)
        def wrapper_profile(*args, **kwargs):
            new_func = profile(func, stream=file)
            return new_func(*args, **kwargs)

    # No profiler
    else:
        @functools.wraps(func)
        def wrapper_profile(*args, **kwargs):
            return func(*args, **kwargs)

    return wrapper_profile

# Based on https://realpython.com/primer-on-python-decorators/


def DEBUG(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug

# Based on https://realpython.com/primer-on-python-decorators/


def TIMER(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
