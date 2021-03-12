import atexit
import exarl.utils.candleDriver as cd
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
