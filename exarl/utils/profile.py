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
import os
import functools
import time
from exarl.utils.globals import ExaGlobals

class ProfileConstants:
    """
    Singleton class to deal with loading results directory from candle parameters.
    The appropriate class is loaded the first time the initializer is called.

    Attributes
    ----------
    initialized : bool
        Indicates if the singleton has been initialized
    started : bool
        Flag indicating if profiler has already been started
    profile_type : string
        This comes from the config/candle driver. Choices are
        line, mem, and intro.
    results_dir : string
        Dir where to write results
    file : string
        File where to write results
    profile : function
        Function pointer of profiler
    ib : ib
        Introspector class
    """
    initialized = False
    started = False
    profile_type = None
    results_dir = None
    file = None
    profile = None

    ib = None
    ib_loaded = lambda: False

    def __init__(self):
        if not ProfileConstants.initialized:
            ProfileConstants.profile_type = ExaGlobals.lookup_params('profile')
            ProfileConstants.results_dir = os.path.join(ExaGlobals.lookup_params('output_dir'), 'Profile')
            if not os.path.exists(ProfileConstants.results_dir):
                os.makedirs(ProfileConstants.results_dir, exist_ok=True)

            if ProfileConstants.profile_type == 'mem':
                import memory_profiler
                ProfileConstants.profile = memory_profiler.profile
                ProfileConstants.file = os.path.join(ProfileConstants.results_dir, 'mem_profile.txt')

            elif ProfileConstants.profile_type == 'line':
                import line_profiler
                ProfileConstants.profile = line_profiler.LineProfiler()
                ProfileConstants.file = os.path.join(ProfileConstants.results_dir, 'line_profile.txt')

                def write_profile_to_file():
                    with open(ProfileConstants.file, 'w') as file:
                        ProfileConstants.profile.print_stats(stream=file)
                atexit.register(write_profile_to_file)

            elif ProfileConstants.profile_type == 'intro':
                import exarl.utils.introspect
                from exarl.base.comm_base import ExaComm
                ProfileConstants.ib = exarl.utils.introspect.ibLoadReplacement(ExaComm.global_comm)
                ProfileConstants.ib_loaded = exarl.utils.introspect.ibLoaded
                atexit.register(lambda: exarl.utils.introspect.ibWrite(ProfileConstants.results_dir))
            ProfileConstants.initialized = True

    @staticmethod
    def introspected():
        """
        Returns if introspector is loaded and ran.
        """
        if ProfileConstants.started:
            return ProfileConstants.ib_loaded()
        return False

def PROFILE(func):
    """
    Invokes line_profiler, memory_profiler, and introspector.
    Based on https://realpython.com/primer-on-python-decorators/

    Parameters
    ----------
    func : function
        function to be profiled

    Returns
    -------
    function
        wrapper profile function
    """
    @functools.wraps(func)
    def wrapper_profile(*args, **kwargs):
        ProfileConstants()
        if ProfileConstants.profile_type is not None:
            if not ProfileConstants.started:
                ProfileConstants.started = True
                if ProfileConstants.profile_type == 'line':
                    new_func = ProfileConstants.profile(func)
                    return new_func(*args, **kwargs)

                elif ProfileConstants.profile_type == 'mem':
                    stream = open(ProfileConstants.file, 'w')
                    new_func = ProfileConstants.profile(func, stream=stream)
                    return new_func(*args, **kwargs)

                elif ProfileConstants.profile_type == 'intro':
                    ProfileConstants.ib.start()
                    ret = func(*args, **kwargs)
                    ProfileConstants.ib.stop()
                    return ret

        return func(*args, **kwargs)

    return wrapper_profile

def DEBUG(func):
    """
    Print the function signature and return value

    Parameters
    ----------
    func : function
        function to be wrapped for debuggin

    Returns
    -------
    function
        debug wrapper
    """
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

def TIMER(func):
    """
    Print the runtime of the decorated function

    Parameters
    ----------
    func : function
        function to be wrapped for printing runtime

    Returns
    -------
    function
       timer wrapper function that returns the function return value
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def TIMERET(func):
    """
    Print the runtime of the decorated function

    Parameters
    ----------
    func : function
        function to be wrapped for printing runtime

    Returns
    -------
    function
       timer wrapper function that returns the runtime of the function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return run_time
    return wrapper_timer
