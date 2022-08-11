import functools
from time import time_ns as globalTimeStamp

# Try to import introbind and replace if fail
try:
    raise
    import introbind as ib

    def ibLoaded():
        return True

    def ibLoadReplacement(comm):
        return ib

    def ibWrite(writeDir):
        pass

except:
    class ib:
        """This class enables tracing of an MPI rank's function execution \\
            by using Python decorators to wrap a function. It is used to replace an external introbind library when not found.
        Attributes
        ----------
        replace : bool
            True if we are wrapping this function for tracing
        init : bool
            True if the class has been initialized.
        rank : int
            MPI rank of process being traced
        skew : list
            Global time stamps
        start_time : int
            Time stamp when tracing started, time in nanoseconds since the epoch.
        end_time : int
            Time stamp when tracing ended, time in nanoseconds since the epoch.
        metric_window : dictionary
            Dictionary with keys as metric names being traced and values as tuple of time window when it was traced.
        metric_list : dictionary
            Dictionary with keys as metric names being traced and values of the history of time windows of traces.
        metric_trace : dictionary
            Dictionary with keys as metric names being traced and values as tuples of counts and timestamps.
        metric_trace_count : dictionary
            Dictionary with keys as metric names being traced and values as number of traces for that name.
        last_trace : str
            Name of metric that was last traced.


        """
        replace = False
        init = False
        rank = None
        skew = []
        start_time = None
        end_time = None
        metric_window = {}
        metric_list = {}
        metric_trace = {}
        metric_trace_count = {}
        last_trace = None

        def __init__(self, comm):
            """Initializer which checks if it has not been initialized, then it does that.

            Parameters
            ----------
            comm : mpi4py.MPI.Comm
                MPI communication object associated with this trace.
            """
            if ib.replace and not ib.init:
                ib.init = True
                ib.rank = comm.rank

                # For skew files
                self.skew.append(globalTimeStamp())
                comm.barrier()
                self.skew.append(globalTimeStamp())
                comm.barrier()
                self.skew.append(globalTimeStamp())
                comm.barrier()

        @staticmethod
        def start():
            """Starts tracing this rank.

            Returns
            -------
            integer
                0 if we are not tracing, 1 if tracing.
            """
            if ib.replace:
                print("---------------STARTING REPLACEMENT IB", ib.rank, "---------------", flush=True)
                ib.start_time = globalTimeStamp()
                return 1
            return 0

        @staticmethod
        def stop():
            """Stops tracing for this rank.
            """
            if ib.replace:
                print("---------------STOPPING REPLACEMENT IB", ib.rank, "---------------", flush=True)
                ib.end_time = globalTimeStamp()

        @staticmethod
        def update(name, toAdd):
            """Updates tracing metrics for the given name.

            Parameters
            ----------
            name : str
                name of function being timed
            toAdd : int
                amount being added to metric

            Returns
            -------
            int
                1 if metric is updated, -1 otherwise
            """
            if ib.replace:
                if ib.start_time is not None and ib.end_time is None:
                    if name not in ib.metric_window:
                        ib.metric_window[name] = (0, ib.start_time, 0)
                        ib.metric_list[name] = []

                    old = ib.metric_window[name]
                    new = ib.metric_window[name] = (old[0] + toAdd, globalTimeStamp(), max([old[0] + toAdd, old[0]]))
                    ib.metric_list[name].append((old[0], old[1], new[0], new[1], new[2]))
                    return 1
            return -1

        @staticmethod
        def startTrace(name, size):
            """Begin a trace of a metric for a function.

            Parameters
            ----------
            name : str
                name of metric
            size : int


            Returns
            -------
            int
                1 if trace started, 0 otherwise
            """
            if ib.replace:
                if ib.start_time is not None and ib.end_time is None:
                    if ib.last_trace is None:
                        if name not in ib.metric_trace:
                            ib.metric_trace[name] = []
                            ib.metric_trace_count[name] = 1

                        ib.metric_trace[name].append((size, ib.metric_trace_count[name], globalTimeStamp()))
                        ib.metric_trace_count[name] += 1
                        ib.last_trace = name
                        return 1
            return 0

        @staticmethod
        def simpleTrace(name, size, seqNum, endTimeStamp, trace):
            """Create a trace associated with a function

            Parameters
            ----------
            name : str
                name of function to trace
            size : int

            seqNum : int

            endTimeStamp : float

            trace : int


            Returns
            -------
            int
                1 if data appended to the trace, 0 otherwise
            """
            if ib.replace:
                if ib.start_time is not None and ib.end_time is None:
                    if name not in ib.metric_trace:
                        ib.metric_trace[name] = []
                    ib.metric_trace[name].append((size, seqNum, globalTimeStamp(), endTimeStamp, trace))
                    return 1
            return 0

        @staticmethod
        def stopTrace():
            """Stops the trace if it was started.
            """
            if ib.replace:
                if ib.start_time is not None and ib.end_time is None:
                    if ib.last_trace is not None:
                        size, seqNum, startTimeStamp = ib.metric_trace[ib.last_trace].pop()
                        ib.metric_trace[ib.last_trace].append((size, seqNum, startTimeStamp, globalTimeStamp()))
                        ib.last_trace = None

    def ibWrite(writeDir):
        """Write out the trace information to files.

        Parameters
        ----------
        writeDir : str
            Directory within which trace files beginning with "nodeMetric_, trace_ and skew_" are written
        """
        if writeDir is not None and ib.replace and ib.init:
            for name in ib.metric_list:
                filename = writeDir + "/nodeMetric_" + name.replace('_', '') + "_" + str(ib.rank) + ".ct"
                with open(filename, "w") as writeFile:
                    for data in ib.metric_list[name]:
                        toWrite = '0,1,' + ','.join([str(d) for d in data])
                        writeFile.write(toWrite + "\n")

                for name in ib.metric_trace:
                    filename = writeDir + "/trace_" + name.replace('_', '') + "_" + str(ib.rank) + ".ct"
                    with open(filename, "w") as writeFile:
                        for data in ib.metric_trace[name]:
                            toWrite = '0,' + str(ib.rank) + ',0,0,' + ','.join([str(d) for d in data])
                            if len(data) == 4:
                                toWrite = toWrite + ","
                            writeFile.write(toWrite + "\n")

                filename = writeDir + "/skew_" + str(ib.rank) + ".ct"
                with open(filename, "w") as writeFile:
                    for i in ib.skew:
                        writeFile.write(str(i) + "\n")

    def ibLoaded():
        """Checks if we are introspecting

        Returns
        -------
        bool
            True if we are introspecting, False otherwise
        """
        return ib.replace

    def ibLoadReplacement(comm):
        """Start tracing.

        Parameters
        ----------
        comm : MPI communicator
            [description]

        Returns
        -------
        introspection object
            introspection object initialized for the given communicator
        """

        ib.replace = True
        return ib(comm)

def introspectTrace(position=None, keyword=None, default=0, name=False):
    """Defines a decorator used to trace functions.  Arguments are used to predifine values to trace.
    E.g. A decorated call that should always increment trace by 1 should set the default=1
    A call that should trace a value that is passed to the decorated function as the 2 positional argument should set position=1
    A call that needs to trace a value that is passed as the size keywork to the underlying argument should set keyword='size'

    Parameters
    ----------
    position : int, optional
        size is set to the value in function args[position], by default None
    keyword : int, optional
        size is set to this keyword value, by default None
    default : int, optional
        size is set to this value, by default 0
    name : bool, optional
        prepend an additional name to the trace name, by default False
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            traceName = func.__name__
            if name:
                traceName = args[0].name + "_" + func.__name__
            size = default
            if position:
                size = args[position]
            elif keyword:
                size = kwargs.get(keyword, default)
            flag = ib.startTrace(traceName, size)
            result = func(*args, **kwargs)
            if flag:
                ib.stopTrace()
            return result

        return wrapper

    return decorator

def introspect(func):
    """A decorator that wraps a function with so that when called it updates a function metric with 1

    Parameters
    ----------
    func : function
        function to be wrapped

    Returns
    -------
    function
        the wrapper for the func argument
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        ib.update(func.__name__, 1)
        return result

    return wrapper
