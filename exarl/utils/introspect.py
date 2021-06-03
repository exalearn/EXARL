import functools
from time import time_ns as globalTimeStamp

# Try to import introbind and replace if fail
# try:
#     import introbind as ib
#     def ibLoaded():
#         return True

#     def ibLoadReplacement(comm, writeDir):
#         pass

# def ibWrite():
#     pass

# except:
class ib:
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


    def start():
        if ib.replace:
            print("---------------STARTING REPLACEMENT IB---------------")
            ib.start_time = globalTimeStamp()
            return 1
        return 0

    def stop():
        if ib.replace:
            print("---------------STOPPING REPLACEMENT IB---------------")
            ib.end_time = globalTimeStamp()

    def update(name, toAdd):
        if ib.replace:
            if ib.start_time is not None and ib.end_time is None:
                if name not in ib.metric_window:
                    ib.metric_window[name] = (0, ib.start_time, 0)
                    ib.metric_list[name] = []

                old = ib.metric_window[name]
                new = ib.metric_window[name] = (old[0]+toAdd, globalTimeStamp(), max([old[0]+toAdd, old[0]]))
                ib.metric_list[name].append((old[0], old[1], new[0], new[1], new[2]))
                return 1
        return -1

    def startTrace(name, size):
        if ib.replace:
            if ib.start_time is not None and ib.end_time is None:
                if name not in ib.metric_trace:
                    ib.metric_trace[name] = []
                    ib.metric_trace_count[name] = 1
                    
                ib.metric_trace[name].append((size, ib.metric_trace_count[name], globalTimeStamp()))
                ib.metric_trace_count[name]+=1
                ib.last_trace = name
                return 1
        return 0

    def simpleTrace(name, size, seqNum, endTimeStamp, trace):
        if ib.replace:
            if ib.start_time is not None and ib.end_time is None:
                if name not in ib.metric_trace:
                    ib.metric_trace[name] = []
                ib.metric_trace[name].append((size, seqNum, globalTimeStamp(), endTimeStamp, trace))
                return 1
        return 0

    def stopTrace():
        if ib.replace:
            if ib.start_time is not None and ib.end_time is None:
                if ib.last_trace is not None:
                    size, seqNum, startTimeStamp = ib.metric_trace[ib.last_trace].pop()
                    ib.metric_trace[ib.last_trace].append((size, seqNum, startTimeStamp, globalTimeStamp()))
                    ib.last_trace = None
        
def ibWrite(writeDir):
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
    return ib.replace

def ibLoadReplacement(comm):
    ib.replace = True
    return ib(comm)

def introspectTrace(position=None, keyword=None, default=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            size = default
            if position:
                size = args[position]
            elif keyword:
                size = kwargs.get(keyword, default)
            flag = ib.startTrace(func.__name__, size)
            result = func(*args, **kwargs)
            if flag:
                ib.stopTrace()
            return result

        return wrapper

    return decorator


def introspect(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        ib.update(func.__name__, 1)
        return result

    return wrapper

