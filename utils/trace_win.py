import sys
import os
import functools
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


class Trace_Win:
    _instances = {}
    _ON = False

    def __new__(cls, name=None, *args, **kwargs):
        if name not in Trace_Win._instances:
            return super(Trace_Win, cls).__new__(cls)
        return Trace_Win._instances[name]

    def __init__(self, arrayType=np.int64, name=None, comm=None):
        if Trace_Win._ON and name not in Trace_Win._instances:
            Trace_Win._instances[name] = self

            self.comm = comm.raw()

            self.name = name
            self.arrayType = arrayType

            self.trace = None
            self.counters = None

            if comm.rank == 0:
                self.trace = []
                self.counters = np.zeros(comm.size, dtype=arrayType)

            if arrayType == np.int64:
                self.winCounter = MPI.Win.Create(
                    self.counters, disp_unit=MPI.INT64_T.Get_size(), comm=self.comm
                )
            else:
                self.winCounter = MPI.Win.Create(
                    self.counters, disp_unit=MPI.DOUBLE.Get_size(), comm=self.comm
                )

            self.winCounter.Fence()

    def update(self, value=None):
        if Trace_Win._ON:
            if value is not None:
                if self.arrayType == np.int64:
                    data = np.array([value], dtype=np.int64)
                else:
                    data = np.array([value], dtype=np.float64)

                self.winCounter.Lock(0)
                self.winCounter.Accumulate(
                    data, 0, target=[self.comm.rank, 1], op=MPI.REPLACE
                )
                self.winCounter.Unlock(0)
            else:
                data = np.ones(1, dtype=np.int64)
                self.winCounter.Lock(0)
                self.winCounter.Accumulate(
                    data, 0, target=[self.comm.rank, 1], op=MPI.SUM
                )
                self.winCounter.Unlock(0)

    def snapshot(self):
        if Trace_Win._ON:
            if self.comm.rank == 0:
                if self.arrayType == np.int64:
                    counts = np.zeros(self.comm.size, dtype=np.int64)
                else:
                    counts = np.zeros(self.comm.size, dtype=np.float64)

                self.winCounter.Lock(0)
                self.winCounter.Flush_all()
                self.winCounter.Get_accumulate(counts, counts, 0, op=MPI.NO_OP)
                self.winCounter.Unlock(0)

                self.trace.append(counts)
                return counts

    def write(log_dir):
        if Trace_Win._ON:
            for name in Trace_Win._instances:
                tw = Trace_Win._instances[name]
                tw.winCounter.Fence()
                if tw.comm.rank == 0:
                    tw.snapshot()
                    with open(log_dir + "/" + str(name) + ".txt", "w") as f:
                        for count in tw.trace:
                            line = ",".join([str(x) for x in count]) + "\n"
                            f.write(line)
                    print("Wrote", name)

    def plot(log_dir, hist=True, bins=100):
        if Trace_Win._ON:
            for name in Trace_Win._instances:
                tw = Trace_Win._instances[name]
                tw.winCounter.Fence()
                if tw.comm.rank == 0:
                    data = []
                    for j in range(tw.comm.size):
                        temp = []
                        for l in tw.trace:
                            temp.append(l[j])
                        data.append(temp)

                    for i in range(tw.comm.size):
                        if i > 0:
                            if hist:
                                plt.hist(
                                    data[i],
                                    bins=len(data[0]),
                                    alpha=0.5,
                                    label="Node " + str(i),
                                )
                            else:
                                plt.plot(range(len(data[i])), data[i], label=i)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.xlabel("Step")
                    plt.ylabel("Occurance")
                    print(log_dir + "/" + str(name) + ".pdf")
                    plt.savefig(log_dir + "/" + str(name) + ".pdf", bbox_inches="tight")
                    plt.close()

    def plotSteps(log_dir, name):
        if Trace_Win._ON:
            tw = Trace_Win._instances[name]
            tw.winCounter.Fence()
            tw.winCounter.Fence()
            if tw.comm.rank == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                last = tw.trace[-1]
                data = []
                for j in range(tw.comm.size):
                    temp = []
                    for l in tw.trace:
                        temp.append(l[j])
                    data.append(temp)

                for i in range(tw.comm.size):
                    if i > 0:
                        ax1.hist(
                            data[i], bins=last[i], alpha=0.5, label="Node " + str(i)
                        )
                        temp = np.histogram(data[i], bins=last[i])
                        ax2.hist(
                            temp[0],
                            density=True,
                            histtype="step",
                            cumulative=True,
                            label="Node " + str(i),
                        )
                ax1.legend(loc="upper left")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Number of times a step was used")
                ax2.legend(loc="upper left")
                ax2.set_xlabel("Number of times a step was used")
                ax2.set_ylabel("CDF")
                plt.savefig(log_dir + "/" + str(name) + ".pdf", bbox_inches="tight")

    def plotModel(log_dir, name):
        if Trace_Win._ON:
            tw = Trace_Win._instances[name]
            tw.winCounter.Fence()
            tw.winCounter.Fence()
            if tw.comm.rank == 0:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                last = tw.trace[-1]
                data = []
                for j in range(tw.comm.size):
                    temp = []
                    for l in tw.trace:
                        temp.append(l[j])
                    data.append(temp)

                totalDelay = []
                for i in range(tw.comm.size):
                    if i > 0:
                        delay = [x - y for x, y in enumerate(data[i])]
                        ax1.step(range(len(data[i])), data[i], label="Node " + str(i))
                        ax1.step(
                            range(len(data[i])),
                            delay,
                            label="Node " + str(i) + " Delay",
                        )
                        ax2.hist(delay, alpha=0.5, label="Node " + str(i))
                        temp = np.histogram(delay, bins=last[i])
                        ax3.hist(
                            temp[0],
                            density=True,
                            histtype="step",
                            cumulative=True,
                            label="Node " + str(i),
                        )
                ax1.legend()
                ax1.set_xlabel("Train")
                ax1.set_ylabel("Model Version")
                ax2.legend()
                ax2.set_xlabel("Size of Delay")
                ax2.set_ylabel("Count of the Size of Delays")
                ax3.set_xlabel("Size of Delay")
                ax3.set_ylabel("CDF")
                ax3.legend()
                plt.savefig(log_dir + "/" + str(name) + ".pdf", bbox_inches="tight")


def Trace_Win_Up(name, comm, arrayType=np.int64, position=None, keyword=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace = Trace_Win(arrayType=arrayType, name=name, comm=comm)
            value = None
            if position is not None:
                value = args[position]
            elif keyword is not None:
                value = kwargs.get(keyword, default)
            trace.update(value=value)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def Trace_Win_Snap(name, comm, arrayType=np.int64):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace = Trace_Win(arrayType=arrayType, name=name, comm=comm)
            result = func(*args, **kwargs)
            trace.snapshot()
            return result

        return wrapper

    return decorator
