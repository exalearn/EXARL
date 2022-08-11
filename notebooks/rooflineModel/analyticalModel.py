import os
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from . import opts

class abstractTopology:
    """
    This class provides the abstraction for the network topology.

    TODO: Fill in more comments/implementation

    The evaluation class can use this class and its members to
    adjust/perform its evaluation.
    """
    def __init__(self, topology, numNodes, technology):
        self.topology = topology
        self.numNodes = numNodes
        self.technology = technology

    def getLatency(self):
        """
        TODO: Think how this should be done...
        1. We could pass in OSU micro benchmark results
        2. We could try to calculate from the technology
            - Technology could be tricky.  Getting latency
            numbers does not seem to be easy from docs.
            - Could try to do a back of the envelop calculation
            but this could have significant error

        Whatever we decide for the latency of a link, we then
        multiply by the diameter in the evaluation.
        """
        pass

    def getBandwidth(self):
        """
        TODO: More thinking...
        The problem is similar to the latency, but is a little
        better since manuals typically will tell the max bandwidth
        of a link.
        """
        pass

    def getDiameter(self):
        """
        TODO: Some directions:
        1. We can do a lookup if we assume topology is from a dictionary of set options
        2. We can do some graph algorithm if we take in a graph of the topology
            - If we go this route it should connect with Nathan's generator...
        """
        pass

class abstractApplication:
    """
    This class provides the abstraction for the application of interest.

    TODO: Fill in more comments/implementation
    
    This class will hold several configuration parameters.  There is less
    to calculate here compared to the abstractNetwork class.  Instead the
    evaluation can query this class adjusting its final result accordingly.
    """
    def __init__(self, packetSize, opsPerPacket, numPackets, injectionRate, trafficType, trafficPattern):
        self.packetSize = packetSize
        self.opsPerPacket = opsPerPacket
        self.numPackets = numPackets
        self.injectionRate = injectionRate
        self.trafficType = trafficType
        self.trafficPattern = trafficPattern

    def getOperationalIntensity(self):
        """
        TODO: While a first pass at this is trivial, we need to think about how
        this should work for phasey applications.  That is if we assume bursty traffic type.
        We also need to think how the traffic pattern effects this result.
        """
        return self.opsPerPacket / self.packetSize

class abstractHardware:
    """
    This class provides the abstraction for the compute hardware.
    For now we just assume CPU.  We can do some thinking/adjusting
    for gpus later.

    TODO: Fill in more comments/implementation

    The intended use of this class is simlar to the abstractApplication class.  Besides
    getting the operational intensity, this class can be used by the evaluation class
    by looking up its members.
    """
    def __init__(self, intPerCycle, floatPerCycle, cores, clock):
        self.intPerCycle = intPerCycle
        self.floatPerCycle = floatPerCycle
        self.cores = cores
        self.clock = clock

    def getOpRate(self, which="float"):
        """
        TODO: This should give us the highest upper bound given by the hardware.
        In reality, we want to bring this number down to something more realistic, 
        but that is all dependent on if we can approximate the performance of the
        application/kernel (without networking).  The higher this boundary, the
        less accurate the prediction will be.
        """
        pass

class rooflineEvaluation:
    """
    This class provides the function to evaluate an
    application for a given network/hardware configuration.

    TODO: Fill in more comments/implementation
    """
    def __init__(self, topology, application, hardware):
        self.topology = topology
        self.application = application
        self.hardware = hardware

    def evaluate(self, constraints=None):
        """
        TODO: Build roofline and evaluate if it is network bound or if constraints
        given, compare utilization and total runtime
        """
        self.topology.getLatency()
        self.topology.getBandwidth()
        self.hardware.getOpRate()
        self.application.getOperationalIntensity()
        self.duration()
        return False

    def duration(self):
        """
        TODO: This should return the approximate runtime based on performance.
        """
        pass