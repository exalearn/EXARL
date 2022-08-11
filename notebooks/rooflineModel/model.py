import os
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from . import opts

def clusterData(df, numClusters):
    """
    Takes a dataframe and clusters data in Time column.
    A column cluster is added with the resulting cluster for
    each entry.
    
    Parameters
    ----------
    dir : str
        Directory to get files from
    numClusters : int
        Number of clusters

    Returns
    -------
    pd.DataFrame
        Dataframe of all data contained in dir
    """
    data = df["Time"].to_frame()
    data["cluster"] = -1 
    model = KMeans(n_clusters=numClusters)
    model.fit(data)
    clusterPerElement = model.predict(data)
    df = df.assign(cluster=clusterPerElement)
    return df

def getClusters(data, percent):
    """
    This function clusters data into N clusters.  To figure
    out the number of clusters, we begin at two and continue
    increasing N by 1 until the max and min of each cluster
    is less than a threshold.  The threshold is given by:
    percent * (max - min).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to cluster.  Must have a Time column.
    percent : float
        Number from 0 to 1

    Returns
    -------
    new
        Dataframe of clustered data with newly added cluster column
    numClusters
        The number of clusters 
    """
    minTime = data["Time"].min()
    maxTime = data["Time"].max()
    diff = maxTime - minTime
    percentDiff = diff * percent
    attempts = len(data)
    for numClusters in range(2, attempts):
        new = clusterData(data, numClusters)
        clusters = new["cluster"].unique().tolist()
        flags = []
        for cluster in clusters:
            check = new[new["cluster"] == cluster]
            flags.append(percentDiff >= check["Time"].max() - check["Time"].min())
        if all(flags):
            break
    return new, numClusters
            
def plotCluster(data, yaxis):
    """
    Plots clustered data. X-axis is the source node.
    Y-axis is the bandwidth/latency.  Color indicates
    cluster.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to cluster.  Must have a Time column.
    percent : float
        Number from 0 to 1

    Returns
    -------
    new
        Dataframe of clustered data with newly added cluster column
    numClusters
        The number of clusters 
    """
    index = data["src"].unique().tolist()
    clusters = data["cluster"].unique().tolist()
    for cluster in clusters:
        toPlot = data[data["cluster"] == cluster]
        x = [index.index(x) for x in toPlot["src"]]
        y = toPlot["Time"]
        plt.scatter(x, y)
    plt.xlabel("Node Index")
    plt.ylabel(yaxis)
    plt.show()

def getMaxCluster(data, numClusters, minMax=max):
    """
    Returns the cluster with the most entries.
    
    Parameters
    ----------
    data : pd.DataFrame
        Clustered data. Must have a cluster column.
    numClusters : int
        Number of clusters

    Returns
    -------
    maxCluster
        The max size
    average
        The average of all metric data in the cluster
    """
    allSizes = {i:len(data[data["cluster"] == i]) for i in range(numClusters)}
    maxCluster = max(allSizes, key=allSizes.get)
    return maxCluster, data[data["cluster"] == maxCluster]["Time"].mean()

def getBoundingCluster(data, numClusters, minMax=max):
    """
    
    Parameters
    ----------
    data : pd.DataFrame
        Clustered data. Must have a cluster column.
    numClusters : int
        Number of clusters

    Returns
    -------
    maxCluster
        The max size
    average
        The average of all metric data in the cluster
    """
    allSizes = {i:data[data["cluster"] == i]["Time"].mean() for i in range(numClusters)}
    maxCluster = minMax(allSizes, key=allSizes.get)
    return maxCluster, data[data["cluster"] == maxCluster]["Time"].mean()

def getClusterPerSize(dir, percent, metric="bw", rank=2, plot=False, which=getMaxCluster, minMax=max, **kwargs):
    """
    Clusters data per message size.
    
    Parameters
    ----------
    dir : str
        Director with data 
    percent : float
        Percent difference to set the threshold for max cluster size
    metric : str, optional
        Which metric to cluster
    plot : bool, optional
        Flag to plot data
    which : function, optional
        which clustering function to run

    Returns
    -------
    ret
        Dictionary of message sizes and average metric
    """
    ret = {}
    data = opts.getOSUDataframe(dir, **kwargs)
    data = data[data["metric"] == metric]
    sizes = data["Size"].unique().tolist()
    for size in sizes:
        temp = data[data["Size"] == size]
        temp = temp[temp["ranks"] == rank]
        cluster, numClusters = getClusters(temp, percent)
        index, ave = which(cluster, numClusters, minMax=minMax)
        if plot:
            print("Rank", rank, "Size", size, "NumClusters", numClusters, "Index", index, "Ave", ave)
            plotCluster(cluster)
        ret[size] = ave
    return ret

#%%

def plotNetworkCurves(dir, **kwargs):
    """
    Plots results from the OSU bandwidth and latency curves
    side-by-side
    
    Parameters
    ----------
    dir : str
        Director with data 
    """
    data = opts.getOSUDataframe(dir, **kwargs)
    df = data[data["metric"] == "lat"]
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    sns.violinplot(ax=axes[0], data=df, x="Size", y="Time")
    axes[0].set_ylabel("Latency (us)")
    axes[0].set_ylim(bottom = 0)
    axes[0].set_xlabel("Message Size (Bytes)")
    axes[0].set_title("OSU MPI Latency Test v5.8")

    df = data[data["metric"] == "bw"]
    sns.violinplot(ax=axes[1], data=df, x="Size", y="Time")
    axes[1].set_ylabel("Bandwidth (MB/s)")
    axes[1].set_ylim(bottom = 0)
    axes[1].set_xlabel("Message Size (Bytes)")
    axes[1].set_title("OSU MPI Bandwidth Test v5.8")
    plt.show()

#%%
@opts.csvWriter(prefix="messageBandwidth")
def getMessageBandwidths(resultsDir, percent=.2):
    """
    Returns a dataframe of the bandwidths based on OSU bandwidth and latency tests.
    We return three sets of numbers:
        Max: This is the average of the cluster with the highest value
        Ave: This is the average of the largest cluster (number of entries)
        Min: This is the average of the cluster with the lowest value
    All results are presented in bytes/second

    Parameters
    ----------
    resultsDir : str
        Path with OSU output
    percent : float
        Value between 0 and 1 indicating the max difference in cluster size.

    Returns
    -------
    Dataframe
        Datafram of bandwidths

    """
    data = opts.getOSUDataframe(resultsDir)
    ranks = data["ranks"].unique()
    ranks.sort()
    df = None
    for rank in ranks:

        bandwidths = getClusterPerSize(resultsDir, percent, rank=rank)
        bandwidthMax = getClusterPerSize(resultsDir, percent, rank=rank, which=getBoundingCluster, minMax=max)
        bandwidthMin = getClusterPerSize(resultsDir, percent, rank=rank, which=getBoundingCluster, minMax=min)
        
        latencies = getClusterPerSize(resultsDir, percent, metric="lat", rank=rank)
        latencyMax = getClusterPerSize(resultsDir, percent, metric="lat", rank=rank, which=getBoundingCluster, minMax=max)
        latencyMin = getClusterPerSize(resultsDir, percent, metric="lat", rank=rank, which=getBoundingCluster, minMax=min)

        bandwidthsBW = {X : Y*1024*1024 for X, Y in zip(bandwidths.keys(), bandwidths.values())}
        bandwidthMaxBW = {X : Y*1024*1024 for X, Y in zip(bandwidthMax.keys(), bandwidthMax.values())}
        bandwidthMinBW = {X : Y*1024*1024 for X, Y in zip(bandwidthMin.keys(), bandwidthMin.values())}

        latenciesBW = {X : (X*1E6)/Y for X, Y in zip(latencies.keys(), latencies.values())}
        latencyMaxBW = {X : (X*1E6)/Y for X, Y in zip(latencyMax.keys(), latencyMax.values())}
        latencyMinBW = {X : (X*1E6)/Y for X, Y in zip(latencyMin.keys(), latencyMin.values())}

        bandwidthDF = pd.DataFrame([bandwidthMaxBW, bandwidthsBW, bandwidthMinBW, latencyMaxBW, latenciesBW, latencyMinBW])
        bandwidthDF = bandwidthDF.drop(columns=0)
        bandwidthDF = bandwidthDF.transpose()
        bandwidthDF.columns = ["Max Bandwidth", "Ave Bandwidth", "Min Bandwidth", "Max Latency", "Ave Latency", "Min Latency"]
        bandwidthDF["ranks"] = rank
        if df is None:
            df = bandwidthDF
        else:
            df = pd.concat([df,bandwidthDF])
    return df

#%%