#%%
import sys
import os
import re
import functools
import pandas as pd
#%%

def csvWriter(prefix=None):
    """
    A decorator to read data/write data to a directory as csv
    instead of parsing it every time.
    To use add the kwarg:
        csvDir - the directory to read/write csv
        force - bool to force parsing data and write new csv

    Parameters
    ----------
    func : function
        function to be wrapped
    Returns
    -------
    function
        the wrapper for the func argument
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dir = args[0]
            force = False
            localPrefix = None
            if "csvDir" in kwargs:
                csvDir = kwargs.pop("csvDir")
                if "force" in kwargs:
                    force = kwargs.pop("force")
                if "prefix" in kwargs:
                    localPrefix = kwargs.pop("prefix")
            else:
                return func(*args, **kwargs)

            ret = None
            csvName = None
            if csvDir:
                tempDir = dir.replace("/", "_")
                tempDir = tempDir.replace(".", "_")
                if localPrefix is not None:
                    tempDir = localPrefix + "_" + tempDir
                if prefix is not None:
                    tempDir = prefix + "_" + tempDir
                csvName = csvDir + "/" + tempDir + ".csv"
                if csvDir is not None and not force:
                    try:
                        ret = pd.read_csv(csvName, index_col=0)
                    except:
                        force = True

            if ret is None or force:
                ret = func(*args, **kwargs)
                if csvName is not None:
                    print("Writing", csvName)
                    ret.to_csv(csvName)
            else:
                print("Reading", csvName)
            return ret

        return wrapper
    return decorator

def getFiles(dir, filter=None, singleLevel=False):
    """
    This grabs all the files in a directory.  The filter is used
    to select files that match a sub-string.  If no filters is
    supplied, all files in the directory will be return.

    Parameters
    ----------
    dir : str
        Directory to get files from
    filter : str, optional
        Sub-strings to look for in filename

    Returns
    -------
    list
        sorted list of filenames
    """
    ret = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if filter is None or filter in f:
                file = os.path.join(root, f)
                ret.append(file)
        if singleLevel:
            break
    ret.sort()
    return ret

def readCartFile(filename):
    """
    Reads a file output from the stream benchmark and parses it into a list.
    
    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    List
        Data from the stream benchmark to be put into a dataframe
    """
    lastPart = filename.split("/")[-1]
    base = lastPart[:-4]
    parts = base.split("_")
    which = parts[0]
    episode_block = parts[1]
    batch_frequency = int(parts[2])
    ranks = int(parts[3])
    if len(parts) >= 5:
        train_frequency = int(parts[4])
    else:
        train_frequency = 1
    if len(parts) >= 6:
        batch_size = int(parts[5])
    else:
        batch_size = 32

    time = None
    altTime = 0

    with open(filename) as f:
        converged=-1
        alive=-1
        last=-1
        average=None
        lines = f.readlines()
        for line in lines:
            if "Maximum elapsed time" in line:
                temp = re.split(' ', line.rstrip())
                time = float(temp[-1])
            elif "Converged:" in line:
                temp = re.split(' ', line.rstrip())
                converged = int(temp[1])
                alive = int(temp[3])
                average = float(temp[5])
                last = float(temp[7])
            elif "Time = " in line:
                temp = re.split(' ', line.rstrip())
                altTime = max(altTime, float(temp[-1]))
            elif "Total Reward:" in line:
                if average is None:
                    temp = re.split(' ', line.rstrip())
                    average = float(temp[-1])
        if average is None:
            average = -1

    if time is None:
        time = altTime
    return (which, episode_block, batch_frequency, ranks, train_frequency, batch_size, time, converged, alive, average, last)

@csvWriter()
def getCartDataframe(dir):
    """
    Gets a dataframe from directory of stream output files.
    All files in the directory should have .txt as the file extension.
    
    Parameters
    ----------
    dir : str
        Path with stream output

    Returns
    -------
    Dataframe
        Datafram from a directory of stream output.
    """
    files = getFiles(dir, ".txt", True)
    temp = []
    for f in files:
        try:
            temp.append(readCartFile(f))
        except:
            print("Failed:", f)
    ret = pd.DataFrame(temp, columns=("which", "episode_block", "batch_step_frequency", "ranks", "train_frequency", "batch_size", "time", "converged", "alive", "average", "last"))
    
    truthy = {"1":True, "true":True, "True":True, "0":False, "false":False, "False":False}
    ret["episode_block"] = ret["episode_block"].map(truthy)
    
    return ret

def getFailedCart(dir):
    files = getFiles(dir, ".txt", True)
    ret = []
    for f in files:
        try:
            readCartFile(f)
        except Exception as e:
            print(e)
            ret.append(f)
    return ret

def readTraceFile(filename):
    """
    Reads a file output from the stream benchmark and parses it into a list.
    
    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    List
        Data from the stream benchmark to be put into a dataframe
    """
    
    dirs = filename.split("/")
    base = dirs[dirs.index("EXP000")-1]
    parts = base.split("_")
    which = parts[0]
    episode_block = parts[1]
    batch_frequency = int(parts[2])
    ranks = int(parts[3])
    if len(parts) >= 5:
        train_frequency = int(parts[4])
    else:
        train_frequency = 1
    if len(parts) >= 6:
        batch_size = int(parts[5])
    
    lastPart = dirs[-1]
    base = lastPart[:-3]
    parts = base.split("_")
    metric = parts[1]
    rank = int(parts[2])
    ret = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            temp = re.split(',', line.rstrip())
            time = int(temp[-2]) - int(temp[-3])
            count = int(temp[-4])
            ret.append((which, episode_block, batch_frequency, ranks, train_frequency, batch_size, metric, rank, time, count))
    return ret

@csvWriter()
def getTraceDataframe(dir, filter):
    """
    Gets a dataframe from directory of stream output files.
    All files in the directory should have .txt as the file extension.
    
    Parameters
    ----------
    dir : str
        Path with stream output

    Returns
    -------
    Dataframe
        Datafram from a directory of stream output.
    """
    files = getFiles(dir, filter, False)
    temp = []
    for f in files:
        try:
            temp.extend(readTraceFile(f))
        except Exception as e:
            print("Failed:", f, e)
    ret = pd.DataFrame(temp, columns=("which", "episode_block", "batch_frequency", "ranks", "train_frequency", "batch_size", "metric", "rank", "time", "count"))
    
    truthy = {"1":True, "true":True, "True":True, "0":False, "false":False, "False":False}
    ret["episode_block"] = ret["episode_block"].map(truthy)
    
    return ret

#%%
if __name__ == "__main__":
    if sys.argv[1] == "cart":
        df = getCartDataframe(sys.argv[2], csvDir=sys.argv[3], force=True)
        print(df.head())
    elif sys.argv[1] == "trace":
        getTraceDataframe(sys.argv[2], sys.argv[3], csvDir=sys.argv[4], force=True)
    else:
        files = getFailedCart(sys.argv[1])
        for f in files:
            print(f)
#%%