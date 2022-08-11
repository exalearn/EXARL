import math
import matplotlib.pyplot as plt

def createLineSlope(slope, yInt):
    """
    Creates a line through the origin following:
    y = slope * x + yInt
    
    Parameters
    ----------
    slope : int
        Slope of line
    yInt : int
        Y intercept
    
    Returns
    -------
    list
        Two points on line in a list in
        format [x0, x1, y0, y1]
    """
    x2 = 10.0 #Some point
    y2 = float(slope) * x2 + float(yInt)
    return [0, x2, yInt, y2]

def findLineIntersect(alpha, beta):
    """
    Finds the intersection between two lines.
    The input is two points on a line (output
    of createLineSlope).
    
    Parameters
    ----------
    alpha : int
        Two points on alpha line
    beta : int
        Two points on beta line
    
    Returns
    -------
    list
        Point of intersection [x, y]
    """
    slopeAlpha = (alpha[2] -  alpha[3]) / float(alpha[0] - alpha[1])
    slopeBeta = (beta[2] - beta[3]) / float(beta[0] - beta[1])
    x_int = (slopeAlpha * alpha[1] - alpha[3] - slopeBeta * beta[1] + beta[3]) / (slopeAlpha - slopeBeta)
    y_int = slopeAlpha * (x_int - alpha[1]) + alpha[3]
    return [x_int, y_int]

class Roofline:
    """
    This class is used to plot a roofline and experimental runs.
    The Roofline object allows for the plot to be built in parts
    (the rooflines and consequent data).

    Attributes
    ----------
    alpha : list
        List of alpha lines to plot
    beta : list
        List of beta lines to plot
    constBoxMultiplier : int
        Multiplier to increase plot range and domain by
    xstart : float
        Min x value to plot
    xend : float
        Max x value to plot
    ystart : float
        Min y value to plot
    yend : float
        Max y value to plot
    xlim : list
        Adjusts the x range of the plot [min, max]
    ylim : list
        Adjusts the y range of the plot [min, max]
    data : list
        Unlabeled data to plot in roofline
    labeledData : dictionary
        Label data to plot in roofline
    xaxis : str
        X axis label
    yaxis : str
        Y axis label
    title : str
        Plot title
    betaHandles : list
        Plot handles for beta legend
    alphaHandles : list
        Plot handles for alpha legend
    dataHandles : list
        Plot handles for data legend
    axis : axis
        Axis from subplots
    fig : figure
        Figure set if axis not given
    """
    def __init__(self, alpha, beta, xaxis, yaxis, title=None, xlim=None, ylim=None, axis=None):
        """
        Parameters
        ----------
        alpha : list
            List of tuple describing line.  A line is given
            by a name, slope, and y-intercept.
            e.g. [("Base FP", 0, 50), ("Peak FP", 0, 100)]
        beta : list
            List of tuple describing line. A line is given
            by a name, slope, and y-intercept.
            e.g. [("bw", 10, 0), ("Peak FP", 20, 0)]
        xaxis : str
            X axis label
        yaxis : str
            Y axis label
        title : str, optional
            Plot title
        xlim : list
            X axis range in format [xmin, xmax]
        ylim : list
            Y axis range in format [ymin, ymax]
        axis : axis
            Axis from subplot
        """
        # alpha.append(tuple([xl[0], float(xl[1]), float(xl[2]), xl[3]]))
        self.alphas = alpha
        # betas.append(tuple([xl[0], float(xl[1]), float(xl[2])]))
        self.betas = beta

        self.constBoxMultiplier = 100
        self.xstart = 0.001
        self.xend = 1 #0.001
        self.ystart = 0.01
        self.yend = 1 #0.01
        self.xlim = xlim
        self.ylim = ylim

        self.data = {}
        self.labeledData = {}
        self.amortized = {}
        
        self.xaxis = xaxis
        self.yaxis = yaxis

        self.title = title

        self.betaHandles = []
        self.alphaHandles = []
        self.dataHandles = []

        if axis is None:
            self.fig, self.axis = plt.subplots(1, 1)
        else:
            self.fig = None
            self.axis = axis


    def getStartStop(self, a, b, lst):
        """
        Used to figure out the starting and stopping on 
        the plot by returning the mins and maxs of the
        intersection points and the user's desired
        plotting range.

        Parameters
        ----------
        a : float
            Starting point of dimension (i.e. x or y)
        b : float
            Ending point of dimension (i.e. x or y)
        lst : list
            List of x or y coordinate of line intersections
        """
        pos = [a, b]
        pos.extend(lst)
        return min(pos), max(pos)

    def roofPlot(self):
        """
        Plots just the roofline.  This is an internal
        call.  To get the final plot call plot.
        """
        #Set axis
        self.axis.set_xscale("log")
        self.axis.set_yscale("log")
        self.axis.grid(color='black', linestyle='--', linewidth=0.1, axis='both', which='both')

        #Find intersection of alphas and betas
        intersection = []
        for l in self.betas:
            betaLine = createLineSlope(l[1], l[2])
            for a in self.alphas:
                alphaLine = createLineSlope(a[1], a[2])
                intersection.append(findLineIntersect(alphaLine, betaLine))

        #Find min and max for xy axis
        self.xstart, self.xend = self.getStartStop(self.xstart, self.xend, [item[0] for item in intersection])
        self.ystart, self.yend = self.getStartStop(self.ystart, self.yend, [item[1] for item in intersection])
        
        if self.xlim:
            self.xstart = self.xlim[0]
            self.xend = self.xlim[1] / self.constBoxMultiplier
        
        if self.ylim:
            self.ystart = self.ylim[0]
            self.yend = self.ylim[1] / self.constBoxMultiplier

        print(self.xstart, self.xend * self.constBoxMultiplier, self.ystart, self.yend * self.constBoxMultiplier)
        self.axis.plot(
            [self.xstart, 
            self.xend * self.constBoxMultiplier], 
            [self.ystart, 
            self.yend * self.constBoxMultiplier], 
            linestyle='None')

        #Plots the Beta lines and the verticals
        maxBeta = -1
        minIntercept = [0, -1]
        for l in self.betas:
            maxBeta = max(maxBeta, l[1])
            betaname = l[0]
            betaLine = createLineSlope(l[1], l[2])
            for a in self.alphas:
                if minIntercept[1] < a[2]:
                    minIntercept[0] = a[1]
                    minIntercept[1] = a[2]
            alphaLine = createLineSlope(minIntercept[0], minIntercept[1])
            intersect = findLineIntersect(alphaLine, betaLine)
            handle = self.axis.plot([0, intersect[0]], [0, intersect[1]], label=betaname)
            self.betaHandles.append(handle)
            self.axis.axvline(x=intersect[0], ymin=0, ymax=1.0, linestyle='--', dashes=(5, 5), alpha=0.5, c='black')

        #Plots the alphas
        for a in self.alphas:
            alphaname = a[0]
            alphaLine = createLineSlope(a[1], a[2])
            betaLine = createLineSlope(maxBeta, 0)
            intersect = findLineIntersect(alphaLine, betaLine)
            handle = self.axis.plot([intersect[0], self.xend * self.constBoxMultiplier], [alphaLine[2], alphaLine[3]], label=alphaname)
            self.alphaHandles.append(handle)

    def addAmortized(self, name, alphaName, betaName):
        """
        This adds an amortized roofline to be plotted
        by plotAmortized.  An amortized line is made between
        an alpha and beta lines.  The alpha and beta names
        should link to the desired lines.

        Parameters
        ----------
        name : string
            Label for the new line
        alphaName : string
            Name of alpha line (pi)
        betaName : string
            Name of beta line
        """
        self.amortized[name] = (alphaName, betaName)

    def plotAmortized(self, name, alphaName, betaName):
        """
        Plots just the amortized roofline.  This is an internal
        call.  To get the final plot call plot.
        """

        #Find intersection of alphas and betas
        alpha = [a for a in self.alphas if a[0] == alphaName][0]
        beta = [b for b in self.betas if b[0] == betaName][0]
        alphaLine = createLineSlope(alpha[1], alpha[2])
        betaLine = createLineSlope(beta[1], beta[2])
        intersection = findLineIntersect(alphaLine, betaLine)

        points = 10000
        a = math.ceil(math.log10(self.xstart))
        b = math.floor(math.log10(self.xend))
        pairs = [(10**x, 10**(x+1)) for x in range(a, b)]
        x = []
        for A, B in pairs:
            x.extend([A + (X * (B-A)/points) for X in range(points)])
        y = [(alpha[2]*beta[1]*X)/(alpha[2] + beta[1]*X) for X in x]
        handle = self.axis.plot(x, y, label=name)
        self.betaHandles.append(handle)

    def annotateLabel(self, x, y, label):
        """
        Plots the label for the annotated data.

        Parameters
        ----------
        x : float
            X-coordinate of the data point
        y : float
            Y-coordinate of the data point
        label : str
            Label of the coordinate
        """
        self.axis.annotate(
            label, 
            xy=(x, y), 
            xytext=(-10,10), 
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=1.0),
            textcoords='offset points' , 
            arrowprops=dict(arrowstyle='->', 
            connectionstyle='arc3,rad=0'),
            fontsize=8)

    def addLabelData(self, dataDict):
        """
        Adds labeled data to be plotted.
        Data should be in a two level dictionary
        where the series is the key for the upper level
        and the key of the second dictionary is the label
        of the data point.  The value of the inner most
        dictionary is a tuple of the X and Y coordinates.
        X and Y can be lists.

        Parameters
        ----------
        dataDict : Dictionary
            Data to plot.
        """
        self.labeledData.update(dataDict)

    def addData(self, dataDict):
        """
        Add unlabeled data to be plotted.
        Data should be provided in a dictionary where
        the key is the name of the data point and the
        value is a tuple of the x,y coordinates.
        X and Y can be lists.

        Parameters
        ----------
        dataDict : Dictionary
            Data to plot.
        """
        self.data.update(dataDict)

    def plotData(self, xList, yList, dataLabel, labels=None):
        """
        Plots data.  This is an internal plotting call.  Use
        plot to produce final plot.

        Parameters
        ----------
        xList : list
            List of x-coordinates to plot
        yList : list
            List of y-coordinates to plot
        dataLabel : str
            Name of series for legend
        labels : list
            List of labels for each data point
        """
        handle = self.axis.scatter(xList, yList, s=16, alpha=0.5, label=dataLabel)
        # marker = "v" if yList[-1] < yList[0] else "^"
        # handle = axis.plot(xList, yList, marker=marker, alpha=0.5, label=dataLabel)
        self.dataHandles.append(handle)
        if labels != None:
            [self.annotateLabel(x, y, label) for x, y, label in zip(xList, yList, labels)]
        self.xstart, self.xend = self.getStartStop(self.xstart, self.xend, xList)
        self.ystart, self.yend = self.getStartStop(self.ystart, self.yend, yList)

    def plot(self, saveFile=None, xlim=None, ylim=None):
        """
        This call produces the final roofline plot with labelled and unlabelled data.

        Parameters
        ----------
        saveFile : str, optional
            Name of file to save plot as
        xlim : list, optional
            X axis range in format [xmin, xmax]
        ylim : list, optional
            Y axis range in format [ymin, ymax]
        """
        
        # self.axis.clf()
        self.axis.set_xlabel(self.xaxis)
        self.axis.set_ylabel(self.yaxis)
        
        if self.title:
            self.axis.title(self.title)

        for series in self.data:
            self.plotData(self.data[series][0], self.data[series][1], series)
        
        for series in self.labeledData:
            x = []
            y = []
            labels = []
            for label in self.labeledData[series]:
                labels.append(label)
                x.append(self.labeledData[series][label][0])
                y.append(self.labeledData[series][label][1])
            self.plotData(x, y, series, labels)

        self.roofPlot()

        for series in self.amortized:
            self.plotAmortized(series, self.amortized[series][0], self.amortized[series][1])

        # axis = self.axis.gca()
        alphaLeg = self.axis.legend(handles=[item[0] for item in self.alphaHandles], loc='upper right')
        betaLeg = self.axis.legend(handles=[item[0] for item in self.betaHandles], loc='upper left')
        dataLeg = self.axis.legend(handles=[item for item in self.dataHandles], loc='center left', bbox_to_anchor=(1.03, 0.5))
        # dataLeg = axis.legend(handles=self.dataHandles, loc='lower right')
        self.axis.add_artist(alphaLeg)
        self.axis.add_artist(betaLeg)
        self.axis.add_artist(dataLeg)
        if xlim:
            self.axis.set_xlim(xmin=xlim[0], xmax=xlim[1])
        # else:
        #     self.axis.set_xlim(xmin=self.xstart, xmax=self.xend)
        if ylim:
            self.axis.set_ylim(ymin=ylim[0], ymax=ylim[1])
        # else:
        #     self.axis.set_ylim(ymin=self.ystart, ymax=self.yend)
        
        #Show graph!
        if self.fig is not None:
            if saveFile != None:
                self.fig.savefig(saveFile, bbox_inches='tight')
            self.fig.show()
