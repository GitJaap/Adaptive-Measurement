__author__ = 'J.J Wesdorp'
'''
Easy to use O(N) adaptive sampling algorithm, measures more points in 'interesting' intervals specified by the choice of a loss function. Works on a point by point basis within a specified x-range.
The user has to perform the following measurement loop: ask for a new coordinate with getNewX(), then insert the measured y-value back in to the algorithm, repeat...
Can also be used in a 2D scan on a line by line basis where previously measured lines can be used as information for the current line by calling getNewXUsingPreviousData().

    Copyright (C) 2015 J.J Wesdorp

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Contacting can be performed by email through Jaapwesdorp@gmail.com 

'''
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import ast
import scipy

class AdaptiveAlgorithm:
    '''
    Adaptive algorithm class used for optimized non-equidistant measurements. Measures more points in 'interesting' regions specified by the choice of a loss function.
    '''


    def __init__(self, xMin, xMax, nPrecision = 10e16, filename = "", filepath = "", qualityFunction = 'Second order diff',
                 historyType = 'Combined switch without alpha', spyViewPrecision = 300, initialValues = [[] ,[]]):
        '''initialize the lists of x and y values and others used by the class
        Parameters
        ------
        xMin: float
            minimum x-value of the measurement range
        xMax: float
            maximum x-value of the measurement range
        nPrecision : int
            sets the maximum precision to the precision you would have if you take an equidistant scan in the same range
            useful for forcing the algoritm to not take too many points in the peak for example
            the minimum distance between two x-points is now calculated by (xMax-xMin) / nPrecision
        filename : String
            sets the filename of the data file. If this is not specified or equal to "" it will not create a data file
        filepath : String
            Sets the filepath of data file, default is the current working directory
            file is stored as filepath/filename_filedate
        qualityFunction : String
            Dictionary keys used to specify the loss function used(previously called quality function) when calling getNewX()
            allowed values: see dictionary below
        historyType: String
            Dictionary key used to specify the type of history used when calling getNewXUsingPreviousData()
        spyViewPrecision: int
            size of interpolated lines when spyView compatible data format is chosen for output        
        initialValues : [[xValues], [yValues]] : [[float],[float]]
            intitial x and y values appended to the xCur and yCur so the measurement doesnt start at zero
            xValues and yValues should be the same length
        -------
        Returns
        -------
        function is void
        ------
        '''
        #create a dictionary for the different quality types available
        self.qTypes = {
        'Equidistant' : self.dEqui,
        'First order diff': self.dDiff,
        'Second order diff divided by dx': self.dDiff2,
        'First order with y-preference' : self.dDiffY,
        'Second order diff' : self.dDiff2dx,
        'Second order diff sqrt': self.dDiff2dxSqrt,
        'First + Second order diff' : self.dDiff12dx,
        'Second order diff times dx' : self.dDiff2ddx
        }

        #create a dictionary for the different history types available
        self.hTypes = {
            'Combined switch with alpha no dx in history' : self.calcXNextByFitAndDiff21,
            'Combined switch with alpha with dx in history' : self.calcXNextByFitAndDiff2,
            'Combined switch without alpha' : self.calcXNextByFitAndDiff,
            'General switch ' : self.calcXNextByFitAndDiff2,
            'Single quality without switch' : self.calcXNextByFitAndDiff3,
            'Old history quality without maximum crit' : self.calcXNextbyFitAndDiff4
        }
        self.hType = historyType
        self.xCur = []
        self.yCur = []
        #allocate memory for the sorted version of the x and y-values
        self.xSort = []
        self.ySort = []
        self.xMin = float(xMin)
        self.xMax = float(xMax)
        self.quality = qualityFunction
        self.blocks = []
        self.writtenBlocksCounter = 0
        #algoritm based on minimum separation
        self.nPrecision = nPrecision
        self.minimumInterval = float(xMax-xMin) / nPrecision
        self.z = 0
        self.spyViewPrecision = spyViewPrecision
        #algorithm based on prev measurement variable
        self._DevMode = False #DEVELOPMENT MODE USED for enhanced plotting(more information)
        #---only for devmode
        self.normInfo = [] #list of the same size of xCur, which holds info about which part of which norm was used to take the new point
        #------

        #add possible initial values to the lists
        self.xCur.extend(initialValues[0])
        self.yCur.extend(initialValues[1])

        #initialize the filereader and open the file to write
        if(filename != ""):
            if len(filepath) == 0:
                self.f = open(filename + time.strftime("_%d_%m_%Y_%H_%M_%S") + ".dat", 'w')
            else:
                self.f = open(filepath + '/' + filename + time.strftime("_%d_%m_%Y_%H_%M_%S") + ".dat", 'w')
            #Write intial information
            self.f.write("# Adaptive Measurement data, Interpolated to a regular grid for view in spyView \n")
            self.f.write("# Column 1: x-values \n")
            self.f.write("# Column 2: z-values\n")
            self.f.write("# Column 3: corresponding y - values \n")

            self.f.write("# This data has the maximum precision of an equidistant scan of %d points\n" % (self.nPrecision))
            self.f.write("# Adaptive algorithm Normtype is: %s\n" % (self.quality))
            self.f.write("# Maximum x-value = %f \n" %(self.xMax))
            self.f.write("# Minimum x-value = %f \n\n" %(self.xMin))

        self.switched = False

    def getNewX(self):
        '''Calculates the next x-value to measure only based on the quality function specified and previous data in this current line.
        Also appends this xNext to the list of currently measured points xCur
        ------
        REQUIRES:
        ------
         Before calling this function a second time, the y-value corresponding to xNext should be added to the list of measured values by
         using the function addY(yValue)
        ------
        Returns:
        ------
            xNext: [float]
                the next x-value to measure.
            '''
        if(len(self.xCur) != len(self.yCur)):
            print('Yvalues not set properly: please append an y value after each new X request')
        else:
            xNext = -99999
            if (len(self.xCur) <= 2) & self._DevMode:
                    self.normInfo.append([1,0,-1,0])
            if len(self.xCur) == 0:
                xNext = self.xMin
            elif len(self.xCur) == 1:
                xNext = self.xMax
            elif len(self.xCur) == 2:
                xNext = self.xMin + (0.5) * (self.xMax - self.xMin)
            else:
                #create numpy arrays of the point list
                xCurA = np.array(self.xCur)
                yCurA = np.array(self.yCur)
                yCurA = yCurA[np.argsort(xCurA)]
                xCurA.sort()
                #calculate the distances of each interval
                dis = self.qTypes[self.quality](xCurA, yCurA)
                #get the indexes of the largest distances
                maxDisIndices = np.argsort(dis)
                #check that xNext is not within the minimum distance specified
                xFound = False
                i = -1
                #self.minimumInterval = (max(xCurA) - min(xCurA)) / float(10 + len(self.xCur) * 2)
                while(xFound == False):
                    if (xCurA[maxDisIndices[i]+1] - xCurA[maxDisIndices[i]]) <= self.minimumInterval:
                        i = i - 1
                    else:
                        xFound = True
                xNext = xCurA[maxDisIndices[i]] + abs(xCurA[maxDisIndices[i]+1] - xCurA[maxDisIndices[i]]) / 2
                if self._DevMode:
                    xWeight = (((xCurA[maxDisIndices[i]+1] - xCurA[maxDisIndices[i]]) / (max(xCurA) - min(xCurA))) ** 2) / dis[maxDisIndices[i]] ** 2
                    yWeight = 1 - xWeight
                    self.normInfo.append([xWeight,yWeight, -1, 0])

            self.xCur.append(xNext)
            return xNext

    def getNewXUsingPreviousData(self, prevBlock = -1, alpha = 0.00):
        '''
        Calculates the next x-value to measure based on the quality function specified and previous data of the line specified in prevBlock.
        Tries to reduce the distance between an interpolation of the current data on the previous data as fast as possible,
        until for an interval the distance reduction is smaller than the actual distance between previous and current measurement. The
        algorithm will then switch to the explorative quality function instead of using the previous data. When alpha is specified,
        the algorithm will also switch when the current data interpolated on previous data is within alpha percent of the y-span
        Also appends this xNext to the list of currently measured points xCur
        ------
        Parameters:
            prevBlock:
                The previously measured line used as a starting point for this measurement(used for deciding the next point when
                history is applicable.
            alpha:
                desired history precision parameter for each interval. When the gain in distance between the current and previous measurement is
                smaller than alpha * ySpan the algorithm will switch to the specified explorative quality function. This allows the user to control
                the way points are taken after a reasonable approximation to the previous data has been measured.
        ------
        REQUIRES:
        ------
         Before calling this function a second time, the y-value corresponding to xNext should be added to the list of measured values by
         using the function addY(yValue)
        ------
        Returns:
        ------
            xNext: [float]
                the next x-value to measure.
        '''
        if len(self.blocks) == 0:
            return self.getNewX() #if there is no previous data don't use history norm
        if(len(self.xCur) != len(self.yCur)):
            print('Yvalues not set properly: please append an y value after each new X request')
        else:
            #----DEVELOPMODE
            if (len(self.xCur) <= 2) & self._DevMode:
                self.normInfo.append([1,0,-1,0])
            #---------
            if len(self.xCur) == 0:
                xNext = self.xMin
            elif len(self.xCur) == 1:
                xNext = self.xMax
            elif len(self.xCur) == 2:
                xNext = self.xMin + (0.5) * (self.xMax - self.xMin)
            else:
                #create numpy arrays of the point list
                xCurA = np.array(self.xCur)
                yCurA = np.array(self.yCur)
                yCurA = yCurA[np.argsort(xCurA)]
                xCurA.sort()
                 #estimate the change of the function by interpolating the previous 2 lines
                #if len(self.blocks) >= 2:
                #    yChange = np.interp(self.blocks[-1][0], self.blocks[-2][0], self.blocks[-2][1]) - self.blocks[-1][1]
                #    yExtrapolated = self.blocks[-1][1] + yChange
                #else:
                yExtrapolated = self.blocks[prevBlock][1]
                #calculate xNext based on history specified
                xNext = self.hTypes[self.hType](xCurA, yCurA, np.array(self.blocks[prevBlock][0]), np.array(yExtrapolated), alpha)

            self.xCur.append(xNext)
            return xNext
    '''
#-------------------------------------------------------------------------------------------------------------------
#
# Different Loss functions usable for the algorithm(previously called quality functions)
#
# -------------------------------------------------------------------------------------------------------------------
    '''

    def dEqui(self,xVec, yVec):
        '''calculates the equidistant distance norm: just dx
        :return dis[i] = distance between point i and i-1
        '''
        return np.diff(xVec / (max(xVec) - min(xVec)))
    def dDiff(self, xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector dx^2+dy^2
        :returns dis[i] = distance between point i+1 and i'''

        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))

        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))

        #calculate the distance norm
        dis = np.sqrt(np.diff(xVecNorm) ** 2 + np.diff(yVecNorm) ** 2)
        return dis

    def dDiff2(self,xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector based on dx dy and second derivative of y
        :returns dis[i] = distance between point i and i-1'''

        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))

        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))
        #calculate the second derivate
        dydx = np.diff(yVecNorm) / np.diff(xVecNorm)
        #dydxNorm = dydx / (max(dydx) - min(dydx))
        diff2 = np.diff(dydx)
        diff2AV = np.zeros(len(xVecNorm) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:]))  / 2.0
        diff2AV = diff2AV / np.diff(xVecNorm)
        #calculate the distance norm
        dis = np.sqrt(np.diff(xVecNorm) ** 2 + diff2AV ** 2)
        #print('max xVecnorm value: %f, max diff2AVnorm hvalue: %f' % (max(np.diff(xVecNorm) ** 2), max(diff2AV ** 2 )))
        return dis

    def dDiff2dx(self,xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector based on dx dy and second derivative of y
        :returns dis[i] = distance between point i and i-1'''
        l = 0.5
        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))
        dx = np.diff(xVecNorm)
        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))
        #calculate the second derivate
        diff2 = np.diff(np.diff(yVecNorm) / dx)
        diff2AV = np.zeros(len(xVecNorm) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx
        '''
        if(hasattr(self,'disCounter')):
            self.disCounter += 1
        else:
            self.disCounter = 0
        print('%d max ddy^2: %.2e , max dx^2: %.2e' % (self.disCounter, max(diff2AV ** 2), max(dx ** 2)))
        '''
        #calculate the distance norm
        dis = np.sqrt(dx ** 2 * (l) + diff2AV ** 2 * (1-l))
        #print('max xVecnorm value: %f, max diff2AVnorm hvalue: %f' % (max(np.diff(xVecNorm) ** 2), max(diff2AV ** 2 )))
        return dis

    def dDiff2dxSqrt(self,xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector based on dx dy and second derivative of y
        :returns dis[i] = distance between point i and i-1'''
        l = 0.5
        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))
        dx = np.diff(xVecNorm)
        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))
        #calculate the second derivate
        diff2 = np.diff(np.diff(yVecNorm) / dx)
        diff2AV = np.zeros(len(xVecNorm) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx ** 1.1
        #calculate the distance norm
        dis = np.sqrt(dx ** 2 * (l) + diff2AV ** 2 * (1-l))
        #print('max xVecnorm value: %f, max diff2AVnorm hvalue: %f' % (max(np.diff(xVecNorm) ** 2), max(diff2AV ** 2 )))
        return dis

    def dDiff2ddx(self,xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector based on dx dy and second derivative of y multiplied by dx^2
        :returns dis[i] = distance between point i and i-1'''
        l = 0.5
        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))
        dx = np.diff(xVecNorm)
        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))
        #calculate the second derivate
        diff2 = np.diff(np.diff(yVecNorm) / dx)
        diff2AV = np.zeros(len(xVecNorm) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx * dx
        #calculate the distance norm
        dis = np.sqrt(dx ** 2 * (l) + diff2AV ** 2 * (1-l))
        #print('max xVecnorm value: %f, max diff2AVnorm hvalue: %f' % (max(np.diff(xVecNorm) ** 2), max(diff2AV ** 2 )))
        return dis

    def dDiff12dx(self,xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector based on dx dy and second derivative of y
        :returns dis[i] = distance between point i and i-1'''

        #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))
        dx = np.diff(xVecNorm)
        if(np.max(yVec) == np.min(yVec)):
            yVecNorm = yVec / np.max(yVec)
        else:
            yVecNorm = yVec / (np.max(yVec) - np.min(yVec))
        dy = np.diff(yVecNorm)
        #calculate the second derivate
        diff2 = np.diff(dy / dx)
        diff2AV = np.zeros(len(xVecNorm) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx
        #calculate the distance norm
        dis = np.sqrt(dx ** 2 + 0.5 * (diff2AV ** 2 + dy ** 2))
        ''' #to show whhat choice is being made in the algorithm.
        if(hasattr(self,'disCounter')):
            self.disCounter += 1
        else:
            self.disCounter = 0
        dxmax = dx[np.argmax(dis)] ** 2
        ddydymax = 0.5 * (diff2AV[np.argmax(dis)] ** 2 + dy[np.argmax(dis)] ** 2)
        if(dx[np.argmax(dis)] ** 2 >  0.5 * (diff2AV[np.argmax(dis)] ** 2 + dy[np.argmax(dis)] ** 2)):
            choice = 'DX'
        else:
            choice = 'DY/DDY'
        print('%.4d, dx^2: %.2e,dyddy: %.7e , x=%.2f, [%s]' % (self.disCounter, max(dx ** 2), max(dy **2 + diff2AV ** 2) / 2.0 , xVec[np.argmax(dis)]+ np.diff(xVec)[np.argmax(dis)]/2.0, choice))
        '''
        #print('max xVecnorm value: %f, max diff2AVnorm hvalue: %f' % (max(np.diff(xVecNorm) ** 2), max(diff2AV ** 2 )))
        return dis


    def dDiffY(self, xVec, yVec):
        '''calculates the distance norm between every point in the x and y vector also giving preference to high y-values
        :returns dis[i] = distance between point i and i-1'''
         #first normalize all vectors used in the norm so they have equal weights
        xVecNorm = xVec / (np.max(xVec) - np.min(xVec))
        yVecNorm = yVec / (np.max(yVec) - np.min(yVec))

        #get the average y-value for each interval
        yAv = 0.5 * abs(yVecNorm[0:-1] + yVecNorm[1:])

        #calculate the distance norm taken the x-distance y-distance into account
        dis = np.sqrt(np.diff(xVecNorm) ** 2 + np.diff(yVecNorm) ** 2)
        #now multiply the distance by a factor between 1 and 2 depending on the y-coordinate
        dis = dis * (yAv + 0.6) #extra constant is added so zero y-values are not completely ignored
        return dis
    '''
#-------------------------------------------------------------------------------------------------------------------
#
# Utility functions
#
#-------------------------------------------------------------------------------------------------------------------
    '''
    def sort(self):
        '''sorts the xCur and yCur list and puts them in self.xSort and self.ySort'''
        xCurA = np.array(self.xCur)
        yCurA = np.array(self.yCur)
        self.ySort = yCurA[np.argsort(xCurA)].tolist()
        self.xSort = xCurA[np.argsort(xCurA)].tolist()

    def plotValues1D(self, mark = 'x', block = -9999, n =-9999):
        '''plots the current x-values vs current yvalues with the given mark'''
        self.sort()
        if block == -9999: #no block specified so use current measurement
            if(n ==-9999):
                n = len(self.xCur)
            xCur = self.xCur[0:n]
            yCur = self.yCur[0:n]
            if(self._DevMode):
                normInfo = self.normInfo
        else:
            if(n ==-9999):
                n = len(self.blocks[block][2])
            xCur = self.blocks[block][2][0:n]
            yCur = self.blocks[block][3][0:n]

            if(self._DevMode):
                normInfo = self.blocks[block][5]
        xCurA = np.array(xCur)
        yCurA = np.array(yCur)
        ySort = yCurA[np.argsort(xCurA)].tolist()
        xSort = xCurA[np.argsort(xCurA)].tolist()
        plt.plot(xSort,ySort, mark)
        #--------DEV MODE gives extra plot info
        if self._DevMode:
            t = np.linspace(min(ySort), max(ySort), len(ySort))
            if(len(normInfo) != 0):
                for i in range(0,len(xCur)):
                    #check which norminfo is used
                    if(normInfo[i][2] == -1):
                        if((normInfo[i][0] >= normInfo[i][1]) & bool(normInfo[i][3])): #x greater than ypart and fit true
                            plt.plot(xCur[i], t[i], 'sg', markersize = 6)
                        elif (normInfo[i][0] < normInfo[i][1]) & bool(normInfo[i][3]): #x smaller than ypart and fit true
                            plt.plot(xCur[i], t[i], 'sr', markersize = 6)
                        elif (normInfo[i][0] >= normInfo[i][1]) & (normInfo[i][3] == False): #x greater than ypart and fit false
                            plt.plot(xCur[i], t[i], 'pg', markersize = 6)
                        else: #x smaller than ypart and fit false
                            plt.plot(xCur[i], t[i],'pr', markersize = 6)
                    else:#using all in one norm
                        mi = np.argmax(normInfo[i])
                        if(mi == 0):
                             plt.plot(xCur[i], t[i], 'pg', markersize = 6)
                        elif(mi == 1):
                             plt.plot(xCur[i], t[i], 'pr', markersize = 6)
                        elif(mi == 2):
                            plt.plot(xCur[i], t[i], 'pc', markersize = 6)
                        elif(mi == 3):
                             plt.plot(xCur[i], t[i], 'sy', markersize = 6)
            else:
                plt.plot(xCur, t, 'pg')
    def plotValues2D(self, fig, ax, nGrid, vmin = 0, vmax = 0, extent = [0,1,0,1], cbLabel = ''):
        import pylab
        aspect = abs(extent[1] - extent[0]) / abs(extent[2] - extent[3])
        meshGrid = np.zeros((len(self.blocks),nGrid))
        for j in range(0,len(self.blocks)):
            meshGrid[j,:] = np.interp(np.linspace(self.xMin,self.xMax, nGrid),self.blocks[j][0],self.blocks[j][1])
        im = ax.imshow(meshGrid, cmap=pylab.cm.RdBu, vmin=vmin, vmax=vmax,extent=extent, aspect=aspect)
        im.set_interpolation('none')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(cbLabel)

    def addY(self, yValue):
        '''Important function used to call every time after getNewX() to append an y-value to the current lists'''
        self.yCur.append(yValue)

    def animate(self, interval = 40):
        '''animates the measured x and y values in sequence with the measurement
        :param interval specifies the waiting time between showing new points
        '''
        nPoints = len(self.xCur)
        self.fig, ax = plt.subplots()
        plt.title('animation of Adaptive algoritm with %d Points' % nPoints)
        self.line, = ax.plot(self.xCur, self.yCur, 'x')
        for i in range(2, len(self.xCur)):
            xCurTemp = np.array(self.xCur[0:i])
            xCurTemp.sort()
        #animate the measurement protocol in a figure
        animation = ani.FuncAnimation(self.fig, self.__readPointAnimation, frames = nPoints, interval = interval, repeat = True, blit=True)
        plt.show()

    def __readPointAnimation(self,i):
        '''function required by the matplotlib.FuncAnimation function'''
        self.line.set_xdata(self.xCur[0:i])
        self.line.set_ydata(self.yCur[0:i])
        return self.line,


    def writeRawData(self, filename = "", filepath = "", addDate = True):
        '''writes the current values to a file specified in filename in special adaptive format keeping the order of points taken'''
        if len(filepath) == 0:
            if(addDate == True):
                f = open(filename + time.strftime("_%d_%m_%Y_%H_%M_%S") + ".Adat", 'w')
            else:
                f = open(filename + ".Adat", 'w')
        else:
            if(addDate == True):
                f = open(filepath + '/' + filename + time.strftime("_%d_%m_%Y_%H_%M_%S") + ".Adat", 'w')
            else:
                f = open(filepath + '/' + filename + ".Adat", 'w')
        f.write("# Adaptive Measurement data, cannot be viewed in spyView \n")
        f.write("# Column 1: x-values \n")
        f.write("# Column 2: z-values\n")
        f.write("# Column 3: corresponding y - values \n")
        f.write("# Column 4: sorted x-data\n")
        f.write("# Column 5: sorted y-data\n")
        if self._DevMode:
            f.write("# Column 6: norminfo[dxWeight, y part of norm weight, fit used(boolean)]\n")
        f.write("# This data has the maximum precision of an equidistant scan of %d points\n" % (self.nPrecision))
        f.write("# Normtype is: %s\n" % (self.quality))
        f.write("# Maximum x-value = %f \n" %(self.xMax))
        f.write("# Minimum x-value = %f \n" %(self.xMin))
        f.write("# History used: %s\n" % (self.hType))
        f.write("# Developmentmode: %s\n\n" % self._DevMode)
        #write all blocks
        for i in range(0, len(self.blocks)):
            for j in range(0,len(self.blocks[i][0])):
                f.write('%.6e' % self.blocks[i][0][j] + '\t' + '%.6e' % self.blocks[i][4] + '\t'+
                        '%.6e' % (self.blocks[i][1][j])+ '\t%.6e' %(self.blocks[i][2][j]) + '\t%.6e' % self.blocks[i][3][j])
                if self._DevMode:
                    f.write('\t[%.2f,%.2f, %.2f, %.2f]' %(self.blocks[i][5][j][0], self.blocks[i][5][j][1], self.blocks[i][5][j][2],  self.blocks[i][5][j][3]))
                f.write('\n')
            f.write('\n')
        #finally write the current measurement
        for i in range(0, len(self.yCur)):
            f.write('%.6e' % self.xCur[i] + '\t' + 'NotSpecified\t' + '%.6e' % self.yCur[i] + '\n')
        f.close()

    def readDataFromFile(self, filename):
        '''reads single 1D plot data from special Adat file created by the write1DFile function'''
        with open(filename,'r') as f:
            #loop through all lines except the comments and newlines
            data = []
            started = False
            for line in f:
                if((line[0] != '#') & (line[0] != '\n') & (line[0] != ' ')):
                    started = True
                    dataLine = line.split('\t') # separate the columns in a list
                    self.xCur.append(float(dataLine[3])) #convert to floats
                    self.yCur.append(float(dataLine[4]))
                    self.xSort.append(float(dataLine[0]))
                    self.ySort.append(float(dataLine[2]))
                    if(self._DevMode):
                        self.normInfo.append(ast.literal_eval(dataLine[5]))
                        #print(ast.literal_eval(dataLine[5]))
                else:
                    if started == True:
                        self.startNewBlock(float(dataLine[1]))


    def startNewBlock(self, zValue = 0):
        '''starts a new block of data, so resets the x and y data and stores the previous measurement in a block, with the amount of points in the mesh specified by the maximum precision given
        :param zValue [float] The new z-value of this block.
        '''
        #first add sorted and unsorted data to the old block
        self.z = zValue
        self.sort()
        if self._DevMode:
            self.blocks.append([self.xSort, self.ySort, self.xCur, self.yCur, self.z, self.normInfo])
            self.normInfo = []
        else:
            self.blocks.append([self.xSort, self.ySort, self.xCur, self.yCur, self.z])
        #now delete the old lists of current data
        self.xCur = []
        self.yCur = []
        self.xSort = []
        self.ySort = []

    def writeBlock(self):
        '''writes the last block to a file in standard .dat format readable by spyView'''
        #
        xSpy = np.arange(self.xMin, self.xMax, (self.xMax - self.xMin) / self.spyViewPrecision)
        ySpy = np.interp(xSpy, self.blocks[-1][0], self.blocks[-1][1])
        #data format will be xSpy, z, ySpy, xSort, ySort, xCur, yCur
        for i in range(0, self.spyViewPrecision):#len(self.blocks[-1][0])):
            self.f.write("%.6e\t%.6e\t%.6e\t\n" %
                (xSpy[i], self.blocks[-1][4], ySpy[i]))
        self.f.write('\n')
        self.f.flush()

    def closeFile(self):
        self.f.close()

    def setPrecision(self, n):
        self.nPrecision = n
        self.minimumInterval = (self.xMax-self.xMin) / self.nPrecision






    '''
#-----------------------------------------------------------------------------------------------------------------------

#Different history using switching methods

#----------------------------------------------------------------------------------------------------------------------
    '''
    def calcXNextByFitAndDiff(self, xNew, yNew, xPrev, yPrev, alpha = -9999):
        #history norm with maximum based switching criterium and dx in his norm
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        #now calculate for each interval the distance between previous and current scan
        yIntp = np.interp(xNew,xPrev,yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)

        fitDis = abs(yPrev - yPrevIntp)
        #now we have to determine the maximum distance per interval
        intervalIndices = np.searchsorted(xPrev, xNew)
        intervalList = np.split(fitDis, intervalIndices)[1:-1]
        intervalList = [(i if len(i) else [0]) for i in intervalList] #Add negative values for intervals which contain no xprev, so they will not be used in the fitnorm
        maxList = np.array(map(max,intervalList)) #max distance per interval
        maxFitIndices = np.array(map(np.argmax, intervalList))
        #calculate the average error per xNew value
        errors = abs(np.interp(xNew,xPrev,yPrev) - yNew)
        maxErrors = np.maximum(errors[0:-1],errors[1:])
        #create a boolean vector for which distance norm should be used
        useFit = maxErrors < maxList
        #calculate the distance vectors
        disEx = self.qTypes[self.quality](xNew, yNew)
        disHis =  (useFit) * np.sqrt(dx ** 2 + (maxList / yNorm) **2)
        maxExIndices = np.argsort(disEx)
        maxHisIndices = np.argsort(disHis)

        #now find the interval which will reduce the quality the most according to assumptions
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        j = -1
        while(xFound == False):
            maxInt = maxExIndices[i]
            if useFit[maxInt]: #We have to use the history Q in this interval so we cannot use this one
                if maxExIndices[i] == maxHisIndices[j]:
                    i = i - 1
                maxInt = maxHisIndices[j]
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    j = j - 1
                else:
                    xFound = True
            else:
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    i = i-1
                else:
                    xFound = True
        #fit is used so determine the new x coordinate in a smarter way
        if useFit[maxInt] == True:
            if((dx[maxInt] ** 2) / ((1 - useFit[maxInt]) * disEx[maxInt] + useFit[maxInt] * disHis[maxInt]) ** 2 < 0.5): #y-part is more important
                xNext = xPrev[intervalIndices[maxInt]+maxFitIndices[maxInt]]
            else: #x-part is more important so choose the next x in the middle to decrease dx as much as possible
                xNext = xNew[maxInt] + abs(xNew[maxInt+1] - xNew[maxInt]) / 2
            #print('fit used on x = %f' % xNext)
            #print(sum(useFit))
        else:
            xNext = xNew[maxInt] + abs(xNew[maxInt+1] - xNew[maxInt]) / 2
            #print('explorative norm used on x = %f' % xNext)
            #print(sum(useFit))
        #---------------------------------extra developer info#
        if self._DevMode:
            xWeight = (dx[maxInt] ** 2) / (((1 - useFit[maxInt]) * disEx[maxInt] + useFit[maxInt] * disHis[maxInt]) ** 2)
            yWeight = 1 - xWeight
            self.normInfo.append([xWeight,yWeight, -1, useFit[maxInt]])
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext

    def calcXNextByFitAndDiff2(self, xNew, yNew, xPrev, yPrev, alpha = 0.01):
        #history norm with maximum based switching criterium and dx in his norm and alpha as max his precision
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        #now calculate for each interval the distance between previous and current scan
        yIntp = np.interp(xNew,xPrev,yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)

        fitDis = abs(yPrev - yPrevIntp)
        #now we have to determine the maximum distance per interval
        intervalIndices = np.searchsorted(xPrev, xNew)
        intervalList = np.split(fitDis, intervalIndices)[1:-1]
        intervalList = [(i if len(i) else [0]) for i in intervalList] #Add negative values for intervals which contain no xprev, so they will not be used in the fitnorm
        maxList = np.array(map(max,intervalList)) #max distance per interval
        maxFitIndices = np.array(map(np.argmax, intervalList))
        #calculate the average error per xNew value
        errors = abs(np.interp(xNew,xPrev,yPrev) - yNew)
        maxErrors = np.maximum(errors[0:-1],errors[1:])
        #create a boolean vector for which distance norm should be used
        useFit = (maxErrors < maxList) & (maxList / yNorm > alpha)

        #also check for the stopping criterium to be reached.

        #calculate the distance vectors
        disEx = self.qTypes[self.quality](xNew, yNew)
        disHis =  (useFit) * np.sqrt(dx ** 2 + (maxList / yNorm) **2)
        maxExIndices = np.argsort(disEx)
        maxHisIndices = np.argsort(disHis)

        #now find the interval which will reduce the quality the most according to assumptions
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        j = -1
        while(xFound == False):
            maxInt = maxExIndices[i]
            if useFit[maxInt]: #We have to use the history Q in this interval so we cannot use this one
                if maxExIndices[i] == maxHisIndices[j]:
                    i = i - 1
                maxInt = maxHisIndices[j]
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    j = j - 1
                else:
                    xFound = True
            else:
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    i = i-1
                else:
                    xFound = True
      #fit is used so determine the new x coordinate in a smarter way
        if useFit[maxInt] == True:
            if((dx[maxInt] ** 2) / ((1 - useFit[maxInt]) * disEx[maxInt] + useFit[maxInt] * disHis[maxInt]) ** 2 < 0.5): #y-part is more important
                xNext = xPrev[intervalIndices[maxInt]+maxFitIndices[maxInt]]
            else: #x-part is more important so choose the next x in the middle to decrease dx as much as possible
                xNext = xNew[maxInt] + abs(xNew[maxInt+1] - xNew[maxInt]) / 2
            #print('fit used on x = %f' % xNext)
            #print(sum(useFit))
        else:
            xNext = xNew[maxInt] + abs(xNew[maxInt+1] - xNew[maxInt]) / 2
            #print('explorative norm used on x = %f' % xNext)
            #print(sum(useFit))
        #---------------------------------extra developer info#
        if self._DevMode:
            xWeight = (dx[maxInt] ** 2) / (((1 - useFit[maxInt]) * disEx[maxInt] + useFit[maxInt] * disHis[maxInt]) ** 2)
            yWeight = 1 - xWeight
            self.normInfo.append([xWeight,yWeight,-1, useFit[maxInt]])
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext
    def calcXNextByFitAndDiff21(self, xNew, yNew, xPrev, yPrev, alpha = 0.03):
        #dhistory with alpha and no dx, but switching at maximum criterium
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        #now calculate for each interval the distance between previous and current scan
        yIntp = np.interp(xNew,xPrev,yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)

        fitDis = abs(yPrev - yPrevIntp)
        #now we have to determine the maximum distance per interval
        intervalIndices = np.searchsorted(xPrev, xNew)
        intervalList = np.split(fitDis, intervalIndices)[1:-1]
        intervalList = [(i if len(i) else [0]) for i in intervalList] #Add negative values for intervals which contain no xprev, so they will not be used in the fitnorm
        maxList = np.array(map(max,intervalList)) #max distance per interval
        maxFitIndices = np.array(map(np.argmax, intervalList))
        #calculate the average error per xNew value
        errors = abs(np.interp(xNew,xPrev,yPrev) - yNew)
        maxErrors = np.maximum(errors[0:-1],errors[1:])
        #create a boolean vector for which distance norm should be used
        useFit = (maxErrors < maxList) & (maxList / yNorm > alpha)

        #also check for the stopping criterium to be reached.

        #calculate the distance vectors
        disEx = self.qTypes[self.quality](xNew, yNew)
        disHis =  (useFit) * np.sqrt((maxList / yNorm) **2)
        maxExIndices = np.argsort(disEx)
        maxHisIndices = np.argsort(disHis)

        #now find the interval which will reduce the quality the most according to assumptions
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        j = -1
        while(xFound == False):
            maxInt = maxExIndices[i]
            if useFit[maxInt]: #We have to use the history Q in this interval so we cannot use this one
                if maxExIndices[i] == maxHisIndices[j]:
                    i = i - 1
                maxInt = maxHisIndices[j]
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    j = j - 1
                else:
                    xFound = True
            else:
                if (xNew[maxInt+1] - xNew[maxInt]) <= self.minimumInterval:
                    i = i-1
                else:
                    xFound = True
      #fit is used so determine the new x coordinate in a smarter way
        if useFit[maxInt] == True:
            xNext = xPrev[intervalIndices[maxInt]+maxFitIndices[maxInt]]
        else:
            xNext = xNew[maxInt] + abs(xNew[maxInt+1] - xNew[maxInt]) / 2
            #print('explorative norm used on x = %f' % xNext)
            #print(sum(useFit))
        #---------------------------------extra developer info#
        if self._DevMode:
            xWeight = (1-useFit[maxInt]) * (dx[maxInt] ** 2) / (disEx[maxInt] **2)
            yWeight = 1 - xWeight
            self.normInfo.append([xWeight,yWeight, -1,  useFit[maxInt]])
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext

    def calcXNextByFitAndDiff22(self,xNew, yNew, xPrev, yPrev, alpha = -9999):
        #history with general switching criterium
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        #now calculate for each interval the distance between an interpolation of the previous scan with current x, and the previous scan itself.
        yIntp = np.interp(xNew, xPrev, yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)
        fitDis = abs(yPrev - yPrevIntp)
        #now we have to determine the maximum distance per interval
        intervalIndices = np.searchsorted(xPrev, xNew)
        intervalList = np.split(fitDis, intervalIndices)[1:-1]
        intervalList = [(i if len(i) else [0]) for i in intervalList] #Add negative values for intervals which contain no xprev, so they will not be used in the fitnorm
        maxList = np.array(map(max,intervalList)) #max distance per interval
        maxFitIndices = np.array(map(np.argmax, intervalList))
        #calculate the average error per xNew value
        errors = abs(yIntp - yNew)
        #create a boolean vector for which distance norm should be used
        useFit = max(errors) < max(fitDis)
        if(useFit == False) & (self.switched == False):
            self.switched = True
            print('Line %d switched to explorative mode at %d Points ' %(len(self.blocks), len(self.xCur)))
        if(useFit == True):
            self.switched = False
        #calculate the distance vector
        dis = (useFit) * np.sqrt(dx ** 2 + (maxList / yNorm) **2) +  self.qTypes[self.quality](xNew, yNew) * (1 - useFit)
        #print(useFit)
        #search for the next X to send out based on the dis vector
        maxDisIndices = np.argsort(dis)
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        #interval = max([(self.xMax - self.xMin) / (len(xNew) * 4), (self.xMax - self.xMin) / self.nPrecision])
        #interval = max([(self.xMax - self.xMin) / (len(xNew) * 4), (self.xMax - self.xMin) / self.nPrecision])
        #check if minimum interval size is not exceeded
        while(xFound == False):
            if (xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) <= self.minimumInterval:
                i = i - 1
            else:
                xFound = True
        #fit is used so determine the new x coordinate in a smarter way
        if useFit == True:
            if((dx[maxDisIndices[i]] ** 2) / dis[maxDisIndices[i]] ** 2 < 0.5): #y-part is more important
                xNext = xPrev[intervalIndices[maxDisIndices[i]]+maxFitIndices[maxDisIndices[i]]]
            else: #x-part is more important so choose the next x in the middle to decrease dx as much as possible
                xNext = xNew[maxDisIndices[i]] + abs(xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) / 2
            #print('fit used on x = %f' % xNext)
            #print(sum(useFit))

        else:
            xNext = xNew[maxDisIndices[i]] + abs(xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) / 2
            #print('explorative norm used on x = %f' % xNext)
            #print(sum(useFit))
        #---------------------------------extra developer info#
        if self._DevMode:
            xWeight = (dx[maxDisIndices[i]] ** 2) / dis[maxDisIndices[i]] ** 2
            yWeight = 1 - xWeight
            self.normInfo.append([xWeight,yWeight,-1, useFit])
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext

    def calcXNextByFitAndDiff3(self,xNew, yNew, xPrev, yPrev, alpha = -9999):
        #dis with not using usefit
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        dy = np.diff(yNew / yNorm)
        #now calculate for each xPrev the distance between an interpolation of the previous scan with current x, and the previous scan itself.
        yIntp = np.interp(xNew, xPrev, yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)
        fitDis = abs(yPrev - yPrevIntp) / yNorm

        #calc second order diff
        diff2 = np.diff(dy / dx)
        diff2AV = np.zeros(len(xNew) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx
        #calc distance per interval exploratively
        xPrefFactor = 1 #value chosen 3, since q consist of 3 y-parts and 1 x-part
        exDis = xPrefFactor * abs(dx) + abs(dy) + abs(diff2AV)

        #calculate the gain for each point in xPrev !!!!!!!!! ASSUME min(xPrev) = min(xNew)  !!!!!
        ss = np.searchsorted(xNew,xPrev)
        diffs = xNew[ss]-xNew[ss-1]
        xPerc = (xNew[ss]-xPrev) / diffs
        exGainFactor = np.minimum(xPerc, 1-xPerc)
        gainXPrev = fitDis + exGainFactor * exDis[ss-1]

        #now we need to calculate the gain for x's taken in the middle of an interval
        #first create xVec
        xMid = (xNew[0:-1] + np.diff(xNew) / 2)
        hisMidDis = abs(np.interp(xMid,xNew,yIntp) - np.interp(xMid,xPrev,yPrev)) / yNorm

        gainXMid = hisMidDis + 0.5 * exDis #0.5 since all x'es are in the middle so have maximum xplorative gain

        #now get the unique x-vals
        xUnique, iX = np.unique(np.append(xPrev,xMid), return_index = True)
        gainUnique = np.append(gainXPrev,gainXMid)[iX]
        #find the maximum gain!
        maxGainIndices = np.argsort(gainUnique)
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        #check if minimum interval size is not exceeded
        #calc distance in x
        ss2 = np.searchsorted(xNew,xUnique)
        disRight = xNew[ss2] - xUnique
        disLeft = xUnique - xNew[ss2-1]
        mindis = np.minimum(disRight,disLeft)

        while(xFound == False):
            if mindis[maxGainIndices[i]] <= self.minimumInterval:
                i = i - 1
            else:
                xFound = True
        else:
            xNext = xUnique[maxGainIndices[i]]
        #---------------------------------extra developer info#
        if self._DevMode:
            k = maxGainIndices[i]
            gainDX = np.append(exGainFactor * xPrefFactor * dx[ss-1],0.5 * xPrefFactor *  dx)[iX]
            gainDY = np.append(exGainFactor * dy[ss-1],0.5* dy)[iX]
            gainDDY = np.append(exGainFactor * diff2AV[ss-1], 0.5 * diff2AV)[iX]
            gainHis = np.append(fitDis, hisMidDis)[iX]
            xWeight = (xUnique[maxGainIndices[i]] / xNorm) / gainUnique[maxGainIndices[i]]
            yWeight = 1 - xWeight
            self.normInfo.append([gainDX[k], gainDY[k], gainDDY[k], gainHis[k]] / gainUnique[k] )
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext

    def calcXNextByFitAndDiff31(self,xNew, yNew, xPrev, yPrev, alpha=-9999):
        #dis with not using usefit
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        dy = np.diff(yNew / yNorm)
        #now calculate for each xPrev the distance between an interpolation of the previous scan with current x, and the previous scan itself.
        yIntp = np.interp(xNew, xPrev, yPrev)
        yPrevIntp = np.interp(xPrev, xNew, yIntp)
        fitDis = abs(yPrev - yPrevIntp) / yNorm

        #calc second order diff
        diff2 = np.diff(dy / dx)
        diff2AV = np.zeros(len(xNew) -1)
        diff2AV[0] = diff2[0]
        diff2AV[-1] = -diff2[-1]
        diff2AV[1:-1] = (abs(diff2[0:-1]) + abs(diff2[1:])) / (2.0)
        diff2AV = diff2AV * dx
        #calc distance per interval exploratively
        exDis = np.maximum(abs(dx),abs(dy),abs(diff2AV))

        #calculate the gain for each point in xPrev !!!!!!!!! ASSUME min(xPrev) = min(xNew)  !!!!!
        ss = np.searchsorted(xNew,xPrev)
        diffs = xNew[ss]-xNew[ss-1]
        xPerc = (xNew[ss]-xPrev) / diffs
        exGainFactor = np.minimum(xPerc, 1-xPerc)
        gainXPrev = np.maximum(fitDis , exGainFactor * exDis[ss-1])
        #now we need to calculate the gain for x's taken in the middle of an interval
        #first create xVec
        xMid = (xNew[0:-1] + np.diff(xNew) / 2)
        hisMidDis = abs(np.interp(xMid,xNew,yIntp) - np.interp(xMid,xPrev,yPrev)) / yNorm

        gainXMid = np.maximum(hisMidDis, 0.5 * exDis) #0.5 since all x'es are in the middle so have maximum xplorative gain

        #now get the unique x-vals
        xUnique, iX = np.unique(np.append(xPrev,xMid), return_index = True)
        gainUnique = np.append(gainXPrev,gainXMid)[iX]
        #find the maximum gain!
        maxGainIndices = np.argsort(gainUnique)
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        #check if minimum interval size is not exceeded
        #calc distance in x
        ss2 = np.searchsorted(xNew,xUnique)
        disRight = xNew[ss2] - xUnique
        disLeft = xUnique - xNew[ss2-1]
        mindis = np.minimum(disRight,disLeft)

        while(xFound == False):
            if mindis[maxGainIndices[i]] <= self.minimumInterval:
                i = i - 1
            else:
                xFound = True
        else:
            xNext = xUnique[maxGainIndices[i]]
        #---------------------------------extra developer info#
        if self._DevMode:
            k = maxGainIndices[i]
            gainDX = np.append(exGainFactor * dx[ss-1],0.5 * dx)[iX]
            gainDY = np.append(exGainFactor * dy[ss-1],0.5* dy)[iX]
            gainDDY = np.append(exGainFactor * diff2AV[ss-1], 0.5 * diff2AV)[iX]
            gainHis = np.append(fitDis, hisMidDis)[iX]
            xWeight = (xUnique[maxGainIndices[i]] / xNorm) / gainUnique[maxGainIndices[i]]
            yWeight = 1 - xWeight
            self.normInfo.append([gainDX[k], gainDY[k], gainDDY[k], gainHis[k]] / gainUnique[k] )
            #print(self.normInfo[-1])
            #print(maxDisIndices[i])
            #print(dis)
        #-------------------------------------------------------------
        return xNext

    def calcXNextbyFitAndDiff4(self, xNew, yNew, xPrev, yPrev, alpha =-9999):
        #OLD HISTORY USED FOR REFERENCE
        xNorm = self.xMax - self.xMin
        yNorm = max(yNew) - min(yNew)
        dx = np.diff(xNew) / xNorm
        #now calculate for each interval the distance between previous and current scan
        yPrevIntp = np.interp(xPrev, xNew, yNew)
        fitDis = abs(yPrev - yPrevIntp)
        #now we have to determine the maximum distance per interval
        intervalIndices = np.searchsorted(xPrev, xNew)
        intervalList = np.split(fitDis, intervalIndices)[1:-1]
        intervalList = [(i if len(i) else [0]) for i in intervalList] #Add negative values for intervals which contain no xprev, so they will not be used in the fitnorm
        maxList = np.array(map(max,intervalList)) #max distance per interval
        maxFitIndices = np.array(map(np.argmax, intervalList))
        #calculate the average error per xNew value
        errors = abs(np.interp(xNew,xPrev,yPrev) - yNew)
        maxErrors = np.maximum(errors[0:-1],errors[1:])
        #create a boolean vector for which distance norm should be used
        useFit = maxErrors < maxList
        #calculate the distance vector
        dis = (useFit) * np.sqrt(dx ** 2 + (maxList / yNorm) **2) + self.qTypes[self.quality](xNew, yNew) * (1 - useFit)
        #print(useFit)
        #search for the next X to send out based on the dis vector
        maxDisIndices = np.argsort(dis)
        #check that xNext is not within the minimum distance specified
        xFound = False
        i = -1
        #interval = max([(self.xMax - self.xMin) / (len(xNew) * 4), (self.xMax - self.xMin) / self.nPrecision])
        #interval = max([(self.xMax - self.xMin) / (len(xNew) * 4), (self.xMax - self.xMin) / self.nPrecision])
        #check if minimum interval size is not exceeded
        while(xFound == False):
            if (xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) <= self.minimumInterval:
                i = i - 1
            else:
                xFound = True
        #fit is used so determine the new x coordinate in a smarter way
        if useFit[maxDisIndices[i]] == True:
            if((dx[maxDisIndices[i]] ** 2) / dis[maxDisIndices[i]] ** 2 < 0.5): #y-part is more important
                xNext = xPrev[intervalIndices[maxDisIndices[i]]+maxFitIndices[maxDisIndices[i]]]
            else: #x-part is more important so choose the next x in the middle to decrease dx as much as possible
                xNext = xNew[maxDisIndices[i]] + abs(xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) / 2
        #print('fit used on x = %f' % xNext)
        #print(sum(useFit))
        else:
            xNext = xNew[maxDisIndices[i]] + abs(xNew[maxDisIndices[i]+1] - xNew[maxDisIndices[i]]) / 2
        #print('explorative norm used on x = %f' % xNext)
        #print(sum(useFit))
        #---------------------------------extra developer info#
        if self._DevMode:
            xWeight = (dx[maxDisIndices[i]] ** 2) / dis[maxDisIndices[i]] ** 2
            yWeight = 1 - xWeight
            self.normInfo.append([xWeight,yWeight,-1, useFit[maxDisIndices[i]]])
        #print(self.normInfo[-1])
        #print(maxDisIndices[i])
        #print(dis)
        #-------------------------------------------------------------
        return xNext