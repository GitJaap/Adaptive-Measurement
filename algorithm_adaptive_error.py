__author__ = 'Jaap'
'''
Experimental extension of the adaptive algorithm described below. This algorithm does not output coordinates to measure, but also reduces noise when appropriate by choosing the option with the highest gain in reducing the loss per interval.
Easy to use O(N) adaptive sampling algorithm, measures more points in 'interesting' intervals specified by the choice of a loss function. Works on a point by point basis within a specified x-range.
The user has to perform the following measurement loop: ask for a new coordinate with getNewX(), then insert the measured y-value back in to the algorithm, repeat...

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
class AdaptiveErrorAlgorithm:

    def __init__(self, xMin, xMax, e0, t0, printChoice = False, gamma = 10):
        self.xMin =xMin
        self.xMax = xMax
        self.e0 = e0
        self.t0 = t0
        #create lists for holding the current x, y, error, and measured time per point data (the processed raw data)
        self.xCur = []
        self.yCur = []
        self.eCur = []
        self.tCur = []
        self.printChoice = printChoice
        self.qOld = []
        self.gamma  = gamma


    def getNewX(self):
        choice = 2
        if(len(self.xCur) != len(self.yCur) | (len(self.xCur) != len(self.eCur))):
            print('Yvalues not set properly: please append an y value after each new X request')
        else:
            if len(self.xCur) == 0:
                xNext = self.xMin
                tNext = self.t0
            elif len(self.xCur) == 1:
                xNext = self.xMax
                tNext = self.t0
            elif len(self.xCur) == 2:
                xNext = self.xMin + (0.5) * (self.xMax - self.xMin)
                tNext = self.t0
            else:
                #factor = np.sqrt(self.e0)
                #create numpy arrays of the point list
                xCurA = np.array(self.xCur)
                yCurA = np.array(self.yCur)
                eCurA = np.array(self.eCur)
                tCurA = np.array(self.tCur)
                sortIndexes = np.argsort(xCurA)
                yCurA = yCurA[sortIndexes]
                eCurA = eCurA[sortIndexes]# * factor
                tCurA = tCurA[sortIndexes]
                xCurA.sort()
                #calculate the distances of each interval
                dis = self.dDiff(xCurA, yCurA, eCurA)
                iMax = np.argmax(dis)

                #now we have to chose between either decreasing the error on yiMax or yiMax+1 or add another point
                #calc the norm for all three
                disY1 = self.dDiff(xCurA, yCurA, eCurA, [xCurA[iMax:iMax + 2], yCurA[iMax:iMax + 2], np.array([eCurA[iMax] / 2, eCurA[iMax + 1]])])
                disY2 = self.dDiff(xCurA, yCurA, eCurA, [xCurA[iMax:iMax + 2], yCurA[iMax:iMax + 2], np.array([eCurA[iMax], eCurA[iMax + 1] / 2])])
                #calculate the distance for a point in the middle of the connecting line
                disX = self.dDiff(xCurA, yCurA, eCurA, [xCurA[iMax:iMax+2] / 2, yCurA[iMax:iMax + 2] / 2, eCurA[iMax:iMax+2]])
                #disX = self.dDiff(xCurA, yCurA, eCurA, [xCurA[iMax:iMax+2] / 2, yCurA[iMax:iMax + 2] / 2, np.array([self.e0,self.e0])])
                #now calculate the biggest Loss defined by OldDis-newDis / Cost in millis
                #print(np.array([(dis[iMax] - disY1) / self.calcTime(eCurA[iMax] / 2),
                                     #        (dis[iMax] - disY2) / self.calcTime(eCurA[iMax + 1] / 2),
                                      #       (dis[iMax] - disX) / self.calcTime(self.e0)]))
                                             #(dis[iMax] - disX) / self.calcTime((eCurA[iMax] + eCurA[iMax + 1]) / 2)]))
                choice = np.argmax(np.array([(dis[iMax] - disY1) / self.calcTime(eCurA[iMax] / 2),
                                             (dis[iMax] - disY2) / self.calcTime(eCurA[iMax + 1] / 2),
                                             #(dis[iMax] - disX) / self.calcTime(self.e0)]))
                                             (dis[iMax] - disX) / self.calcTime((eCurA[iMax] + eCurA[iMax + 1]) / 2)]))
                eCurA = eCurA#/ factor
                if(choice == 0):
                    xNext = xCurA[iMax]
                    tNext = self.calcTime(eCurA[iMax] / 2)#return the x coordinate and the amount of time to measure it
                elif(choice == 1):
                    xNext = xCurA[iMax + 1]
                    tNext = self.calcTime(eCurA[iMax + 1] / 2)
                elif(choice == 2):
                    xNext = xCurA[iMax] + 0.5 * (xCurA[iMax + 1] - xCurA[iMax])
                    tNext = self.calcTime((eCurA[iMax] + eCurA[iMax + 1]) / 2)
                    #tNext = self.t0
                if(self.printChoice == True):
                    print("choice: %d, avgerror: %.3f" %(choice, np.average(eCurA)))
            #now check if the x has already been measured
            if(choice == 1) | (choice == 0):
                index = self.xCur.index(xNext)
                del(self.xCur[index])
                del(self.yCur[index])
                del(self.eCur[index])
                del(self.tCur[index])
            self.xCur.append(xNext)
            self.tCur.append(tNext)
            return xNext, tNext
    def calcTime(self, e):
        return ((self.e0 / e) ** 2) * self.t0




    def dDiff(self, x, y, e, vals = []):
        if(len(vals) == 0):
            return np.sqrt((np.diff(x) / (np.max(x) - np.min(x))) ** 2 +
                       (np.diff(y) / (np.max(y) - np.min(y))) ** 2 +
                       (self.gamma*(e[:-1] + e[1:]) / (2.0*(np.max([np.max(y)-np.min(y), np.max(e)])))) ** 2)
        else:#vals has been given, so it will calculate the values of the norm only for given values
            return np.sqrt((np.diff(vals[0]) / (np.max(x) - np.min(x))) ** 2 +
                       (np.diff(vals[1]) / (np.max(y) - np.min(y))) ** 2 +
                       (self.gamma*(vals[2][:-1] + vals[2][1:]) / (2.0*(np.max([np.max(y)-np.min(y), np.max(e)])))) ** 2)

    def sort(self):
        '''sorts the xCur and yCur list and puts them in self.xSort and self.ySort'''
        xCurA = np.array(self.xCur)
        yCurA = np.array(self.yCur)
        self.ySort = yCurA[np.argsort(xCurA)]
        self.xSort = xCurA[np.argsort(xCurA)]
        self.eSort = np.array(self.eCur)[np.argsort(xCurA)]

    def plotValues1D(self, mark = 'or', error = True):
        self.sort()
        plt.errorbar(self.xSort,self.ySort, yerr=self.eSort*2, fmt=mark)#plot 95% confidence interval in case of gaussian noise

#----------------------------------------------------------------------------------------------------------------
class rawATSMeasurement:
    '''class used for actual ATS qclab integrations'''
    def __init__(self):
        pass #TODO create this class using qclab ATS functions

