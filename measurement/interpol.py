'''
Created on 21 okt. 2014
Interpolation functions collection
@author: Jaap
'''
from numpy import *

def linInterpolate2D(xcd, ycd, X, Y, Z, xvals, yvals):
        '''interpolates the given Z mesh by using the coords(x,y) in the specified columns(columnx,columny)'''
        #find the the four values belonging to the grid around the point(x,y)
        x1 = xvals[xvals <= xcd][0]
        x2 = xvals[xvals >= xcd][0]
        y1 = yvals[yvals <= ycd][0]
        y2 = yvals[yvals >= ycd][0]
        
        #find the corresponding z values
        z11 = Z[where(yvals==y1), where(xvals == x1)]
        z12 = Z[where(yvals==y2), where(xvals == x1)]
        z21 = Z[where(yvals==y1), where(xvals == x2)]
        z22 = Z[where(yvals==y2), where(xvals == x2)]
        
        #now interpolate the values by using the 2D linear interpolation formula
        z = ((1 / ((y2 - y1) * (x1 - x2))) *  
             ((y2 - ycd) * (x2 - xcd) * z11 +
             (y2 - ycd) * (xcd - x1) * z21 +
             (ycd - y1) * (x2 - xcd) * z12 +
             (ycd - y1) * (xcd - x1) * z22))
        return z
    
def linInterpolate1D(xVec,yVec,xValues):
    '''Interpolates a value of the yvector assuming they both have the same amount of elements, returns the y values'''
    #create a list to hold the y values
    ylist = []
    for x in xValues:
        if(x < min(xVec)):
            x1 = min(xVec)
        else:
            x1 = xVec[xVec <= x][-1]
        if(x > max(xVec)):
            x2 = max(xVec)
        else:
            x2 = xVec[xVec >= x][0]
    
     #find the corresponding y values
        y1 = yVec[where(xVec == x1)][0]
        y2 = yVec[where(xVec == x2)][0]
    
    #now interpolate the values using 1D linear interpolation
        if(x2 == x1): #check if last value is asked
            ylist.append(yVec[-1])
        else:
            ylist.append(y1 + ((x-x1)/(x2-x1)) * (y2-y1))
    #now return a numpy array of the y values or just the value itself if there is only one item
    if(len(ylist) == 1):
        return ylist[0]
    return array(ylist)