'''
Created on 21 okt. 2014
measures a point(Gate/Bfield,Bias) and returns that
@author: Jaap
'''
from project.measurement.interpol import linInterpolate2D, linInterpolate1D
from project.measurement.flread import FileReader


def MeasureFromFile(x,y,file):
    '''simulates a measurement from a file'''
    #read the file
    f = FileReader(file);
    X,Y,Z = f.makeMesh();
    measurement = linInterpolate2D(x, y, X,Y,Z,f.x,f.y);
    return measurement;
    
def MeasureFromVecs(x, xVec, yVec):
    '''simulates a measurement from 2 vectors'''
    return linInterpolate1D(xVec,yVec,x) ;   