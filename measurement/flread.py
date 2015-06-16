'''
Created on 21 okt. 2014
Reads data from a file from specified columns
@author: Jaap
'''

from numpy import *
class FileReader:
    def __init__(self, fileIn):
        '''initializes class variables'''
        self.fileName = fileIn; 
        
    def readColumns(self, columns = "ALL",**keyargs):
        '''Opens the file and reads the data separated by kw 'separator' from it into a numpy array called numData, ignores '/n' and '#' '''
        if('separator' in keyargs):
            separator = keyargs['separator'];
        else:
            separator = '\t';
        with open(self.fileName,'r') as f:
            #loop through all lines except the comments and newlines
            data = [];
            for line in f:
                if((line[0] != '#') & (line[0] != '\n')):
                    dataLine = line.split(separator); # separate the columns in a list
                    floatDataLine = []; #create an empty float list
                    for dataString in dataLine:
                        floatDataLine.append(float(dataString)); #convert to floats
                    data.append(floatDataLine); # insert the floats in the main data list
            #Create a numpy array of the data
            self.numData = array(data);
            #return the requested columns of the data and all if no columns are specified
            if(columns == "ALL"):
                return copy(self.numData);
            return copy(self.numData[:,columns]);
    
    def makeMesh(self,xcol=1,ycol=0,zcol=6):
        '''Creates three meshes of x,y and z and returns them as X,Y,Z'''
        try:#check if numdata exists
            self.numData;
        except AttributeError:
            self.readColumns();
        else:
            pass;
        # get the unique voltage values
        y = unique(self.numData[:,ycol]);
        x = unique(self.numData[:,xcol]);
        z = self.numData[:,zcol];
        self.Z = zeros((y.size,x.size));
        #now get the corresponding z values in the same meshgrid
        j = 0;
        for i in arange(0,z.size,y.size):
            self.Z[:,j] = z[i:i+y.size];
            j = j+1;
        self.X,self.Y = meshgrid(x,y);
        self.x = x;
        self.y = y;
        return copy(self.X),copy(self.Y),copy(self.Z);

        