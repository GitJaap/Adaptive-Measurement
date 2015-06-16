# Adaptive-Measurement-Algorithm

Easy to use O(N) adaptive sampling algorithm, measures more points in 'interesting' intervals specified by the choice of a loss function. Works on a point by point basis within a specified x-range.
The user has to perform the following measurement loop: ask for a new coordinate with getNewX(), then insert the measured y-value back in to the algorithm, repeat...
Can also be used in a 2D scan on a line by line basis where previously measured lines can be used as information for the current line by calling getNewXUsingPreviousData().

Can be installed as a toolbox using setup.py

Released under the GPL license by J.J Wesdorp
