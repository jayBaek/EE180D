#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:14:24 2017

@author: JayBaek
"""
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,2.0],[1, 4],[4,1],[4,2]])
    labels = ['A','A','B','B']
    return group, labels

    
def classify0(inX, dataSet, labels, k):
    #calculate the distance betw inX and the current point
    sortedDistIndices = calcDistance(inX, dataSet, labels, k)
    #take k items with lowest distances to inX and find the majority class among k items
    sortedClassCount = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
    
    #sortedClassCount is now[('Action', 2)], therefore return Action
    return sortedClassCount[0][0] #return Action
    
def calcDistance(inX, dataSet, labels, k):
    #shape is a tuple that gives dimensions of the array
    #shape[0] returns the number of rows, here will return 4
    dataSetSize = dataSet.shape[0] #dataSetSize = 4 number of data sets
    
    print(tile(inX, (dataSetSize, 1)) - dataSet)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2
    
    sqDistances = sqDiffMat.sum(axis=1)
    
    distances = sqDistances ** 0.5
    
    sortedDistIndices = distances.argsort()
    return sortedDistIndices
    



group, labels = createDataSet()
result = classify0([2,3], group, labels, 3)
print result

