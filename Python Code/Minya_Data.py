#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:21:25 2017

@author: JayBaek
"""

import siganalysis as sa
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy import signal
import math
from scipy.fftpack import *
import scipy
import pymysql

# for MergedFeatured_data
def load_data(path):
    Energy, Avg_HVel, Max_Acc, Mean_Acc, Std_Acc = np.loadtxt(path, delimiter=',', unpack=True)
    return Energy, Avg_HVel, Max_Acc, Mean_Acc, Std_Acc

path = '/Users/JayBaek/Desktop/Test.csv'

Energy, Avg_HVel, Max_Acc, Mean_Acc, Std_Acc  = load_data(path)

# for feature_array data


dataSet = []
for i in range (len(Energy)):
#    if i == 200:
        #break
    if i<179:
        dataSet.append([Energy[i], Avg_HVel[i], Max_Acc[i], Mean_Acc[i], Std_Acc[i], 'Walking'])
        continue
    if i<290:
        dataSet.append([Energy[i], Avg_HVel[i], Max_Acc[i], Mean_Acc[i], Std_Acc[i],'Running'])
        continue
    if i<331:
        dataSet.append([Energy[i], Avg_HVel[i], Max_Acc[i], Mean_Acc[i], Std_Acc[i],'Jumping'])
        continue
    else:
        dataSet.append([Energy[i], Avg_HVel[i], Max_Acc[i], Mean_Acc[i], Std_Acc[i],'badwalking'])
        continue


class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value):
   # Make a function that tells us if a row is in 
   # the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float):
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)

print divideset(dataSet, 0, 'running')
#print divideset(my_data, 2, 'yes')
#print divideset(my_data1, 0, 'running')
#print divideset(my_data1, 1, 11)

print divideset(dataSet, 1, 9)   

# Create counts of possible results (the last column of 
# each row is the result)
def uniquecounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r]=0
      results[r]+=1
   return results

#print (divideset(my_data1, 0, 'running')[0])
#print (divideset(my_data1, 0, 'running')[1])
#
#print (uniquecounts(divideset(my_data1, 0, 'running')[0]))
#print (uniquecounts(divideset(my_data1, 0, 'running')[1]))


# Probability that a randomly placed item will
# be in the wrong category
def giniimpurity(rows):
  total=len(rows)
  counts=uniquecounts(rows)
  imp=0
  for k1 in counts:
    p1=float(counts[k1])/total
    for k2 in counts:
      if k1==k2: continue
      p2=float(counts[k2])/total
      imp+=p1*p2
  return imp

# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

#set1, set2 = divideset(my_data1, 0, 'running')
#print entropy(set1)
#print entropy(set2)



def printtree(tree,indent=''):
   # Is this a leaf node?
   if tree.results!=None:
      print str(tree.results)
   else:
      # Print the criteria
      print str(tree.col)+':'+str(tree.value)+'? '

      # Print the branches
      print indent+'T->',
      printtree(tree.tb,indent+'  ')
      print indent+'F->',
      printtree(tree.fb,indent+'  ')


def getwidth(tree):
  if tree.tb==None and tree.fb==None: return 1
  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
  if tree.tb==None and tree.fb==None: return 0
  return max(getdepth(tree.tb),getdepth(tree.fb))+1


from PIL import Image,ImageDraw

def drawtree(tree,jpeg='tree.jpg'):
  w=getwidth(tree)*100
  h=getdepth(tree)*100+120

  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  drawnode(draw,tree,w/2,20)
  img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
  if tree.results==None:
    # Get the width of each branch
    w1=getwidth(tree.fb)*100
    w2=getwidth(tree.tb)*100

    # Determine the total space required by this node
    left=x-(w1+w2)/2
    right=x+(w1+w2)/2

    # Draw the condition string
    if tree.col == 0:
        draw.text((x-20,y-10),'energy'+':'+str(tree.value),(0,0,0))
    if tree.col == 1:
        draw.text((x-20,y-10),'Avg_HVel'+':'+str(tree.value),(0,0,0))
    if tree.col == 2:
        draw.text((x-20,y-10),'Max_Acc'+':'+str(tree.value),(0,0,0))
    if tree.col == 3:
        draw.text((x-20,y-10),'Mean_Acc'+':'+str(tree.value),(0,0,0))
    if tree.col == 4:
        draw.text((x-20,y-10),'Std_Acc'+':'+str(tree.value),(0,0,0))


    # Draw links to the branches
    draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
    draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))
    
    # Draw the branch nodes
    drawnode(draw,tree.fb,left+w1/2,y+100)
    drawnode(draw,tree.tb,right-w2/2,y+100)
  else:
    txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
    draw.text((x-20,y),txt,(0,0,0))


def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)

def prune(tree,mingain):
  # If the branches aren't leaves, then prune them
  if tree.tb.results==None:
    prune(tree.tb,mingain)
  if tree.fb.results==None:
    prune(tree.fb,mingain)
    
  # If both the subbranches are now leaves, see if they
  # should merged
  if tree.tb.results!=None and tree.fb.results!=None:
    # Build a combined dataset
    tb,fb=[],[]
    for v,c in tree.tb.results.items():
      tb+=[[v]]*c
    for v,c in tree.fb.results.items():
      fb+=[[v]]*c
    
    # Test the reduction in entropy
    delta=entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)

    if delta<mingain:
      # Merge the branches
      tree.tb,tree.fb=None,None
      tree.results=uniquecounts(tb+fb)

def mdclassify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    if v==None:
      tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
      tcount=sum(tr.values())
      fcount=sum(fr.values())
      tw=float(tcount)/(tcount+fcount)
      fw=float(fcount)/(tcount+fcount)
      result={}
      for k,v in tr.items(): result[k]=v*tw
      for k,v in fr.items(): result[k]=v*fw      
      return result
    else:
      if isinstance(v,int) or isinstance(v,float):
        if v>=tree.value: branch=tree.tb
        else: branch=tree.fb
      else:
        if v==tree.value: branch=tree.tb
        else: branch=tree.fb
      return mdclassify(observation,branch)

def variance(rows):
  if len(rows)==0: return 0
  data=[float(row[len(row)-1]) for row in rows]
  mean=sum(data)/len(data)
  variance=sum([(d-mean)**2 for d in data])/len(data)
  return variance

def buildtree(rows,scoref=entropy):
  if len(rows)==0: return decisionnode()
  current_score=scoref(rows)

  # Set up some variables to track the best criteria
  best_gain=0.0
  best_criteria=None
  best_sets=None
  
  column_count=len(rows[0])-1
  for col in range(0,column_count):
    # Generate the list of different values in
    # this column
    column_values={}
    for row in rows:
       column_values[row[col]]=1
    # Now try dividing the rows up for each value
    # in this column
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value)
      
      # Information gain
      p=float(len(set1))/len(rows)
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
      if gain>best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
  # Create the sub branches   
  if best_gain>0:
    trueBranch=buildtree(best_sets[0])
    falseBranch=buildtree(best_sets[1])
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=uniquecounts(rows))
    
tree = buildtree(dataSet)
print ""
print tree.col
print tree.value
print tree.results
print ""
print tree.tb.col
print tree.tb.value
print tree.tb.results
print ""
print tree.fb.col
print tree.fb.value
print tree.fb.results

print ""
printtree(tree)


drawtree(tree, jpeg='treeview.jpg')
print "\n\nwhat is this?"

#print (classify([30], tree))
result = classify([243.62657580000001, 1.8233577910000001, 1.669363905, 0.69934529400000001, 0.34681884000000002], tree)

print result
for k,v in result.items():
    result = k
print ('Result of test data: %s' %result)





# Animation
import pygame

def animation(string):
    pygame.init()
    pygame.display.set_caption("Animation")
    white = (255,255,255)
    clock = pygame.time.Clock()
    
    FRAME_TIME = 50 #50ms = 20fps
    next_frame_time = pygame.time.get_ticks() + FRAME_TIME


    # Walking Images
    w1 = pygame.image.load("/Users/JayBaek/Desktop/images/w1.png")
    w2 = pygame.image.load("/Users/JayBaek/Desktop/images/w2.png")
    w3 = pygame.image.load("/Users/JayBaek/Desktop/images/w3.png")
    w4 = pygame.image.load("/Users/JayBaek/Desktop/images/w4.png")
    w5 = pygame.image.load("/Users/JayBaek/Desktop/images/w5.png")
    w6 = pygame.image.load("/Users/JayBaek/Desktop/images/w6.png")
    w7 = pygame.image.load("/Users/JayBaek/Desktop/images/w7.png")
    w8 = pygame.image.load("/Users/JayBaek/Desktop/images/w8.png")
    w9 = pygame.image.load("/Users/JayBaek/Desktop/images/w9.png")
    w10 = pygame.image.load("/Users/JayBaek/Desktop/images/w10.png")

    # Running Images
    r1 = pygame.image.load("/Users/JayBaek/Desktop/images/r1.png")
    r2 = pygame.image.load("/Users/JayBaek/Desktop/images/r2.png")
    r3 = pygame.image.load("/Users/JayBaek/Desktop/images/r3.png")
    r4 = pygame.image.load("/Users/JayBaek/Desktop/images/r4.png")
    r5 = pygame.image.load("/Users/JayBaek/Desktop/images/r5.png")
    r6 = pygame.image.load("/Users/JayBaek/Desktop/images/r6.png")
    r7 = pygame.image.load("/Users/JayBaek/Desktop/images/r7.png")
    r8 = pygame.image.load("/Users/JayBaek/Desktop/images/r8.png")
    r9 = pygame.image.load("/Users/JayBaek/Desktop/images/r9.png")
    r10 = pygame.image.load("/Users/JayBaek/Desktop/images/r10.png")
    r11 = pygame.image.load("/Users/JayBaek/Desktop/images/r11.png")



    CurrentImage = 1

    gameLoop=True
    if string == "walking":
        window = pygame.display.set_mode((800,600))
        while gameLoop: 
            for event in pygame.event.get(): 
                if (event.type==pygame.QUIT): 
                    gameLoop=False 
                window.fill(white) 
                if (CurrentImage==1): 
                    window.blit(w1, (10,10)) 
                if (CurrentImage==2): 
                    window.blit(w2, (50,10))
                if (CurrentImage==3): 
                    window.blit(w3, (100,10))
                if (CurrentImage==4): 
                    window.blit(w4, (150,10))
                if (CurrentImage==5): 
                    window.blit(w5, (200,10))
                if (CurrentImage==6): 
                    window.blit(w6, (250,10))
                if (CurrentImage==7): 
                    window.blit(w7, (300,10))
                if (CurrentImage==8): 
                    window.blit(w8, (350,10))
                if (CurrentImage==9): 
                    window.blit(w9, (400,10))
                if (CurrentImage==10): 
                    window.blit(w10, (450,10))
                if (CurrentImage==10): 
                    CurrentImage=1 
                else: CurrentImage+=1; 
            pygame.display.flip() 
            clock.tick(5)


#    gameLoop = True
    if string == "running":
        window = pygame.display.set_mode((750,250))
        while gameLoop:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    gameLoop=False
                window.fill(white)
                if (event.type == pygame.NOEVENT):
                    break
                now = pygame.time.get_ticks()
                if (CurrentImage==1): 
                    window.blit(r1, (10,10)) 
                if (CurrentImage==2): 
                    window.blit(r2, (50,10))
                if (CurrentImage==3): 
                    window.blit(r3, (100,10))
                if (CurrentImage==4): 
                    window.blit(r4, (150,10))
                if (CurrentImage==5): 
                    window.blit(r5, (200,10))
                if (CurrentImage==6): 
                    window.blit(r6, (250,10))
                if (CurrentImage==7): 
                    window.blit(r7, (300,10))
                if (CurrentImage==8): 
                    window.blit(r8, (350,10))
                if (CurrentImage==9): 
                    window.blit(r9, (400,10))
                if (CurrentImage==10): 
                    window.blit(r10, (450,10))
                if (CurrentImage==11): 
                    window.blit(r11, (500,10))
                if (CurrentImage==11): 
                    CurrentImage=1 
                else: CurrentImage+=1;
                
            pygame.display.flip() 
            if now < next_frame_time:
                    pygame.time.wait(next_frame_time - now)
            next_frame_time += FRAME_TIME
            clock.tick(7)
    
    pygame.quit()

    
#animation(result) 

