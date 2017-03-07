#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 22:21:55 2017

@author: JayBaek
"""
#problem : how to combine with data and energy

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

import pylab
from mpl_toolkits.mplot3d import Axes3D

# Load data from path
def load_data(path):
    time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z, Mag_x, Mag_y, Mag_z = np.loadtxt(path, delimiter=',', unpack=True)
    return time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z

path = '/Users/JayBaek/Documents/Github/EE180D/data/running_test1.csv'
time_running, Gyro_x_running, Gyro_y_running, Gyro_z_running, Acc_x_running, Acc_y_running, Acc_z_running = load_data(path)

path1 = '/Users/JayBaek/Documents/Github/EE180D/data/running_test2.csv'
time_running1, Gyro_x_running1, Gyro_y_running1, Gyro_z_running1, Acc_x_running1, Acc_y_running1, Acc_z_running1 = load_data(path1)

path2 = '/Users/JayBaek/Documents/Github/EE180D/data/running_test3.csv'
time_running2, Gyro_x_running2, Gyro_y_running2, Gyro_z_running2, Acc_x_running2, Acc_y_running2, Acc_z_running2 = load_data(path2)

path3 = '/Users/JayBaek/Documents/Github/EE180D/data/running_test4.csv'
time_running3, Gyro_x_running3, Gyro_y_running3, Gyro_z_running3, Acc_x_running3, Acc_y_running3, Acc_z_running3 = load_data(path3)

path4 = '/Users/JayBaek/Documents/Github/EE180D/data/walking_test1.csv'
time_walking1, Gyro_x_walking1, Gyro_y_walking1, Gyro_z_walking1, Acc_x_walking1, Acc_y_walking1, Acc_z_walking1 = load_data(path4)

path5 = '/Users/JayBaek/Documents/Github/EE180D/data/walking_test2.csv'
time_walking2, Gyro_x_walking2, Gyro_y_walking2, Gyro_z_walking2, Acc_x_walking2, Acc_y_walking2, Acc_z_walking2 = load_data(path5)

path6 = '/Users/JayBaek/Documents/Github/EE180D/data/walking_test3.csv'
time_walking3, Gyro_x_walking3, Gyro_y_walking3, Gyro_z_walking3, Acc_x_walking3, Acc_y_walking3, Acc_z_walking3 = load_data(path6)

path7 = '/Users/JayBaek/Documents/Github/EE180D/data/sample_foot3.csv'
time_test, Gyro_x_test, Gyro_y_test, Gyro_z_test, Acc_x_test, Acc_y_test, Acc_z_test= load_data(path7)

# Conversion class for mean, max, min, std, and energy value
class conversion():
    def __init__(self, time, x, y, z, x1, y1, z1):
        self.time = time
        self.Gyro_x = x
        self.Gyro_y = y
        self.Gyro_z = z
        self.Acc_x  = x1
        self.Acc_y = y1
        self.Acc_z = z1
    
    def magnitude(self, string):
        if string == "Acc":
            m = np.sqrt(self.Acc_x*self.Acc_x + self.Acc_y*self.Acc_y + self.Acc_z*self.Acc_z)
            return m
        if string == "Gyro":
            m = np.sqrt(self.Gyro_x*self.Gyro_x + self.Gyro_y*self.Gyro_y + self.Gyro_z*self.Gyro_z)
            return m
            
    def feature(self, string, magnitude):
        fftsize = 256
        overlap = 4
        temp = []
        size = []
        hop = fftsize / overlap
        print 'size: %s' %(len(magnitude)-fftsize)
        for i in range(0, len(magnitude)-fftsize, hop):
            if string == "mean":
                temp.append(np.mean(magnitude[i:i+fftsize]))
            if string == "max":
                temp.append(np.max(magnitude[i:i+fftsize]))
            if string == "min":
                temp.append(np.min(magnitude[i:i+fftsize]))
            if string == "std":
                temp.append(np.std(magnitude[i:i+fftsize]))
            if string == "energy":
                temp.append(np.sum(np.power(magnitude[i:i+fftsize],2)))
        for i in range (len(temp)):
            size.append(i)
        return temp, size
        
    def get_value(self, string, string1):
        if string == "Acc":
            m = self.magnitude(string)
            return self.feature(string1, m)
        if string == "Gyro":
            m = self.magnitude(string)
            return self.feature(string1, m)
            
#class plot():
#    def __init__(self, column, row):
#        self.column = column
#        self.row = row
#    def draw(self, x, y, color):
#        plt.figure(figsize=(8,10))
#        plt.tight_layout()
#        plt.plot(x, y, color)

            
a = conversion(time_running, Gyro_x_running, Gyro_y_running, Gyro_z_running, Acc_x_running, Acc_y_running, Acc_z_running)
energy_ruuning, size_running = a.get_value("Acc", "energy")

#m1, size1 = a.get_value("Acc", "min")

#plt.figure(figsize=(8,10))
#plt.subplot(511)
#plt.tight_layout() #for bigger layout
#plt.subplots_adjust(hspace = 0.45)
#plt.plot(size, m, 'r')
#plt.plot(size1, m1, 'b')


#conversion to velocity
def get_velocity(Acc_data):
    vel = []
    for i in range (len(Acc_data)-1):
        if i == 0:
            vel.append(0)
        if (Acc_data[i] < -0.015):
            vel.append(0)
        else: 
            vel.append(vel[i] + Acc_data[i])
    return vel
    
#get velocity from acceleration
vel_running = get_velocity(Acc_x_running)
vel_running1 = get_velocity(Acc_x_running1)
vel_running2 = get_velocity(Acc_x_running2)
vel_running3 = get_velocity(Acc_x_running3) 
vel_walking1 = get_velocity(Acc_x_walking1)
vel_walking2 = get_velocity(Acc_x_walking2)
vel_walking3 = get_velocity(Acc_x_walking3)

vel_test = get_velocity(Acc_x_test/4)

def highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#high pass filter
fps= 30
cutoff = 0.08
filtered_running = highpass_filter(vel_running, cutoff, fps)
filtered_running1 = highpass_filter(vel_running1, cutoff, fps)
filtered_running2 = highpass_filter(vel_running2, cutoff, fps)
filtered_running3 = highpass_filter(vel_running3, cutoff, fps)
filtered_walking1 = highpass_filter(vel_walking1, cutoff, fps)
filtered_walking2 = highpass_filter(vel_walking2, cutoff, fps)
filtered_walking3 = highpass_filter(vel_walking3, cutoff, fps)
filtered_test = highpass_filter(vel_test, cutoff, fps)



def get_peak(data):
    temp = data
    cal = 0
    new = []
    count = 0
    peak = 0
    for i in range (len(data)-1):
        if (peak == 100): #for test
            break
        if (temp[i] <= data[i+1]):
            cal += temp[i]
            count += 1
            
        if (temp[i] - data[i+1]) > 1: # 다음꺼가 전꺼랑 비슷한 값일때 0되는걸 피할려고
            new.append(cal/count)
            cal = 0
            count = 0
            peak += 1
    print ('how many peak?: ', peak)
    return new
            
def get_avgVelocity(data):
    sample=[]
    size = 0
    for i in range (len(data)):
        if (data[i]< 0):
            continue
        if (data[i] > 4):
            size += 1
            if (len(sample)==0):
                sample.append(data[i])
            else:
                sample.append(data[i])
    
    result = get_peak(sample)
    return result
                
avg_vel_running = get_avgVelocity(filtered_running)
avg_vel_running1 = get_avgVelocity(filtered_running1)  
avg_vel_running2 = get_avgVelocity(filtered_running2) 
avg_vel_running3 = get_avgVelocity(filtered_running3)      
avg_vel_walking1 = get_avgVelocity(filtered_walking1) 
avg_vel_walking2 = get_avgVelocity(filtered_walking2)
avg_vel_walking3 = get_avgVelocity(filtered_walking3)     
avg_vel_test = get_avgVelocity(filtered_test)      

def get_separate_sequence(data):
    sample = []
    if (len(data)>= 17):
        sample1 = []
    if (len(data)>=34):
        print ("here")
        sample2 = []
    for i in range (len(data)):
        if (i>=34):
            if (i==51):
                break;
            sample2.append(data[i])
            continue
        if (i>=17 and i<34):
            if (i>=17 and len(data) < 34):
                break
            sample1.append(data[i])
            continue
        if (i<17):
            sample.append(data[i])
        else:
            break
    if (len(data) >= 51):
        return sample, sample1, sample2
    if (len(data)>34 and len(data) < 51):
        return sample, sample1
    return sample
#size of a must be 17

avg_vel_running = get_separate_sequence(avg_vel_running)
avg_vel_running1 = get_separate_sequence(avg_vel_running1)
avg_vel_running2 = get_separate_sequence(avg_vel_running2)
avg_vel_running3 = get_separate_sequence(avg_vel_running3)
avg_vel_walking1 = get_separate_sequence(avg_vel_walking1)
avg_vel_walking2 = get_separate_sequence(avg_vel_walking2)
avg_vel_walking3 = get_separate_sequence(avg_vel_walking3)

avg_vel_test1, avg_vel_test2, avg_vel_test3 = get_separate_sequence(avg_vel_test)

##plot
plt.figure(figsize=(8,10))
plt.subplot(511)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
plt.plot(time_running, Acc_x_running, 'r')
plt.plot(time_walking1, Acc_x_walking1, 'b')
plt.xlim([1000, 2000])
plt.title('Horizontal Acceleration')
plt.xlabel('Sample')

plt.figure(figsize=(8,10))
plt.subplot(512)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
#plt.plot(time_running, vel_running, 'r')
#plt.plot(time_walking1, vel_walking1, 'b')
plt.plot(time_test, vel_test, 'g')
#plt.xlim([1000, 2000])
plt.title('Filtered Horizontal Velocity')
plt.xlabel('Sample')

plt.figure(figsize=(8,10))
plt.subplot(513)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
plt.plot(time_test, filtered_test, 'g')
#plt.plot(time2, filtered_x2, 'b')
#plt.plot(time5, vel_x5, 'g')
#plt.xlim([1000, 2000])
plt.title('High-pass Filtered Horizontal Velocity')
plt.xlabel('Sample')


plt.figure(figsize=(8,10))
plt.subplot(514)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
#plt.plot(avg_vel, avg_vel1, 'ro')
plt.plot(avg_vel_running1, avg_vel_running, 'ro')
plt.plot(avg_vel_running2, avg_vel_running, 'ro')
plt.plot(avg_vel_running3, avg_vel_running, 'ro')
plt.plot(avg_vel_walking1, avg_vel_running, 'bo')
plt.plot(avg_vel_walking2, avg_vel_running, 'bo')
plt.plot(avg_vel_walking3, avg_vel_running, 'bo')
#for test data
plt.plot(avg_vel_test1, avg_vel_running, 'go')
plt.plot(avg_vel_test2, avg_vel_running, 'go')
plt.plot(avg_vel_test3, avg_vel_running, 'go')



#plt.plot(time, vel_z, color= 'green')
#plt.plot(time3, m3, color= 'cyan')

plt.title('Average Velocity of a Period')
plt.xlabel('red=running, blue=walking, green=test')
#plt.xlim([1100, 2000])
#plt.ylim([0, 40])

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def data_save(data, path):
#    path_check = '/Users/JayBaek/Desktop/' + path
#    print path_check
#    remove_file(path_check)
#    remove_file(path)
    # for running
#    if (identifier == 0):
#        for i in range (len(data)):
#            saveFile = open(path, 'a')
#            saveFile.write(str(data[i]))
#            saveFile.write('\n')
#        saveFile.close()
#    if (identifier == 1):
#        for i in range (len(data)):
#            saveFile = open(path, 'a')
#            saveFile.write(str(data[i]))
#            saveFile.write('\n')
#        saveFile.close()
    for i in range (len(data)):
        saveFile = open(path, 'a')
        saveFile.write(str(data[i]))
        saveFile.write('\n')
    saveFile.close()
    
#identifier (running = 0, walking = 1)
#need to fix below
data_save(avg_vel_running, 'running.csv')
data_save(avg_vel_running1, 'running1.csv')
data_save(avg_vel_running2, 'running2.csv')
data_save(avg_vel_running3, 'running3.csv')
data_save(avg_vel_walking1, 'walking1.csv')
data_save(avg_vel_walking2, 'walking2.csv')
data_save(avg_vel_walking3, 'walking3.csv')

