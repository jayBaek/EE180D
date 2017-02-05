#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:39:37 2017

@author: JayBaek
"""

import siganalysis as sa
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import math
from scipy.fftpack import *
import scipy
from mpl_toolkits.mplot3d import Axes3D
#loading running data 1
path = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/running_test1.csv'
time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z, Mag_x, Mag_y, Mag_z = np.loadtxt(path, delimiter = ',', unpack = True)

path = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/walking_test1.csv'
time1, Gyro_x1, Gyro_y1, Gyro_z1, Acc_x1, Acc_y1, Acc_z1, Mag_x1, Mag_y1, Mag_z1 = np.loadtxt(path, delimiter = ',', unpack = True)

#From Accelerometer
mpre = Acc_x*Acc_x + Acc_y*Acc_y + Acc_z*Acc_z
m = np.sqrt(mpre)

mpre1 = Acc_x1*Acc_x1 + Acc_y1*Acc_y1 + Acc_z1*Acc_z1
m1 = np.sqrt(mpre1)


#From Gyroscope
mpre_gyro = Gyro_x*Gyro_x + Gyro_y*Gyro_y + Gyro_z*Gyro_z
m_gyro = np.sqrt(mpre_gyro)

mpre_gyro1 = Gyro_x1*Gyro_x1 + Gyro_y1*Gyro_y1 + Gyro_z1*Gyro_z1
m_gyro1 = np.sqrt(mpre_gyro1)

#From Magnetometer

mpre_mag = Mag_x*Mag_x + Mag_y*Mag_y + Mag_z*Mag_z
m_mag = np.sqrt(mpre_mag)


#print(m)

#for Accelration 

plt.figure(figsize=(8,10))
plt.subplot(511)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)

plt.plot(time, m, color='blue')
plt.title('Magnitude of Acc for back sensor')
plt.xlabel('time')
#plt.xlim([600, 1000])


#for Zyro 
'''
plt.figure(figsize=(8,10))
plt.subplot(511)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
plt.plot(time, m_gyro, color='blue')
plt.plot(time1, m_gyro1, color= 'red')
plt.plot(time2, m_gyro2, color= 'green')
plt.plot(time3, m_gyro3, color= 'cyan')

plt.title('Magnitude of Zyro with 4 running data samples')
plt.xlabel('time')
plt.xlim([600, 1000])
'''

#for Magnetometer
'''
plt.figure(figsize=(8,10))
plt.subplot(511)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)
plt.plot(time, m_mag, color='blue') #blue menas running
plt.plot(time1, m_mag1, color= 'red') #red menas walking
plt.title('Magnitude of Magnetometer')
plt.xlabel('time')
plt.xlim([0, 3000])
'''
def conversion(m):
    mean_value = []
    max_value = []
    min_value = []
    std_value = []
    energy_value = []
    for i in range (3000):
#        print(m_value[i])
        mean_value.append(np.mean(m[i:i+3000]))
        max_value.append(np.max(m[i]))
        min_value.append(np.min(m[i]))
        std_value.append(np.std(m[i]))
        energy_value.append(np.sum(np.power((m[i]), 2)))

    return mean_value, max_value, min_value, std_value, energy_value
    

mean_value, max_value, min_value, std_value, energy_value = conversion(m)
mean_value1, max_value1, min_value1, std_value1, energy_value1 = conversion(m1)

plt.subplot(512, projection='3d')
fig = plt.figure(figsize=(7,5))
ax = Axes3D(fig)
line1 = ax.plot(energy_value, energy_value1,energy_value1 , 'ok', color='red')

ax.view_init(azim=220, elev=22)



#
#def feature(x, fftsize=256, overlap=4):
#    meanamp = []
#    maxamp = []
#    minamp = []
#    stdamp = []
#    energyamp = []
#
#    hop = fftsize / overlap
#    for i in range(0, len(x)-fftsize, hop):
#        #print(i)
#        #print(np.mean(x[i:i+fftsize]))
#        meanamp.append(np.array(np.mean(x[i:i+fftsize])))
#        maxamp.append(np.array(np.max(x[i:i+fftsize])))
#        minamp.append(np.array(np.min(x[i:i+fftsize])))
#        stdamp.append(np.array(np.std(x[i:i+fftsize])))
#        energyamp.append(np.array(np.sum(np.power(x[i:i+fftsize],2))))
#        
#    return meanamp ,maxamp ,minamp,stdamp,energyamp
#
# 
#valmean, valmax, valmin, valstd, valenergy = feature(m)
#'''
#valmean1, valmax1, valmin1, valstd1, valenergy1 = feature(m1)
#valmean2, valmax2, valmin2, valstd2, valenergy2 = feature(m2)
#valmean3, valmax3, valmin3, valstd3, valenergy3 = feature(m3)
#'''
#
#valmean_gyro, valmax_gyro, valmin_gyro, valstd_gyro, valenergy_gyro = feature(m_gyro)
#'''
#valmean_gyro1, valmax_gyro1, valmin_gyro1, valstd_gyro1, valenergy_gyro1 = feature(m_gyro1)
#valmean_gyro2, valmax_gyro2, valmin_gyro2, valstd_gyro2, valenergy_gyro2 = feature(m_gyro2)
#valmean_gyro3, valmax_gyro3, valmin_gyro3, valstd_gyro3, valenergy_gyro3 = feature(m_gyro3)
#'''
#'''
#valmean_mag, valmax_mag, valmin_mag, valstd_mag, valenergy_mag = feature(m_mag)
#valmean_mag1, valmax_mag1, valmin_mag1, valstd_mag1, valenergy_mag1 = feature(m_mag1)
#'''
##for acceleration
#
#freq_domain = []
#freq_domain1 = []
#freq_domain2 = []
#freq_domain3 = []
#
#
#for i in range (len(valmean)):
#    freq_domain.append(i)
#    '''
#for i in range (len(valmean1)):
#    freq_domain1.append(i)
#for i in range (len(valmean2)):
#    freq_domain2.append(i)
#for i in range (len(valmean3)):
#    freq_domain3.append(i)
#'''
#    
##for gyro
#
#freq_domain_gyro = []
#freq_domain_gyro1 = []
#freq_domain_gyro2 = []
#freq_domain_gyro3 = []
#
#for i in range (len(valmean_gyro)):
#    freq_domain_gyro.append(i)
#
#
##for magnetometer (for frequency domain)
#'''
#freq_domain_mag = []
#freq_domain_mag1 = []
#for i in range (len(valmean_mag)):
#    freq_domain_mag.append(i)
#for i in range (len(valmean_mag1)):
#    freq_domain_mag1.append(i)
#'''
#
###############PLOT 
##for acceleration   
#'''
#plt.subplot(512)
#plt.plot(freq_domain,valmean, color='blue')
#plt.title('Mean')
##plt.xlim([0,40])
#
#plt.subplot(513)
#plt.plot(freq_domain,valmax, color='blue')
#
#plt.title('Max')
##plt.xlim([0,40])
#
#plt.subplot(514)
#plt.plot(freq_domain,valmin, color='blue')
#plt.title('Min')
##plt.xlim([0,40])
#
#plt.subplot(515)
#plt.plot(freq_domain,valenergy, color='blue')
#
#plt.title('Energy')
##plt.xlim([0,40])
#'''
#
##for zyro
#'''
#plt.subplot(512)
#plt.plot(freq_domain_gyro,valmean_gyro, color='blue')
#plt.title('Mean')
##plt.xlim([0,40])
#
#plt.subplot(513)
#plt.plot(freq_domain_gyro,valmax_gyro, color='blue')
#plt.title('Max')
##plt.xlim([0,40])
#
#plt.subplot(514)
#plt.plot(freq_domain_gyro,valmin_gyro, color='blue')
#plt.title('Min')
##plt.xlim([0,40])
#
#plt.subplot(515)
#plt.plot(freq_domain_gyro,valenergy_gyro, color='blue')
#plt.title('Energy')
##plt.xlim([0,40])
#'''
#
##for magnetometer
#'''
#plt.subplot(512)
#plt.plot(freq_domain_mag,valmean_mag, color='blue')
#plt.plot(freq_domain_mag1,valmean_mag1 , color='red')
#plt.title('Mean')
#plt.xlim([0,40])
#
#plt.subplot(513)
#plt.plot(freq_domain_mag,valmax_mag, color='blue')
#plt.plot(freq_domain_mag1,valmax_mag1 , color='red')
#plt.title('Max')
#plt.xlim([0,40])
#
#plt.subplot(514)
#plt.plot(freq_domain_mag,valmin_mag, color='blue')
#plt.plot(freq_domain_mag1,valmin_mag1 , color='red')
#plt.title('Min')
#plt.xlim([0,40])
#
#
#plt.subplot(515)
#plt.plot(freq_domain_mag, valenergy_mag, color='blue')
#plt.plot(freq_domain_mag1, valenergy_mag1, color='red')
#plt.title('Energy')
#plt.xlim([0,40])
#'''
#
#
##plot test for Scatter
#
#plt.subplot(512, projection='3d')
#fig = plt.figure(figsize=(7,5))
#ax = Axes3D(fig)
#line1 = ax.plot(m, m_gyro, valenergy, 'ok', color='red')
