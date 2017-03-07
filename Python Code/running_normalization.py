#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:32:05 2017

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
import pylab

#loading running data 1
path = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/running_test1.csv'
time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z, Mag_x, Mag_y, Mag_z = np.loadtxt(path, delimiter = ',', unpack = True)

#loading running data 2 
path1 = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/running_test2.csv'
time1, Gyro_x1, Gyro_y1, Gyro_z1, Acc_x1, Acc_y1, Acc_z1, Mag_x1, Mag_y1, Mag_z1 = np.loadtxt(path1, delimiter = ',', unpack = True)

#loading running data 3 
path2 = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/running_test3.csv'
time2, Gyro_x2, Gyro_y2, Gyro_z2, Acc_x2, Acc_y2, Acc_z2, Mag_x2, Mag_y2, Mag_z2 = np.loadtxt(path2, delimiter = ',', unpack = True)

#loading running data 4 
path3 = '/Users/JayBaek/Google Drive/UCLA/CLASSES/EE180D/Project/data/running_test4.csv'
time3, Gyro_x3, Gyro_y3, Gyro_z3, Acc_x3, Acc_y3, Acc_z3, Mag_x3, Mag_y3, Mag_z3 = np.loadtxt(path3, delimiter = ',', unpack = True)


#From Accelerometer
mpre = Acc_x*Acc_x + Acc_y*Acc_y + Acc_z*Acc_z
mpre1 = Acc_x1*Acc_x1 + Acc_y1*Acc_y1 + Acc_z1*Acc_z1
mpre2 = Acc_x2*Acc_x2 + Acc_y2*Acc_y2 + Acc_z2*Acc_z2
mpre3 = Acc_x3*Acc_x3 + Acc_y3*Acc_y3 + Acc_z3*Acc_z3


m = np.sqrt(mpre)
m1 = np.sqrt(mpre1)
m2 = np.sqrt(mpre2)
m3 = np.sqrt(mpre3)


#From Gyroscope
mpre_gyro = Gyro_x*Gyro_x + Gyro_y*Gyro_y + Gyro_z*Gyro_z
mpre_gyro1 = Gyro_x1*Gyro_x1 + Gyro_y1*Gyro_y1 + Gyro_z1*Gyro_z1
mpre_gyro2 = Gyro_x2*Gyro_x2 + Gyro_y2*Gyro_y2 + Gyro_z2*Gyro_z2
mpre_gyro3 = Gyro_x3*Gyro_x3 + Gyro_y3*Gyro_y3 + Gyro_z3*Gyro_z3


m_gyro = np.sqrt(mpre_gyro)
m_gyro1 = np.sqrt(mpre_gyro1)
m_gyro2 = np.sqrt(mpre_gyro2)
m_gyro3 = np.sqrt(mpre_gyro3)

#From Magnetometer
'''
mpre_mag = Mag_x*Mag_x + Mag_y*Mag_y + Mag_z*Mag_z
mpre_mag1 = Mag_x1*Mag_x1 + Mag_y1*Mag_y1 + Mag_z1*Mag_z1
m_mag = np.sqrt(mpre_mag)
m_mag1 = np.sqrt(mpre_mag1)
'''

#print(m)

#for Accelration 

plt.figure(figsize=(8,10))
plt.subplot(511)
plt.tight_layout() #for bigger layout
plt.subplots_adjust(hspace = 0.45)

'''
plt.plot(time, m, color='blue')
plt.plot(time1, m1, color= 'red')
plt.plot(time2, m2, color= 'green')
plt.plot(time3, m3, color= 'cyan')

plt.title('Magnitude of Acceleration with 4 running data samples')
plt.xlabel('time')
plt.xlim([600, 1000])
'''

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

def feature(x, fftsize=256, overlap=4):
    meanamp = []
    maxamp = []
    minamp = []
    stdamp = []
    energyamp = []

    hop = fftsize / overlap
    for i in range(0, len(x)-fftsize, hop):
        #print(i)
        #print(np.mean(x[i:i+fftsize]))
        meanamp.append(np.array(np.mean(x[i:i+fftsize])))
        maxamp.append(np.array(np.max(x[i:i+fftsize])))
        minamp.append(np.array(np.min(x[i:i+fftsize])))
        stdamp.append(np.array(np.std(x[i:i+fftsize])))
        energyamp.append(np.array(np.sum(np.power(x[i:i+fftsize],2))))
        
    return meanamp ,maxamp ,minamp,stdamp,energyamp

 
valmean, valmax, valmin, valstd, valenergy = feature(m)
valmean1, valmax1, valmin1, valstd1, valenergy1 = feature(m1)
valmean2, valmax2, valmin2, valstd2, valenergy2 = feature(m2)
valmean3, valmax3, valmin3, valstd3, valenergy3 = feature(m3)


valmean_gyro, valmax_gyro, valmin_gyro, valstd_gyro, valenergy_gyro = feature(m_gyro)
valmean_gyro1, valmax_gyro1, valmin_gyro1, valstd_gyro1, valenergy_gyro1 = feature(m_gyro1)
valmean_gyro2, valmax_gyro2, valmin_gyro2, valstd_gyro2, valenergy_gyro2 = feature(m_gyro2)
valmean_gyro3, valmax_gyro3, valmin_gyro3, valstd_gyro3, valenergy_gyro3 = feature(m_gyro3)

'''
valmean_mag, valmax_mag, valmin_mag, valstd_mag, valenergy_mag = feature(m_mag)
valmean_mag1, valmax_mag1, valmin_mag1, valstd_mag1, valenergy_mag1 = feature(m_mag1)
'''
#for acceleration

freq_domain = []
freq_domain1 = []
freq_domain2 = []
freq_domain3 = []


for i in range (len(valmean)):
    freq_domain.append(i)
for i in range (len(valmean1)):
    freq_domain1.append(i)
for i in range (len(valmean2)):
    freq_domain2.append(i)
for i in range (len(valmean3)):
    freq_domain3.append(i)

    
#for gyro

freq_domain_gyro = []
freq_domain_gyro1 = []
freq_domain_gyro2 = []
freq_domain_gyro3 = []

for i in range (len(valmean_gyro)):
    freq_domain_gyro.append(i)
for i in range (len(valmean_gyro1)):
    freq_domain_gyro1.append(i)
for i in range (len(valmean_gyro2)):
    freq_domain_gyro2.append(i)
for i in range (len(valmean_gyro3)):
    freq_domain_gyro3.append(i)
    


#for magnetometer (for frequency domain)
'''
freq_domain_mag = []
freq_domain_mag1 = []
for i in range (len(valmean_mag)):
    freq_domain_mag.append(i)
for i in range (len(valmean_mag1)):
    freq_domain_mag1.append(i)
'''

##############PLOT 
#for acceleration   
'''
plt.subplot(512)
plt.plot(freq_domain,valmean, color='blue')
plt.plot(freq_domain1,valmean1 , color='red')
plt.plot(freq_domain2,valmean2 , color='green')
plt.plot(freq_domain3,valmean3 , color='cyan')
plt.title('Mean')
plt.xlim([0,40])

plt.subplot(513)
plt.plot(freq_domain,valmax, color='blue')
plt.plot(freq_domain1,valmax1 , color='red')
plt.plot(freq_domain2,valmax2 , color='green')
plt.plot(freq_domain3,valmax3 , color='cyan')
plt.title('Max')
plt.xlim([0,40])

plt.subplot(514)
plt.plot(freq_domain,valmin, color='blue')
plt.plot(freq_domain1,valmin1 , color='red')
plt.plot(freq_domain2,valmin2 , color='green')
plt.plot(freq_domain3,valmin3 , color='cyan')
plt.title('Min')
plt.xlim([0,40])

plt.subplot(515)
plt.plot(freq_domain,valenergy, color='blue')
plt.plot(freq_domain1,valenergy1 , color='red')
plt.plot(freq_domain2,valenergy2 , color='green')
plt.plot(freq_domain3,valenergy3 , color='cyan')
plt.title('Energy')
plt.xlim([0,40])
'''

#for zyro
'''
plt.subplot(512)
plt.plot(freq_domain_gyro,valmean_gyro, color='blue')
plt.plot(freq_domain_gyro1,valmean_gyro1 , color='red')
plt.plot(freq_domain_gyro2,valmean_gyro2 , color='green')
plt.plot(freq_domain_gyro3,valmean_gyro3 , color='cyan')
plt.title('Mean')
plt.xlim([0,40])

plt.subplot(513)
plt.plot(freq_domain_gyro,valmax_gyro, color='blue')
plt.plot(freq_domain_gyro1,valmax_gyro1 , color='red')
plt.plot(freq_domain_gyro2,valmax_gyro2 , color='green')
plt.plot(freq_domain_gyro3,valmax_gyro3 , color='cyan')
plt.title('Max')
plt.xlim([0,40])

plt.subplot(514)
plt.plot(freq_domain_gyro,valmin_gyro, color='blue')
plt.plot(freq_domain_gyro1,valmin_gyro1 , color='red')
#plt.plot(freq_domain_gyro2,valmin_gyro2 , color='greed')
plt.plot(freq_domain_gyro3,valmin_gyro3 , color='cyan')
plt.title('Min')
plt.xlim([0,40])

plt.subplot(515)
plt.plot(freq_domain_gyro,valenergy_gyro, color='blue')
plt.plot(freq_domain_gyro1,valenergy_gyro1 , color='red')
plt.plot(freq_domain_gyro2,valenergy_gyro2 , color='green')
plt.plot(freq_domain_gyro3,valenergy_gyro3 , color='cyan')
plt.title('Energy')
plt.xlim([0,40])
'''

#for magnetometer
'''
plt.subplot(512)
plt.plot(freq_domain_mag,valmean_mag, color='blue')
plt.plot(freq_domain_mag1,valmean_mag1 , color='red')
plt.title('Mean')
plt.xlim([0,40])

plt.subplot(513)
plt.plot(freq_domain_mag,valmax_mag, color='blue')
plt.plot(freq_domain_mag1,valmax_mag1 , color='red')
plt.title('Max')
plt.xlim([0,40])

plt.subplot(514)
plt.plot(freq_domain_mag,valmin_mag, color='blue')
plt.plot(freq_domain_mag1,valmin_mag1 , color='red')
plt.title('Min')
plt.xlim([0,40])


plt.subplot(515)
plt.plot(freq_domain_mag, valenergy_mag, color='blue')
plt.plot(freq_domain_mag1, valenergy_mag1, color='red')
plt.title('Energy')
plt.xlim([0,40])
'''

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
    
def value_save(val1, val2, val3, val4, domain1, domain2, domain3, domain4):
    data = [len(domain1), len(domain2), len(domain3), len(domain4)]
    index = min(data)
    print (index)
    path = '/Users/JayBaek/Desktop/energy_value_running.csv'
    remove_file(path)
    for i in range (index):
        saveFile = open('energy_value_running.csv', 'a')
        saveFile.write(str(i) + ',' + str(val1[i]) + ',' + str(val2[i]) + ',' + str(val3[i]) + ',' + str(val4[i]))
        saveFile.write('\n')
    saveFile.close()

value_save(valenergy, valenergy1, valenergy2, valenergy3, freq_domain, freq_domain1, freq_domain2, freq_domain3)


def normalization(val):
    energy_nor = (val - min(val)) / (max(val)-min(val)) 
    return energy_nor

#for acceleration    
energy_nor = normalization(valenergy)
energy_nor1 = normalization(valenergy1)
energy_nor2 = normalization(valenergy2)
energy_nor3 = normalization(valenergy3)

#for gyroscope
energy_gyro_nor = normalization(valenergy_gyro)
energy_gyro_nor1 = normalization(valenergy_gyro1)
energy_gyro_nor2 = normalization(valenergy_gyro2)
energy_gyro_nor3 = normalization(valenergy_gyro3)


#figure to compare nomalized and unnormalized
plt.subplot(511)
plt.plot(freq_domain, energy_nor, color='blue')
plt.plot(freq_domain1, energy_nor1, color = 'red')
plt.plot(freq_domain2, energy_nor2, color = 'green')
plt.plot(freq_domain3, energy_nor3, color = 'cyan')
plt.title('Normalized Energy for Acceleration')
plt.xlim([0,40])

plt.subplot(512)
plt.plot(freq_domain,valenergy, color='blue')
plt.plot(freq_domain1,valenergy1, color='red')
plt.plot(freq_domain2,valenergy2, color='green')
plt.plot(freq_domain3,valenergy3, color='cyan')
plt.title('Energy')
plt.xlim([0,40])

plt.subplot(513)
plt.plot(freq_domain_gyro, energy_gyro_nor, color='blue')
plt.plot(freq_domain_gyro1,energy_gyro_nor1, color='red')
plt.plot(freq_domain_gyro2,energy_gyro_nor2, color='green')
plt.plot(freq_domain_gyro3, energy_gyro_nor3, color='cyan')
plt.title('Normalized Energy for Gyro')

plt.subplot(514)
plt.plot(freq_domain_gyro, valenergy_gyro, color='blue')
plt.plot(freq_domain_gyro1,valenergy_gyro1, color='red')
plt.plot(freq_domain_gyro2,valenergy_gyro2, color='green')
plt.plot(freq_domain_gyro3, valenergy_gyro3, color='cyan')
plt.title('Energy for Gyro')

print('Normalization is done')

def avg_value(val1, val2, val3, val4, domain1, domain2, domain3, domain4):
    data = [len(domain1), len(domain2), len(domain3), len(domain4)]
    index = min(data)
    print (index)
    path = '/Users/JayBaek/Desktop/Avg_conversion.csv'
    remove_file(path)
    for i in range (index):
        saveFile = open('Avg_conversion.csv', 'a')
        saveFile.write(str(val1[i]) + ',' + str(val2[i]) + ',' + str(val3[i]) + ',' + str(val4[i]))
        saveFile.write('\n')
    saveFile.close()
    data_path = '/Users/JayBaek/Desktop/Avg_conversion.csv'
    file_data = np.loadtxt(data_path, delimiter=',')
    
    path = '/Users/JayBaek/Desktop/Avg_data.csv'
    remove_file(path)
    avg_val = []
    for i in range (index):
        avg_val= sum(file_data[i])/4
        #print(avg_val)
        saveFile = open('Avg_data.csv', 'a')
        saveFile.write(str(avg_val))
        saveFile.write('\n')
    saveFile.close()
        
        
avg_value(energy_nor, energy_nor1, energy_nor2, energy_nor3, freq_domain, freq_domain1, freq_domain2, freq_domain3)

