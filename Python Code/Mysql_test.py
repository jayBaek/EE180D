#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:53:19 2017

@author: JayBaek
"""

import mysql.connector
import numpy as np

config = {
'user': 'root', 
'password':'s04j15', 
'host': '127.0.0.1', 
'port':'3306',
'database': 'DATA', 
'raise_on_warnings': True,}

conn = mysql.connector.connect(**config)
curs = conn.cursor()

##data fetch
#sql = "select * from new_table where time > 0 and time <= 400".format(', ')
#curs.execute(sql)
#rows = curs.fetchall()
#
#sql = "select Acc_x from new_table".format(',')
#curs.execute(sql)
#rows1 = curs.fetchall()
#print rows1[0]

# Drop table if it already exist using execute() method.
#curs.execute("DROP TABLE IF EXISTS EMPLOYEE")

# Create table as per requirement
name = "TEST"
sql = """CREATE TABLE  %s (
         Time    float  NOT NULL,
         Gyro_x  float  NOT NULL,
         Gyro_y  float  NOT NULL,
         Gyro_z  float  NOT NULL,
         Acc_x   float  NOT NULL,
         Acc_y   float  NOT NULL,
         Acc_z   float  NOT NULL
         )""" %(name)

#curs.execute(sql)

def load_data(path):
    time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z, Mag_x, Mag_y, Mag_z = np.loadtxt(path, delimiter=',', unpack=True, dtype=np.float)
    return time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z

path = '/Users/JayBaek/Documents/Github/EE180D/data/running_test1.csv'
time, x, y, z, x1, y1, z1 = load_data(path)

array = []
array.append([2, 0.231, 1.2131023, 3.12313, 2.123131, 4.123123, 4.12312])
array.append([3, 0.231, 1.2131023, 3.12313, 2.123131, 4.123123, 4.12312])
array.append([4, 0.231, 1.2131023, 3.12313, 2.123131, 4.123123, 4.12312])

temp = np.array(array, dtype=float)
temp1 =[]
temp1.append(temp)

a = time[1]
print test

#print array[1]
#curs.executemany("""INSERT INTO %s 
#                 VALUES (%%s, %%s, %%s, %%s, %%s, %%s, %%s)"""%name, (temp))

sql = """INSERT INTO TEST
        (Time, Gyro_x, Gyro_y, Gyro_z, Acc_x, Acc_y, Acc_z)
         VALUES (%s, %s, %s, %s, %s, %s, %s)"""%test
try:
   # Execute the SQL command
   curs.execute(sql)
   # Commit your changes in the database
   conn.commit()
except:
   # Rollback in case there is any error
   conn.rollback()

## disconnect from server
#Acc_x = []
#for i in range (curs.rowcount):
#    Acc_x.append(rows1[i])
#
#print curs.rowcount
#for i in range (curs.rowcount):
#    row = curs.fetchone()
##    Acc_x.append(row[0])
##    Acc_x = curs.fetchone
#    print 'Acc_x = %f' %Acc_x
#print(rows)

conn.close()