#include <stdio.h>
//#include <mraa/i2c.h>
#include <math.h>
#include "LSM9DS0.h"
#include "AccGenerate.h"
#include "MahonyAHRS.h"
#include "MiscMathFunc.h"

int main() {
	const int Fs=100; //sample freq
	float deltaT=1000000/Fs;  // sample period in us
	int Window=400;
	int Step=100;
	FILE *pFile;
	char* path="/home/root/DataCollection/datalog.csv";
	
	
	data_t accel_data, gyro_data, mag_data;
	data_t gyro_offset;
	int16_t temperature;
	float a_res, g_res, m_res;
	mraa_i2c_context accel, gyro, mag;
	accel_scale_t a_scale = A_SCALE_4G;
	gyro_scale_t g_scale = G_SCALE_245DPS;
	mag_scale_t m_scale = M_SCALE_2GS;
	int acc_scale, gyro_scale, mag_scale;
	acc_scale=4;
	gyro_scale=245;
	mag_scale=2;

	//initialize sensors, set scale, and calculate resolution.
	accel = accel_init();
	set_accel_scale(accel, a_scale);	
	a_res = calc_accel_res(a_scale);
	
	gyro = gyro_init();
	set_gyro_scale(gyro, g_scale);
	g_res = calc_gyro_res(g_scale);
	
	mag = mag_init();
	set_mag_scale(mag, m_scale);
	m_res = calc_mag_res(m_scale);

	gyro_offset = calc_gyro_offset(gyro, g_res);
	
	printf("x: %f y: %f z: %f\n", gyro_offset.x, gyro_offset.y, gyro_offset.z);
	pFile=fopen(path,"w");
	//fprintf(pFile,"Starting time, %10.6lf , time difference c-s, %f,\n",startlogging_epoch,time_diff);
	fprintf(pFile,"Gyro offset: xyz, %f,%f,%f,\n",gyro_offset.x, gyro_offset.y, gyro_offset.z);
	fclose(pFile);
	printf("\n\t\tAccelerometer\t\t\t||");
	printf("\t\t\tGyroscope\t\t\t||");
	printf("\t\t\tMagnetometer\t\t\t||");
	printf("\tTemperature\n");
	
	//Read the sensor data and print them.
	int count=1;
	float time=0;
	pFile=fopen(path,"a"); // Change the path
	char *header="Index,Time(us), GyroX , GyroY , GyroZ ,AccX , AccY , AccZ , MagX , MagY , MagZ\n";
	fprintf(pFile, "%s", header);
	fclose(pFile);
	
	Array axstep;
	Array aystep;
	Array azstep;
	Array gxstep;
	Array gystep;
	Array gzstep;
	initArray(&axstep,step);
	initArray(&aystep,step);
	initArray(&azstep,step);
	initArray(&gxstep,step);
	initArray(&gystep,step);
	initArray(&gzstep,step);
	while(1) {
		accel_data = read_accel(accel, a_res);
		gyro_data = read_gyro(gyro, g_res);
		mag_data = read_mag(mag, m_res);
		temperature = read_temp(accel); //you can put mag as the parameter too.
		float norm_gyrox,norm_gyroy,norm_gyroz;
		float norm_accx,norm_accy,norm_accz;
		float norm_magx,norm_magy,norm_magz;
		norm_gyrox=gyro_data.x/gyro_scale;
		norm_gyroy=gyro_data.y/gyro_scale;
		norm_gyroz=gyro_data.z/gyro_scale;
		norm_accx=accel_data.x/acc_scale;
		norm_accy=accel_data.y/acc_scale;
		norm_accz=accel_data.z/acc_scale;
		norm_magx=mag_data.x/mag_scale;
		norm_magy=mag_data.y/mag_scale;
		norm_magz=mag_data.z/mag_scale;
		pFile=fopen(path,"a");  // Change the path
		fprintf(pFile,"%i, %f,",count,time);
		fprintf(pFile,"%f, %f, %f,", norm_gyrox,norm_gyroy,norm_gyroz);
		fprintf(pFile,"%f, %f, %f,", norm_accx,norm_accy,norm_accz);
		fprintf(pFile,"%f, %f, %f\n", norm_magx,norm_magy,norm_magz);
		fclose(pFile);
  		printf("Time: %f",time);
		printf("\n");
		usleep(deltaT);
		count=count+1;
		time=time+deltaT;
		if(counter==100)
		{
			
		}
		
		
		
	}	
	return 0;	
}
