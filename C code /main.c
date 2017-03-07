#include "AccGenerate.h"
#include "MahonyAHRS.h"
#include "MiscMathFunc.h"

#include <stdlib.h>
#include <math.h>

#define PI  3.14159265358979323846f

int main()
{
	
	int len=5;
	/*
	float offset_Gx=0.589483;
	float offset_Gy=1.419483;
	float offset_Gz=-5.094889;
	*/
	float offset_Gx=0;
	float offset_Gy=0;
	float offset_Gz=0;
	float Gx[5]={0.001953,0.001953,0.001404,0.001404,0.002075};
	float Gy[5]={0.006409,0.006409,0.006012,0.006012,0.007172};
	float Gz[5]={-0.020691,-0.020691,-0.019806,-0.019806,-0.021637};
	
	float Ax[5]={-0.026978,-0.026978,-0.026978,-0.026978,-0.02655};
	float Ay[5]={0.018768,0.018768,0.018768,0.018768,0.01944};
	float Az[5]={0.244568,0.244568,0.244568,0.244568,0.248383};

	float Mx[5]={-0.010437,-0.010437,-0.010437,-0.010437,-0.016632};
	float My[5]={0.013275,0.013275,0.013275,0.013275,0.011993};
	float Mz[5]={-0.092957,-0.092957,-0.092957,-0.092957,-0.09024};
	
	//Assume we have the data in 9 separate nx1 arrays, Gx Gy Gz Ax Ay Az Mx My Mz
	//n is stored in int len
	
	Array tcAccX;
	Array tcAccY;
	Array tcAccZ;
	initArray(&tcAccX,len);
	initArray(&tcAccY,len);
	initArray(&tcAccZ,len);

	int i=0;
	float new_Gx=0;
	float new_Gy=0;
	float new_Gz=0;
	float tmpAcc[3];
	for(i;i<len;i++){
		new_Gx=(Gx[i]-offset_Gx)*PI/180.0f;
		new_Gy=(Gy[i]-offset_Gy)*PI/180.0f;
		new_Gz=(Gz[i]-offset_Gz)*PI/180.0f;
		tcAcc(tmpAcc, new_Gx, new_Gy, new_Gz, Ax[i], Ay[i], Az[i], Mx[i], My[i], Mz[i]);

		insertArray(&tcAccX,tmpAcc[0]);
		insertArray(&tcAccY,tmpAcc[1]);
		insertArray(&tcAccZ,tmpAcc[2]);
		//printf("\n q0 %f q1 %f q2 %f q3 %f \n",q0,q1,q2,q3);
		printf("Accx %f Accy %f AccZ %f \n \n",tcAccX.array[i],tcAccY.array[i],tcAccZ.array[i]);
	}
	//tilt compensated x direction acc is stored in tcAccX
	freeArray(&tcAccX);
	freeArray(&tcAccY);
	freeArray(&tcAccZ);
}

