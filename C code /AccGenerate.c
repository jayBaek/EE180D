#include "AccGenerate.h"
#include "MiscMathFunc.h"
#include "MahonyAHRS.h"

void tcAcc(float *tcA, float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz)
{	
	float Acc[3]={ax,ay,az};
	float R[9];
	MahonyAHRSupdateIMU(gx, gy, gz, ax, ay, az); // Check if need to change g into radians
	Quat2rotMat(R,q0,q1,q2,q3);
	MatMult33x31(tcA, R,Acc);	
}

