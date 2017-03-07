#include "MiscMathFunc.h"

void Quat2rotMat(float *rotMat, float qw,float qx,float qy,float qz)
{

	rotMat[0]=1-2*qy*qy-2*qz*qz;
	rotMat[1]=2*qx*qy-2*qz*qw;
	rotMat[2]=2*qx*qz+2*qy*qw;
	rotMat[3]=2*qx*qy+2*qz*qw;
	rotMat[4]=1-2*qx*qx-2*qz*qz;
	rotMat[5]=2*qy*qz-2*qx*qw;
	rotMat[6]=2*qx*qz-2*qy*qw;
	rotMat[7]=2*qy*qz+2*qx*qw;
	rotMat[8]=1-2*qx*qx-2*qy*qy;
}

void MatMult33x31(float *product, float m1[9], float m2[3])
{
	product[0]=m1[0]*m2[0]+m1[1]*m2[1]+m1[2]*m2[2];
	product[1]=m1[3]*m2[0]+m1[4]*m2[1]+m1[5]*m2[2];
	product[2]=m1[6]*m2[0]+m1[7]*m2[1]+m1[8]*m2[2];

}

void MatMult31xconst(float *product, float m1[3],float p)
{
	int i=0;
	for (i;i<3;i++)
		product[i]=m1[i]*p;

}

void MatAdd31(float *sum, float m1[3],float m2[3])
{
	int i;
	for (i=0;i<3;i++)
		sum[i]=m1[i]+m2[i];
	
}

void initArray(Array *a, size_t initialSize) {
  a->array = (float *)malloc(initialSize * sizeof(float));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Array *a, float element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = (float *)realloc(a->array, a->size * sizeof(float));
  }
  a->array[a->used++] = element;
}

void freeArray(Array *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}




