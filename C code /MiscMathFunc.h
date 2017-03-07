#ifndef quat2rotMat_h
#define quat2rotMat_h
#include <stdlib.h>

typedef struct {
  float *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize);
void insertArray(Array *a, float element);
void freeArray(Array *a);


void Quat2rotMat(float *rotMat, float qw,float qx,float qy,float qz);
void MatMult33x31(float *product, float m1[9],float m2[3]);
void MatMult31xconst(float *product, float m1[3],float p);
void MatAdd31(float *sum, float m1[3],float m2[3]);

#endif