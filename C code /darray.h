#ifndef darray_h
#define darray_h
#include <stdlib.h>

typedef struct {
  float *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize);
//void initAllArray(Array *a, size_t initialSize);
void insertArray(Array *a, float element);
void freeArray(Array *a);

#endif