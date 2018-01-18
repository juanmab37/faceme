#pragma once

void cov(float* avg,unsigned char* objects,float* covarMatrix, int obj_step, int nObjects,int size_width, int size_height);
void covarOp(unsigned char *objects,float* avg, float *C, int N, int M);
