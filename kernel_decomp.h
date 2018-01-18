#pragma once

void decomp(float* coeffs,float *avg,unsigned char* obj,float* eigInput,int nObjects,int nEigObjs,int objStep,int size_height,int size_width,int offset);
void decompOp(float* coeffs,float *avg,unsigned char* obj,float* eigInput,int nObjects,int nEigObjs,int objStep,int size_height,int size_width,int offset);
