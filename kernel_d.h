#pragma once

void eig(int m1,int nObjects,unsigned char* objs_gpu,float* eigs_gpu,float* avg_gpu,float* eigVals_gpu,float*ev_gpu,
					int eig_size_height,int eig_size_width,int eig_step);

void eigV(int m1,float* eigVals_gpu);
