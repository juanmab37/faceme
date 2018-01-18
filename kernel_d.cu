#include "helpers_cuda.h"
#include "kernel_d.h"

static const unsigned int BLOCK_WIDTH = 8;
static const unsigned int BLOCK_HEIGHT = 8;
							
static __global__ void eig_k(int m1,int nObjects,unsigned char* objs_gpu,float* eigs_gpu,float* avg_gpu,float* eigVals_gpu,
									float*ev_gpu,int eig_size_height,int eig_size_width,int eig_step)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    size_t img_l = eig_size_height*eig_size_width*sizeof(float);
    
	if(x < nObjects && y < m1 )
	{		
		float v = eigVals_gpu[y] * ev_gpu[y * nObjects + x];
		float *be = (eigs_gpu + y*(eig_size_height*eig_size_width));
		unsigned char *bu = (objs_gpu + x*img_l);
		
		float *bf = avg_gpu;
		 
		for(int p = 0; p < eig_size_height; p++, bu += eig_step, bf += eig_step, be += eig_step )
			for(int l = 0; l < eig_size_width - 3; l++ )
			{
				float f = bf[l];
				float u = bu[l];			
				atomicAdd(&be[l],v * (u - f));//be[l] += v * (u - f);
			}
	}

}

/*
		for( k = 0; k < nObjects; k++ )
	{
		for( i = 0; i < m1; i++ )
		{
			float v = eigVals[i] * ev[i * nObjects + k];
			float *bw = output[i];
			uchar *bu = input[k];

			bf = avg;
			
			
			
			bw += v * (bu - bf)
			
			for(int p = 0; p < eig_size_height; p++, bu += eig_step, bf += eig_step, bw += eig_step )
				for(int l = 0; l < eig_size_width; l ++ )
				{
					float f = bf[l];
					float u = bu[l];			
					bw[l] += v * (u - f);
				}				
		}                  
	}  
*/

			
void eig(int m1,int nObjects,unsigned char* objs_gpu,float* eigs_gpu,float* avg_gpu,float* eigVals_gpu,float*ev_gpu,int eig_size_height,int eig_size_width,int eig_step)
{
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(nObjects, block.x), div_round_up(nObjects-1, block.y)); // cantidad bloques
    
    //float* bf=0;
    
    eig_k<<<grid, block>>>(m1,nObjects,objs_gpu,eigs_gpu,avg_gpu,eigVals_gpu,ev_gpu,eig_size_height,eig_size_width,eig_step);
}						

static __global__ void eigV_k(int m1,float* eigVals_gpu)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(x < m1)
	{
		eigVals_gpu[x] = 1.0f / (eigVals_gpu[x] * eigVals_gpu[x]);
	}
}

//eigVals[i] = 1.0f / (eigVals[i] * eigVals[i]);

void eigV(int m1,float* eigVals_gpu)
{
	dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(m1, block.x),1); // cantidad bloques
     
    eigV_k<<<grid, block>>>(m1,eigVals_gpu);
}
