#include "helpers_cuda.h"
#include "kernel_avg.h"


static const unsigned int BLOCK_WIDTH = 16;
static const unsigned int BLOCK_HEIGHT = 16;


static __global__ void sum(float* avg,unsigned char* obj,unsigned int size,unsigned int width,unsigned int height,int nObjects,float m)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( x < width*height)
    {
		//unsigned int size2 = width*height*sizeof(char);
		float res=0;
		
		for(int i=0; i < nObjects; i++)
			res += obj[i*size + x];
			
		avg[x] = res*m;
	}
}
 

void avg_sum(float* avg,unsigned char * obj,unsigned int size, unsigned int width, unsigned int height, int nObjects)
{
   
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads
    dim3 grid(div_round_up(width, block.x), div_round_up(height, block.y)); // cantidad bloques
    
    float m = 1.0f/(float)nObjects; //para ahorrar calculos   

    sum<<<grid, block>>>(avg,obj,size,width,height,nObjects,m);
}
