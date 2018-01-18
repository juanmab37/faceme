#include "helpers_cuda.h"
#include "kernel_decomp.h"

static const unsigned int BLOCK_WIDTH = 32;
static const unsigned int BLOCK_HEIGHT = 32;

static __global__ void decomp_k(float* coeffs,float *avg,unsigned char* obj,float* eigInput,int nObjects, int nEigObjs,int objStep,int size_height,int size_width,int offset)
{	
    unsigned int l = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
	
	float w = 0.0f;
	size_t img_l = size_height*size_width*sizeof(float);
	
	if(l < nObjects && j < nEigObjs )
	{
		unsigned char* bo = &obj[l*img_l]; //(obj + l*img_l)
		float* be = (eigInput + j*size_height*size_width);
		float* ba = avg;
		
		for(int i = 0; i < size_height; i++, bo += objStep, be += objStep, ba += objStep )
			for(int k = 0; k < size_width-3; k ++ )
			{
				float o = (float) bo[k];
				float e = be[k];
				float a = ba[k];
				w += e * (o - a);
			}
			
		coeffs[j+l*offset] = w;
	}
}

__global__ void decompOptima(unsigned char* A,float* B,float *avg, float *C, int M, int N, int U, int offset) {	
	float sum=0.0f;
	int tx, ty, i, j;
	tx = threadIdx.x;
	ty = threadIdx.y;
	i = tx + blockIdx.x * blockDim.x;
	j = ty + blockIdx.y * blockDim.y;
	__shared__ float As[32][32];
	__shared__ float Bs[32][32];
	__shared__ float Avgs[32];
	
	size_t img_l = M*sizeof(float);
	
	/* N -> alto de C
	M -> tamaño en comun con A y B, es decir, M es el tamaño de cada cara
	U -> ancho de C */ 
	
	// Recorre los mosaicos de A y B necesarios para computar la submatriz de C
	for (int tile=0; tile < div_round_up(M,32); tile++)
	{
		if(tile < (M/32) ) {
		//Carga los mosaicos (32x32) de A y B en paralelo (y de forma traspuesta)
			if(i < N)
				As[tx][ty]= (float)A[(i*img_l) + (ty+(tile*32))]; //obj
			if(j < U)
				Bs[tx][ty]= B[j*M + (tx+(tile*32))]; //eig (eigInput + j*size_height*size_width);
			if(ty == 0)
				Avgs[tx] = avg[tx+(tile*32)];
				
			__syncthreads();
			
			//float *a = avg;
			
		// Computa los resultados para la submatriz de C
			if(i < N && j < U) 
				for (int k=0; k<32; k++) {// Los datos también se leerán de forma traspuesta
					float f = Avgs[k];
					sum += Bs[k][ty] * (As[tx][k] - f) ; //(As[tx][k] - f) * (Bs[k][ty] - f);
				}
			__syncthreads();
		} else { //if ( tile == M/32)
        
			if(ty < (M%32) && i < N)
				As[tx][ty]= (float)A[(i*img_l) + (ty+(tile*32))];
			if(tx < (M%32) && j < U)  
				Bs[tx][ty]= B[j*M + (tx+(tile*32))];
			if(ty == 0 && tx < (M%32))
				Avgs[tx] = avg[tx+(tile*32)];
      
			__syncthreads();
		
			if(i < N && j < U) 
				for (int k=0; k<(M%32); k++) {// Los datos también se leerán de forma traspuesta
					float f = Avgs[k];
					sum += Bs[k][ty] * (As[tx][k] - f) ;//(As[tx][k] - f) * (Bs[k][ty] - f);
				}
			__syncthreads();
		}
	}
	// Escribe en paralelo todos los resultados obtenidos por el bloque
	if(i < N && j < U)
		C[j + i*offset] = sum; //coeffs[j+l*offset] = w;
 
}

/*
 for(int l = 0; l < nObjects; l++)
		for(int  j = 0; j < nEigObjs; j++ )
		{
			float *eigObj = eigInput[j];

			w = 0.0f;
			uchar* bo = obj[l];
			float* be = eigObj;
			float* ba = avg;
			
			for(int i = 0; i < size_height; i++, bo += objStep, be += objStep, ba += objStep )
			{
				for(int k = 0; k < size_width; k ++ )
				{
					float o = (float) bo[k];
					float e = be[k];
					float a = ba[k];
					w += e * (o - a);
				}
			}
			coeffs[j+l*offset] = w;
		 } 
*/




void decomp(float* coeffs,float *avg,unsigned char* obj,float* eigInput,int nObjects,int nEigObjs,int objStep,int size_height,int size_width,int offset)
{
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(nObjects, block.x), div_round_up(nEigObjs, block.y)); // cantidad bloques
    
    decomp_k<<<grid, block>>>(coeffs,avg,obj,eigInput,nObjects,nEigObjs,objStep,size_height,size_width,offset);
    
}

void decompOp(float* coeffs,float *avg,unsigned char* obj,float* eigInput,int nObjects,int nEigObjs,int objStep,int size_height,int size_width,int offset)
{
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(nObjects, block.x), div_round_up(nEigObjs, block.y)); // cantidad bloques
    
    decompOptima<<<grid, block>>>(obj,eigInput,avg, coeffs, size_height*size_width, nObjects, nEigObjs,offset);
    
}
