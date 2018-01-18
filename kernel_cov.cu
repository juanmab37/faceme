#include "helpers_cuda.h"
#include "kernel_cov.h"

static const unsigned int BLOCK_WIDTH = 32;
static const unsigned int BLOCK_HEIGHT = 32;

static __global__ void covarianza(float* avg,unsigned char* objects,float* covarMatrix, int obj_step,int nObjects,int size_width, int size_height)
{	
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	
	
	if(y >= x && y < nObjects && x < nObjects) {
		size_t img_l = size_height*size_width*sizeof(float); //tamaño de una imagen en objects
		unsigned char *bu1 = (objects + x*img_l);
		unsigned char *bu2 = (objects + y*img_l);
		
		int k, l;
		float w = 0.f;
		float *a = avg;
	
		for( k = 0; k < size_height; k++, bu1 += obj_step, bu2 += obj_step, a += obj_step )
			for( l = 0; l < size_width - 3; l++)
			{
				float f = a[l];
				float u1 = bu1[l];
				float u2 = bu2[l];
				w += (u1 - f) * (u2 - f);
			}
				 
		covarMatrix[x * nObjects + y] = covarMatrix[y * nObjects + x] = w;
		//atomicExch(&covarMatrix[x * nObjects + y], w);
		//atomicExch(&covarMatrix[y * nObjects + x], w);
	}          
}
      
/*          
     for(i = 0; i < nObjects; i++ )
		for(j = i; j < nObjects; j++ )
		{
			int k, l;
			float w = 0.f;
			float *a = avg;
			uchar *bu1 = objects[i];//bu;
			uchar *bu2 = objects[j];

			for( k = 0; k < size_height; k++, bu1 += obj_step, bu2 += obj_step, a += obj_step )
				for( l = 0; l < size_width - 3; l++)
				{
					float f = a[l];
                    float u1 = bu1[l];
                    float u2 = bu2[l];
                    w += (u1 - f) * (u2 - f);
				}
				
            covarMatrix[i * nObjects + j] = covarMatrix[j * nObjects + i] = w;
        }
*/                     


__global__ void covarianzaOptima(unsigned char* A,float *avg, float *C, int N, int M) {	
	float sum=0;
	int tx, ty, i, j;
	tx = threadIdx.x;
	ty = threadIdx.y;
	i = tx + blockIdx.x * blockDim.x;
	j = ty + blockIdx.y * blockDim.y;
	__shared__ float As[32][32];
	__shared__ float Bs[32][32];
	__shared__ float Avgs[32];
	
	size_t img_l = M*sizeof(float);

//Funcion que multiplica A * A^t

	// Recorre los mosaicos de A y B necesarios para computar la submatriz de C
	for (int tile=0; tile < div_round_up(M,32); tile++)
	{
		if(tile < (M/32) ) {
		//Carga los mosaicos (32x32) de A y B en paralelo (y de forma traspuesta)
			if(i < N)
				As[tx][ty]= A[(i*img_l) + (ty+(tile*32))];
			if(j < N)
				Bs[tx][ty]= A[(j*img_l) + (tx+(tile*32))];
			if(ty == 0)
				Avgs[tx] = avg[tx+(tile*32)];
				
			__syncthreads();
			
			//float *a = avg;
			
		// Computa los resultados para la submatriz de C
			if(j >= i && i < N && j < N) 
				for (int k=0; k<32; k++) {// Los datos también se leerán de forma traspuesta
					float f = Avgs[k];
					sum += (As[tx][k] - f) * (Bs[k][ty] - f);
				}
			__syncthreads();
		} else { //if ( tile == M/32)
        
			if(ty < (M%32) && i < N)
				As[tx][ty]= A[(i*img_l) + (ty+(tile*32))];
			if(tx < (M%32) && j < N)  
				Bs[tx][ty]= A[(j*img_l) + (tx+(tile*32))];
			if(ty == 0 && tx < (M%32))
				Avgs[tx] = avg[tx+(tile*32)];
      
			__syncthreads();
		
			if(j >= i && i < N && j < N) 
				for (int k=0; k<(M%32); k++) {// Los datos también se leerán de forma traspuesta
					float f = Avgs[k];
					sum += (As[tx][k] - f) * (Bs[k][ty] - f);
				}
			__syncthreads();
		}
	}
	// Escribe en paralelo todos los resultados obtenidos por el bloque
	if(j >= i && i < N && j < N)
		C[i*N+j] = C[j*N+i] = sum;
 
}

__global__ void MxMonGPU(float *A, float *B, float *C, int N) {	
	float sum=0;
	int tx, ty, i, j;
	tx = threadIdx.x;
	ty = threadIdx.y;
	i = tx + blockIdx.x * blockDim.x;
	j = ty + blockIdx.y * blockDim.y;
	__shared__ float As[32][32];
	__shared__ float Bs[32][32];

//Funcion que multiplica A * B

	// Recorre los mosaicos de A y B necesarios para computar la submatriz de C
	for (int tile=0; tile < div_round_up(N,32); tile++)
	{
		if(tile < (N/32) ) {
		//Carga los mosaicos (32x32) de A y B en paralelo (y de forma traspuesta)
			if(i < N)
				As[tx][ty]= A[(i*N) + (ty+(tile*32))];
			if(j < N)
				Bs[tx][ty]= B[((tx+(tile*32))*N) + j];
			
			__syncthreads();
		// Computa los resultados para la submatriz de C
			if(i < N && j < N)
				for (int k=0; k<32; k++) {// Los datos también se leerán de forma traspuesta
					sum += As[tx][k] * Bs[k][ty];
				}
			__syncthreads();
		} else { //if ( tile == N/32)
        
			if(ty < (N%32) && i < N)
				As[tx][ty]= A[(i*N) + (ty+(tile*32))];
          
			if(tx < (N%32) && j < N)  
				Bs[tx][ty]= B[((tx+(tile*32))*N) + j];
        
			__syncthreads();
			if(i < N && j < N)
				for (int k=0; k<(N%32); k++) {// Los datos también se leerán de forma traspuesta
					sum += As[tx][k] * Bs[k][ty];
				}
			__syncthreads();
		}
	}
	// Escribe en paralelo todos los resultados obtenidos por el bloque
	if(i < N && j < N)
		C[i*N+j] = sum;
 
}

void covarOp(unsigned char* objects,float * avg, float *C, int N, int M)
{
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(N, block.x), div_round_up(N, block.y)); // cantidad bloques
    
    covarianzaOptima<<<grid, block>>>(objects,avg,C,N,M);
}
           
void cov(float* avg,unsigned char* objects,float* covarMatrix, int obj_step, int nObjects,int size_width, int size_height)
{
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT); // cantidad threads por bloque
    dim3 grid(div_round_up(nObjects, block.x), div_round_up(nObjects, block.y)); // cantidad bloques
    
    covarianza<<<grid, block>>>(avg,objects,covarMatrix, obj_step, nObjects,size_width,size_height);
}			


                 
			

