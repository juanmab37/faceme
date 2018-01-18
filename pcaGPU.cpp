#include "helpers_cuda.h"
#include "kernel_avg.h"
#include "kernel_cov.h"
#include "kernel_d.h"
#include "pca.h"
#include <cuda_runtime.h>
#include <sys/time.h>

float* avg_gpu;
unsigned char *objs_gpu;

int* PCAgpu_i(int nObjects, unsigned char** objs,int obj_step,int eig_step,int obj_size_width, int obj_size_height ,float* avg_data,int avg_step)
{
	int i,j,*device;
	CHECK_CUDA_CALL(cudaSetDevice(1));
	CHECK_CUDA_CALL(cudaGetDevice(device));
	printf("dev: %d\n",*device);
	float *bf = 0;
    int avg_size_width;
    int avg_size_height;	
    int eig_size_width;
    int eig_size_height;	
    struct timeval start_avg, finish_avg, elapsed_avg;
    double ms;
	
	printf("AVG\n");
	
	avg_step /= 4.0f;
    eig_step /= 4;
	if( obj_step == obj_size_width && eig_step == obj_size_width && avg_step == obj_size_width )
    {
		obj_size_width *= obj_size_height;
		obj_size_height = 1;
		obj_step = eig_step = avg_step = obj_size_width;
	}	
	avg_size_width = eig_size_width = obj_size_width;
	avg_size_height = eig_size_height = obj_size_height;
	size_t avg_size_l = avg_size_height*avg_size_width*sizeof(float); //tamaño de la matriz
    
	
	/* Calculation of averaged object */
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] = 0.f;
    
     
    //    CPU   
    #ifdef __AVG_SECUENCIAL__
    float m=0;
    m = 1.0f/(float)nObjects; //para ahorrar calculos
    
    gettimeofday(&start_avg, NULL);

    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        unsigned char *bu = objs[i];
		
        bf = avg_data;
        assert(avg_step == obj_step);
        for( k = 0; k < avg_size_height; k++, bf += avg_step, bu += avg_step )
            for( l = 0; l < avg_size_width; l++ )
                bf[l] += bu[l];
    }
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] *= m; 
            
	gettimeofday(&finish_avg, NULL);
	timersub(&finish_avg, &start_avg, &elapsed_avg);
    
    ms = elapsed_avg.tv_usec / 1000.0 + elapsed_avg.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-serial la avgMat para nObjects = %d y obj_size = %lu \n",ms,nObjects,avg_size_l);	
    #endif
    
       
    //GPU 
    #ifdef __AVG_PARALELO__
    size_t obj_size_l = nObjects*avg_size_l;
    
    gettimeofday(&start_avg, NULL);
        
    CHECK_CUDA_CALL(cudaMalloc(&avg_gpu,avg_size_l ));
    CHECK_CUDA_CALL(cudaMemset(avg_gpu,0,avg_size_l));

    CHECK_CUDA_CALL(cudaMalloc(&objs_gpu,obj_size_l ));
    CHECK_CUDA_CALL(cudaMemset(objs_gpu,0,obj_size_l));
    
    printf("Inicializamos los datos en la placa\n");
    
    for (int i=0;i<nObjects;i++)
		CHECK_CUDA_CALL(cudaMemcpy(objs_gpu + i*avg_size_l,objs[i], avg_size_l, cudaMemcpyDefault));
     
	/*	– cudaMemcpy(dest,src,size,mode);
		– Para copiar hacia la gpu:
			• Mode: cudaMemcpyHostToDevice
		– Para copiar desde la gpu:
			• Mode: cudaMemcpyDeviceToHost */

    avg_sum(avg_gpu,objs_gpu,(unsigned int) avg_size_l,(unsigned int) avg_size_width,(unsigned int) avg_size_height, nObjects);

	gettimeofday(&finish_avg, NULL);
	timersub(&finish_avg, &start_avg, &elapsed_avg);
    
    ms = elapsed_avg.tv_usec / 1000.0 + elapsed_avg.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-paralelo la avgMat para nObjects = %d y obj_size = %lu \n",ms,nObjects,avg_size_l);	
    CHECK_CUDA_CALL(cudaMemcpy(avg_data,avg_gpu, avg_size_l, cudaMemcpyDefault));
    
    
    /*
	float* avg_data2;
	CHECK_CUDA_CALL(cudaMallocHost(&avg_data2,avg_size_l ));
    CHECK_CUDA_CALL(cudaMemcpy(avg_data2,avg_gpu, avg_size_l, cudaMemcpyDefault));
	m = 1.0f/(float)nObjects;
       
	for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        unsigned char *bu = objs[i];
		
        bf = avg_data;
        assert(avg_step == obj_step);
        for( k = 0; k < avg_size_height; k++, bf += avg_step, bu += avg_step )
            for( l = 0; l < avg_size_width; l++ )
                bf[l] += bu[l];
    }
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] *= m;
	
	printf("Comparamos\n");
	float *bu = avg_data2; 
	bf = avg_data;
    for(int k = 0; k < avg_size_height; k++, bf += avg_step, bu += avg_step )
		for(int l = 0; l < avg_size_width; l++ )
			if(bf[l] != bu[l])
				printf("%f != %f\n",bf[l],bu[l]);
    */
	#endif
	
	int* arr = (int*)malloc(3*sizeof(int));
	arr[0] = 0;
	arr[1] = 0;
	arr[2] = 0;

		
	arr[0] = avg_step;
	arr[1] = obj_size_width;
	arr[2] = obj_size_height;
	
    //CHECK_CUDA_CALL(cudaFree(avg_gpu));
    //CHECK_CUDA_CALL(cudaFree(objs_gpu));
	
	return arr;
}

/*
void multsec(float *A, float *B, float *C, int N, int M)
{
  float sum = 0; // A * B = C (CPU)
  
  for(int i = 0; i < N; i++ )
    for(int j = 0; j < N; j++ ) {
      for(int k = 0; k < M; k++ ) {
         float a = A[i*M + k];
         float b = B[k*M + j];
         sum += a*b;    
        }
         C[i*N + j] = sum;
        sum = 0;
      }
}
*/
void multsec(float beta,float *A, float *B, float *C, int N, int M, int U) //La posta
{
  float sum = 0; // A * B = C (CPU)
  
  for(int i = 0; i < N; i++ )
    for(int j = 0; j < M; j++ ) {
      for(int k = 0; k < U; k++ ) {
         float a = A[i*U + k];
         float b = B[k*M + j];
         sum += a*b;    
        }
        C[i*N + j] = beta*sum;
        sum = 0;
      }
}
/*
void multsecV(float *A, float *B, float *C, int N, int M)
{
  float sum = 0; // A * B = C (CPU)
  
  for(int i = 0; i < N; i++ )
      for(int k = 0; k < M; k++ ) {
         float a = A[i*M + k];
         float b = B[k];
         sum += a*b;    
        }
         C[i*N] = sum;
        sum = 0;
}
*/

void multsecT(float beta,float *A, float *B, float *C, int N, int M, int U)
{
  float sum = 0; // A * B = C (CPU)
  
  for(int i = 0; i < N; i++ )
    for(int j = 0; j < M; j++ ) {
      for(int k = 0; k < U; k++ ) {
         float a = A[i*U + k];
         float b = B[j*M + k];//B[k*M + j];
         sum += a*b;    
        }
         C[i*N + j] = sum;
        sum = 0;
      }
}

void multsecT2(unsigned char *A,float *avg, float *C, int N,int M)
{
   for(int i = 0; i < N; i++ )
    {
		//uchar *bu = objects[i];

		for(int j = i; j < N; j++ )
		{
			size_t img_l = M*sizeof(float);
			float w = 0.f;
			float *a = avg;
			uchar *bu1 = (A + i*img_l);
			uchar *bu2 = (A + j*img_l);

			for(int k = 0; k < M; k++)
			{ //falta el -3
				
					float f = a[k];
                    float u1 = bu1[k];
                    float u2 = bu2[k];
                    w += (u1 - f) * (u2 - f);
				              
			}

            C[i * N + j] = C[j * N + i] = w;
        }
    }

}

void PCAgpu_cov(int nObjects, uchar** objects, float *avg, int obj_step,int size_width,int size_height, float *covarMatrix)
{
	
	struct timeval start_cov, finish_cov, elapsed_cov;
    double ms;
	
	#ifdef __COV_SECUENCIAL__
	int i, j;

	gettimeofday(&start_cov, NULL);
	for( i = 0; i < nObjects; i++ )
    {
		uchar *bu = objects[i];

		for( j = i; j < nObjects; j++ )
		{
			int k, l;
			float w = 0.f;
			float *a = avg;
			uchar *bu1 = bu;
			uchar *bu2 = objects[j];

			for( k = 0; k < size_height; k++, bu1 += obj_step, bu2 += obj_step, a += obj_step )
			{
				for( l = 0; l < size_width - 3; l++) 
				{
					float f = a[l];
                    float u1 = bu1[l];
                    float u2 = bu2[l];
                    w += (u1 - f) * (u2 - f);
				}
                
			}

            covarMatrix[i * nObjects + j] = covarMatrix[j * nObjects + i] = w;
        }
    }
    gettimeofday(&finish_cov, NULL);
	timersub(&finish_cov, &start_cov, &elapsed_cov);
    
    ms = elapsed_cov.tv_usec / 1000.0 + elapsed_cov.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-serial la covMat\n",ms);	
	
	#endif
	#ifdef __COV_PARALELO__
    //float * avg_gpu;
    size_t avg_size_l = size_height*size_width*sizeof(float); //tamaño de la matriz
    //unsigned char* objs_gpu;
    //size_t obj_size_l = nObjects*avg_size_l;
    float * covarMatrix_gpu;
    size_t cov_size_l = sizeof( float ) * nObjects * nObjects ;
    
    //CHECK_CUDA_CALL(cudaMalloc(&avg_gpu,avg_size_l ));
    //CHECK_CUDA_CALL(cudaMemset(avg_gpu,0,avg_size_l));

    //CHECK_CUDA_CALL(cudaMalloc(&objs_gpu,obj_size_l ));
    //CHECK_CUDA_CALL(cudaMemset(objs_gpu,0,obj_size_l));
    
    CHECK_CUDA_CALL(cudaMalloc(&covarMatrix_gpu,cov_size_l));
    CHECK_CUDA_CALL(cudaMemset(covarMatrix_gpu,0,cov_size_l));	
    
    printf("Inicializamos los datos en la placa\n");
    //CHECK_CUDA_CALL(cudaMemcpy(avg_gpu,avg, avg_size_l, cudaMemcpyDefault));

    //for (int i=0;i<nObjects;i++)
		//CHECK_CUDA_CALL(cudaMemcpy(objs_gpu + i*avg_size_l,objects[i], avg_size_l, cudaMemcpyDefault));


    /*	– cudaMemcpy(dest,src,size,mode);
		– Para copiar hacia la gpu:
			• Mode: cudaMemcpyHostToDevice
		– Para copiar desde la gpu:
			• Mode: cudaMemcpyDeviceToHost */
    gettimeofday(&start_cov, NULL);
    printf("Ejecutamos el kernel\n");
    //cov(avg_gpu,objs_gpu,covarMatrix_gpu,obj_step,nObjects,size_width,size_height); 
    covarOp(objs_gpu,avg_gpu, covarMatrix_gpu, nObjects,size_width*size_height);
	
	float* covarMatrix2 = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    //CHECK_CUDA_CALL(cudaMemcpy(covarMatrix2,covarMatrix_gpu, cov_size_l, cudaMemcpyDefault));
    CHECK_CUDA_CALL(cudaMemcpy(covarMatrix,covarMatrix_gpu, cov_size_l, cudaMemcpyDefault));
    
    gettimeofday(&finish_cov, NULL);
	timersub(&finish_cov, &start_cov, &elapsed_cov);
    
    ms = elapsed_cov.tv_usec / 1000.0 + elapsed_cov.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-paralelo la covMat para nObjects = %d y obj_size = %lu \n",ms,nObjects,avg_size_l);
    
   /*
    for(int i = 0; i < nObjects; i++ )
    {
		//uchar *bu = objects[i];

		for(int j = i; j < nObjects; j++ )
		{
			float w = 0.f;
			float *a = avg;
			uchar *bu1 = objects[i];
			uchar *bu2 = objects[j];

			for(int k = 0; k < size_height; k++, bu1 += obj_step, bu2 += obj_step , a += obj_step )
			{
				for(int l = 0; l < size_width ; l++)// -3
				{
					float f = a[l];
                    float u1 = bu1[l];
                    float u2 = bu2[l];
                    w += (u1 - f) * (u2 - f);
				}
                
			}

            covarMatrix[i * nObjects + j] = covarMatrix[j * nObjects + i] = w;
        }
    }
  
	size_t obj_size_l = nObjects*avg_size_l;
    printf("obj step = %d\n",obj_step);

    float* bf = covarMatrix;
    float* bf2 = covarMatrix2;
    printf("Comparamos matrices de covarianza\n");

    printf("num = %d \n", nObjects);
    for(int i = 0; i < nObjects-300; i++ )
		for(int j = 0; j < nObjects-300; j++ )
			if(bf2[i * nObjects + j] != bf[i * nObjects + j])
				printf("%f != %f\n",bf[i * nObjects + j],bf2[i * nObjects + j]);
   */
	
  //CHECK_CUDA_CALL(cudaFree(avg_gpu));
    //CHECK_CUDA_CALL(cudaFree(objs_gpu));   
    CHECK_CUDA_CALL(cudaFree(covarMatrix_gpu)); 
	#endif
    
}    
    

void PCAgpu_d(int nObjects, uchar** input,float** output,
					int eig_step,int eig_size_width,int eig_size_height,float* avg,float* eigVals, float* ev)
{
	float epsilon=0.99; // ultimo -> 0.0011; 250 -> 0.002165
	int p, l, i;
	float* eAcum = (float*)malloc(numEigens*sizeof(float));
	float div;
	int nEigens = numEigens; //nObjects - 1;
	int m1= nEigens; //max iter
	 /* Eigen objects number determination */

	/*calculamos la energia acumulativa de cada autovector (wikipedia)*/
	float acum = 0;
	
	for( i = 0; i < m1; i++ ) {
		acum += eigVals[i];
		eAcum[i] = acum;
	}	
	
    for( i = 0; i < m1; i++ ) {
		div = fabs( eAcum[i] / eAcum[m1-1] );
		if( div >= epsilon) {
			printf("div = %f\n",div);
			break;
		}
	}
	
	m1 = i;
	printf("calculado m1 = %d\n",m1);
	//m1=nEigens; //-------------> POR AHORA PARA TRABAJAR COMODOS -------------------------------
	printf("m1 = %d\n",m1);
	
	numEigens = m1;
    //epsilon = (float) fabs( eigVals[m1 - 1] / eigVals[0] );
    //printf("ep = %f\n",epsilon);

    for( i = 0; i < m1; i++ )
        eigVals[i] = (float) (1.0f / sqrt( (double)eigVals[i] ));
        

    for( i = 0; i < m1; i++ )       /* e.o. annulation */
    {
		float *be = output[i];

        for( p = 0; p < eig_size_height; p++, be += eig_step )
			for( l = 0; l < eig_size_width; l++ )
				be[l] = 0.0f;
    }

	struct timeval start_d, finish_d, elapsed_d;
    double ms;
	
	#ifdef __D_SECUENCIAL__
	int k;
	float *bf = 0;
	
	gettimeofday(&start_d, NULL);
	for( k = 0; k < nObjects; k++ )
	{
		for( i = 0; i < m1; i++ )
		{
			float v = eigVals[i] * ev[i * nObjects + k];
			float *bw = output[i];
			uchar *bu = input[k];

			bf = avg;
			
			for(int p = 0; p < eig_size_height; p++, bu += eig_step, bf += eig_step, bw += eig_step )
				for(int l = 0; l < eig_size_width-3; l ++ )
				{
					float f = bf[l];
					float u = bu[l];			
					bw[l] += v * (u - f);
				}				
		}                  
	}                       
	
	for( i = 0; i < m1; i++ )
		eigVals[i] = 1.0f / (eigVals[i] * eigVals[i]);
	
	gettimeofday(&finish_d, NULL);
	timersub(&finish_d, &start_d, &elapsed_d);
    
    ms = elapsed_d.tv_usec / 1000.0 + elapsed_d.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-secuencial _d \n",ms);
	
	#endif
	
	#ifdef __D_PARALELO__
	
    size_t avg_size_l = eig_size_height*eig_size_width*sizeof(float); //tamaño de la matriz
    float* eigsVect_gpu;
    size_t eigsVect_gpu_l = nEigens*avg_size_l;
    float* eigVals_gpu;
    size_t eigVals_gpu_l = nEigens*sizeof(float);
    float* ev_gpu;
    size_t ev_gpu_l = sizeof( float ) * nObjects * nObjects ;
    
    CHECK_CUDA_CALL(cudaMalloc(&eigsVect_gpu,eigsVect_gpu_l));
    CHECK_CUDA_CALL(cudaMemset(eigsVect_gpu,0,eigsVect_gpu_l));
    
    CHECK_CUDA_CALL(cudaMalloc(&eigVals_gpu,eigVals_gpu_l));
	CHECK_CUDA_CALL(cudaMemset(eigVals_gpu,0,eigVals_gpu_l));
	
	CHECK_CUDA_CALL(cudaMalloc(&ev_gpu,ev_gpu_l));
	CHECK_CUDA_CALL(cudaMemset(ev_gpu,0,ev_gpu_l));
	
	printf("Inicializamos los datos en la placa\n");

	gettimeofday(&start_d, NULL);

	CHECK_CUDA_CALL(cudaMemcpy(eigVals_gpu,eigVals,eigVals_gpu_l,cudaMemcpyDefault));
	CHECK_CUDA_CALL(cudaMemcpy(ev_gpu,ev,ev_gpu_l,cudaMemcpyDefault));
	
	for(i=0;i<nEigens;i++)
		CHECK_CUDA_CALL(cudaMemcpy(eigsVect_gpu + i*(eig_size_height*eig_size_width),output[i], avg_size_l, cudaMemcpyDefault));	
	
	printf("Kernel\n");	
	
	eig(m1,nObjects,objs_gpu,eigsVect_gpu,avg_gpu,eigVals_gpu,ev_gpu,eig_size_height,eig_size_width,eig_step);
	
	eigV(m1,eigVals_gpu);	
	
	for(i=0;i<nEigens;i++) 		
		CHECK_CUDA_CALL(cudaMemcpy(output[i],eigsVect_gpu + i*eig_size_height*eig_size_width, avg_size_l, cudaMemcpyDefault));
	
	gettimeofday(&finish_d, NULL);
	timersub(&finish_d, &start_d, &elapsed_d);
    
    ms = elapsed_d.tv_usec / 1000.0 + elapsed_d.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-paralelo _d \n",ms);

	//------------
	/*
	printf("Secuencial para comparar \n");
	int k;
	float *bf = 0;
	for( k = 0; k < nObjects; k++ )
	{
		for( i = 0; i < m1; i++ )
		{
			float v = eigVals[i] * ev[i * nObjects + k];
			float *bw = output[i];
			uchar *bu = input[k];

			bf = avg;
			
			for(int p = 0; p < eig_size_height; p++, bu += eig_step, bf += eig_step, bw += eig_step )
				for(int l = 0; l < eig_size_width-3; l ++ )
				{
					float f = bf[l];
					float u = bu[l];			
					bw[l] += v * (u - f);
				}				
		}                  
	}                      
	
	for( i = 0; i < m1; i++ )
		eigVals[i] = 1.0f / (eigVals[i] * eigVals[i]);

	float *output2;
	CHECK_CUDA_CALL(cudaMallocHost(&output2,sizeof(float)*nEigens*eig_size_height*eig_size_width));
	
	for(i=0;i<nEigens;i++) 		
		CHECK_CUDA_CALL(cudaMemcpy(output2 + i*eig_size_height*eig_size_width,eigsVect_gpu + i*eig_size_height*eig_size_width, avg_size_l, cudaMemcpyDefault));	
	
	
	float **output3 = (float **) cvAlloc( sizeof( float * ) * nEigens );	
	for(int i = 0; i < nEigens; i++ )
		output3[i] = (output2 + i*eig_size_height*eig_size_width);
        
	printf("Comparamos\n");
	
	printf("Comparamos EIG\n");
	for(int l=397; l < m1; l++){
		float* b1 = output[l];
		float* b2 = output3[l];
		for(int i = eig_size_height-5; i < eig_size_height; i++, b1 += eig_step , b2 += eig_step )
			for(int j = eig_size_width-8; j < eig_size_width-3; j++ )
				if(b1[j] != b2[j])
					printf("EIG %.10f =! %.10f\n",b1[j],b2[j]);
	}	
	*/
	/*
	float* ev2;
	CHECK_CUDA_CALL(cudaMallocHost(&ev2,ev_gpu_l));
	CHECK_CUDA_CALL(cudaMemcpy(ev2,ev_gpu, ev_gpu_l, cudaMemcpyDefault));
	for(int i=0; i< nObjects; i++)
		for(int j=0; j< nObjects;j++)
			if(ev[j+i*nObjects] != ev2[j+i*nObjects])
				printf("%f != %f\n",ev[j+i*nObjects],ev2[j+i*nObjects]);
	*/
	/*
	uchar* obj2;
	CHECK_CUDA_CALL(cudaMallocHost(&obj2,sizeof(uchar)*nObjects*avg_size_l));
	for(int i=0;i<nObjects;i++)
		CHECK_CUDA_CALL(cudaMemcpy(obj2 + i*avg_size_l,(uchar*)objs_gpu + i*avg_size_l, avg_size_l, cudaMemcpyDefault));
		
	uchar** obj3 = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
	for(int i = 0; i < nObjects; i++ )
        obj3[i] = (obj2 + i*avg_size_l);
    
	printf("Comparamos OBJ\n");
	for(int l=0; l < nObjects; l++){
		uchar* b1 = input[l];
		uchar* b2 = obj3[l];
		for(int i = 0; i < eig_size_height; i++, b1 += eig_step , b2 += eig_step )
			for(int j = 0; j < eig_size_width; j++ )
				if(b1[j] != b2[j])
					printf("OBJ %d =! %d\n",b1[j],b2[j]);
	}
	*/

/*		
	float* eigVals2;
	CHECK_CUDA_CALL(cudaMallocHost(&eigVals2,eigVals_gpu_l));
	CHECK_CUDA_CALL(cudaMemcpy(eigVals2,eigVals_gpu,eigVals_gpu_l,cudaMemcpyDefault));	
	for( i = 0; i < m1; i++ )
		if(eigVals[i] != eigVals2[i])
			printf("EIGVV %f =! %f\n",eigVals[i],eigVals2[i]);
*/

	CHECK_CUDA_CALL(cudaFree(avg_gpu));
    CHECK_CUDA_CALL(cudaFree(objs_gpu));   
    CHECK_CUDA_CALL(cudaFree(eigsVect_gpu)); 
	CHECK_CUDA_CALL(cudaFree(eigVals_gpu)); 
	CHECK_CUDA_CALL(cudaFree(ev_gpu));
	#endif
}
  
    
    
    
    
