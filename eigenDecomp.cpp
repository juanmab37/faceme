#include "helpers_cuda.h"
#include "decomp.h"
#include "kernel_decomp.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>



void EigenDecomposite_gpu(void* objInput,int nEigObjs,void* eigInput,IplImage* avg,float* coeffs,int offset)
{
	int nObjects = numCarasEntrenamiento, i;
    
    float *avg_data;
    int avg_step = 0;
    CvSize avg_size;
    
    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
   
    IplImage **objects = (IplImage **) (((CvInput *) & objInput)->data);
	IplImage **eigens = (IplImage **) (((CvInput *) & eigInput)->data);

	float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigObjs );
	uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
	
	int eig_step = 0, obj_step = 0;
	CvSize eig_size = avg_size, obj_size = avg_size;
	
	for( i = 0; i < nObjects; i++ )
    {
        IplImage *img = objects[i];
        uchar *obj_data;

        cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
       
        objs[i] = obj_data;
    }
	for( i = 0; i < nEigObjs; i++ )
	{
		IplImage *eig = eigens[i];
		float *eig_data;

		cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );

		eigs[i] = eig_data;
	}
	
	EigenDecomposite_gpu_i( objs,obj_step,nEigObjs, eigs,avg_data,obj_size.height,obj_size.width,coeffs,offset);
	cvFree( &eigs );
}

void EigenDecomposite_gpu_i(uchar** obj,int objStep,int nEigObjs,float **eigInput,float *avg,
								int size_height,int size_width,float *coeffs,int offset)
{
	float w = 0.0f;
	int nObjects = (nEigObjs+1);
	if( size_width == objStep)
    {
        size_width *= size_height;
        size_height = 1;
        objStep = size_width;
    }
    
    struct timeval start_d, finish_d, elapsed_d;
    double ms;
    
    #ifdef __DECOMP_SECUENCIAL__	
	gettimeofday(&start_d, NULL);
	for(int l = 0; l < nObjects; l++)
		for(int  j = 0; j < nEigObjs; j++ )
		{
			w = 0.0f;
			uchar* bo = obj[l];
			float* be = eigInput[j];
			float* ba = avg;
			
			for(int i = 0; i < size_height; i++, bo += objStep, be += objStep, ba += objStep )
			{
				for(int k = 0; k < size_width - 3; k ++ )
				{
					float o = (float) bo[k];
					float e = be[k];
					float a = ba[k];
					w += e * (o - a);
				}
			}
			coeffs[j+l*offset] = w;
		 }
		 
	gettimeofday(&finish_d, NULL);
	timersub(&finish_d, &start_d, &elapsed_d);
    
    ms = elapsed_d.tv_usec / 1000.0 + elapsed_d.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-secuencial \n",ms);	 
	#endif
	
    #ifdef __DECOMP_PARALELO__
    float* avg_gpu;
    float* eigInput_gpu;
    float* coeffs_gpu;
    size_t avg_size_l = size_height*size_width*sizeof(float); //tamaño de la matriz
    unsigned char* objs_gpu;
    size_t obj_size_l = nObjects*avg_size_l;
    size_t eigInput_size_l = nEigObjs*avg_size_l;
    size_t coeffs_size_l = sizeof( float ) * nObjects * nEigObjs;
    
    CHECK_CUDA_CALL(cudaMalloc(&avg_gpu,avg_size_l ));
    CHECK_CUDA_CALL(cudaMemset(avg_gpu,0,avg_size_l));

    CHECK_CUDA_CALL(cudaMalloc(&objs_gpu,obj_size_l ));
    CHECK_CUDA_CALL(cudaMemset(objs_gpu,0,obj_size_l));
    
    CHECK_CUDA_CALL(cudaMalloc(&eigInput_gpu,eigInput_size_l ));
    CHECK_CUDA_CALL(cudaMemset(eigInput_gpu,0,eigInput_size_l));

    CHECK_CUDA_CALL(cudaMalloc(&coeffs_gpu,coeffs_size_l ));
    CHECK_CUDA_CALL(cudaMemset(coeffs_gpu,0,coeffs_size_l));	    		
   
    printf("Inicializamos los datos en la placa\n");
    gettimeofday(&start_d, NULL);
    //borrar prueba
    float* coeffs_gpu5;
	CHECK_CUDA_CALL(cudaMalloc(&coeffs_gpu5,coeffs_size_l ));
    CHECK_CUDA_CALL(cudaMemset(coeffs_gpu5,0,coeffs_size_l));
    CHECK_CUDA_CALL(cudaMemcpy(coeffs_gpu5,coeffs, coeffs_size_l, cudaMemcpyDefault));
    //---
    CHECK_CUDA_CALL(cudaMemcpy(avg_gpu,avg, avg_size_l, cudaMemcpyDefault));
    CHECK_CUDA_CALL(cudaMemcpy(coeffs_gpu,coeffs, coeffs_size_l, cudaMemcpyDefault));    

    for (int i=0;i<nObjects;i++)
		CHECK_CUDA_CALL(cudaMemcpy(objs_gpu + i*avg_size_l,obj[i], avg_size_l, cudaMemcpyDefault));

    for (int i=0;i<nEigObjs;i++)
		CHECK_CUDA_CALL(cudaMemcpy(eigInput_gpu + i*size_height*size_width,eigInput[i], avg_size_l, cudaMemcpyDefault));		
	
	printf("KERNEL DECOMPOSITE INIC\n");
	
	decomp(coeffs_gpu,avg_gpu,objs_gpu,eigInput_gpu,nObjects,nEigObjs,objStep,size_height,size_width,offset);

	CHECK_CUDA_CALL(cudaMemcpy(coeffs,coeffs_gpu, coeffs_size_l, cudaMemcpyDefault));
	
	decompOp(coeffs_gpu5,avg_gpu,objs_gpu,eigInput_gpu,nObjects,nEigObjs,objStep,size_height,size_width,offset);
	
	float* coeffs2 = (float *) cvAlloc( sizeof( float ) * nObjects * nEigObjs );
	CHECK_CUDA_CALL(cudaMallocHost(&coeffs2,coeffs_size_l));
	CHECK_CUDA_CALL(cudaMemcpy(coeffs2,coeffs_gpu5, coeffs_size_l, cudaMemcpyDefault));
	printf("Comparamos\n");
	for(int i=0; i< nObjects; i++)
		for(int j=0; j< nEigObjs;j++)
			if(coeffs[j+i*offset] != coeffs2[j+i*offset])
				printf("%f != %f\n",coeffs[j+i*offset],coeffs2[j+i*offset]);
	
				
	
	//float* avg2;
	//CHECK_CUDA_CALL(cudaMallocHost(&avg2,avg_size_l ));
    CHECK_CUDA_CALL(cudaMemcpy(avg,avg_gpu, avg_size_l, cudaMemcpyDefault));
 /*  
    printf("Comparamos AVG\n");
	float* b1 = avg;
	float* b2 = avg2;
	for(int i = 0; i < size_height; i++, b1 += objStep , b2 += objStep )
        for(int j = 0; j < size_width; j++ )
            if(b1[j] != b2[j])
				printf("AVG %f =! %f\n",b1[j],b2[j]);
*/				
	
	
	//uchar* obj2;
	//CHECK_CUDA_CALL(cudaMallocHost(&obj2,sizeof(uchar)*nObjects*avg_size_l));
	for(int i=0;i<nObjects;i++)
		CHECK_CUDA_CALL(cudaMemcpy(obj[i],(uchar*)objs_gpu + i*avg_size_l, avg_size_l, cudaMemcpyDefault));
		
	//uchar** obj3 = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
	//for(int i = 0; i < nObjects; i++ )
     //   obj[i] = (obj2 + i*avg_size_l);
    /*
	printf("Comparamos OBJ\n");
	for(int l=0; l < nObjects; l++){
		uchar* b1 = obj[l];
		uchar* b2 = obj3[l];
		for(int i = 0; i < size_height; i++, b1 += objStep , b2 += objStep )
			for(int j = 0; j < size_width; j++ )
				if(b1[j] != b2[j])
					printf("OBJ %d =! %d\n",b1[j],b2[j]);
	}
	*/
	//float* eig2;
	//CHECK_CUDA_CALL(cudaMallocHost(&eig2,sizeof(float)*nEigObjs*size_height*size_width));
	for(int i=0;i<nEigObjs;i++)
		CHECK_CUDA_CALL(cudaMemcpy(eigInput[i],eigInput_gpu + i*size_height*size_width, avg_size_l, cudaMemcpyDefault));
		
		
	//float **eig3 = (float **) cvAlloc( sizeof( float * ) * nEigObjs );	
	//for(int i = 0; i < nEigObjs; i++ )
      //  eigInput[i] = (eig2 + i*size_height*size_width);
	/*
	printf("Comparamos EIG\n");
	for(int l=0; l < nEigObjs; l++){
		float* b1 = eigInput[l];
		float* b2 = eig3[l];
		for(int i = 0; i < size_height; i++, b1 += objStep , b2 += objStep )
			for(int j = 0; j < size_width; j++ )
				if(b1[j] != b2[j])
					printf("EIG %f =! %f\n",b1[j],b2[j]);
	}
	*/
	gettimeofday(&finish_d, NULL);
	timersub(&finish_d, &start_d, &elapsed_d);
    
    ms = elapsed_d.tv_usec / 1000.0 + elapsed_d.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-paralelo \n",ms);	
	
	#endif
}

