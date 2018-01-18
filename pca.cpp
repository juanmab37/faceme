#include "pca.h"
#include <time.h>
#include <sys/time.h>

/*------------------PRINCIPAL COMPONENTS ANALYSIS------------------*/
void PCA()
{
	int i;
	/*Estructuras*/
	CvSize tamanoImgCaras;

	/*Cantidad de AutoValores a usar; este es el máximo:*/
	numEigens = numCarasEntrenamiento - 1;

	/*Ajustamos las dimensiones de cada imágen de caras */
	tamanoImgCaras.width  = arrCaras[0]->width;
	tamanoImgCaras.height = arrCaras[0]->height;
	
	arrAutoVectores = (IplImage**)cvAlloc(sizeof(IplImage*) * numEigens);
	arrAutoVectores2 = (IplImage**)cvAlloc(sizeof(IplImage*) * numEigens);
	
	/*Crea en cada celda (desde 0 hasta numEigens - 1) una imágen con tamaño de tamanoImgCaras*/	
	for (i = 0; i < numEigens; i++)
		arrAutoVectores[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);
		
	for (i = 0; i < numEigens; i++)
		arrAutoVectores2[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);

	/*Esta matriz va a alojar los AutoValores respectivos a cada AutoVector.
	Los AutoValores son de tipo Float.*/
	matAutoValores = cvCreateMat( 1, numEigens, CV_32FC1 );
	matAutoValores2 = cvCreateMat( 1, numEigens, CV_32FC1 );

	
	pAvgTrainImg = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);
	pAvgTrainImg2 = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);


	/*Compute average image, eigenvalues, and eigenvectors (this means that'll compute a basis).
	Calcula el subespacio para las caras de entrenamiento*/
	PCAgpu(numCarasEntrenamiento, (void*)arrCaras, (void*)arrAutoVectores, pAvgTrainImg, matAutoValores->data.fl);
	
	/*printf("Secuencial\n");
	CvTermCriteria calcLimit2;
	calcLimit2 = cvTermCriteria( CV_TERMCRIT_ITER, numEigens, 1);
	
	printf("Secuencial Inicio\n");
	cvCalcEigenObjects(
		numCarasEntrenamiento,              
		(void*)arrCaras,                   //(input) Donde están guardadas las caras a quienes les calculamos los AutoVectores y AutoValores.
		(void*)arrAutoVectores,            //(output) Donde los guardamos los AutoVectores.
		CV_EIGOBJ_NO_CALLBACK,              //input/output flags
		0,
		0,
		&calcLimit2,                         //CvTermCriteria const.
		pAvgTrainImg,                       //Guarda la imágen promedio en pAvgTrainImg.
		matAutoValores->data.fl);
	printf("Secuencial Fin\n");	
	*/
	cvNormalize(matAutoValores, matAutoValores, 1, 0, CV_L1, 0);
	
	return;
}
void imprimirMatFloat(CvMat* mat)
{
	int alto = mat->rows;
	int ancho = mat-> cols;
	
	printf("tipo: %d\n",mat->type);
	printf("step: %d\n",mat->step);
	printf("row: %d\n",mat->rows);
	printf("cols: %d\n",mat->cols);
	printf("height: %d\n",mat->height);
	printf("width: %d\n",mat->width);
	
	float* dat = mat->data.fl;
	int i,j;
	for (i=0;i<ancho;i++)
		for (j=0;j<alto;j++)
			printf("%f ",dat[i * alto + j]);
			

	printf("\n");
}
	
void imprimirMat(IplImage* AvgImg)
{
	float *avg_data;
	int avg_step = 0,i;
	CvSize avg_size;
   
	cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );
	avg_step /= 4;

	for( i = 0; i < avg_size.height; i++, avg_data += avg_step )
		for(int j = 0; j < avg_size.width; j++ )
			printf("%f ",avg_data[j]);
}	

void imprimirArrImag(IplImage** arrImg,int n)
{
	for(int i=0; i< n;i++) {
		imprimirMat(arrImg[i]);
		printf("\n");	
	}
}

void PCAgpu(int nObjects, void* input, void* output, IplImage* AvgImg, float* eigVals) {
	//Variables
	int i;
	int nEigens = nObjects - 1;
	 

    IplImage **arrOutput = (IplImage **) (((CvInput *) & output)->data);
	IplImage **arrInput = (IplImage **) (((CvInput *) & input)->data);

	float *avg_data;
    int avg_step = 0;
    CvSize avg_size;

    cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );

	int obj_step = 0, eig_step = 0;
	
	CvSize obj_size = avg_size, eig_size = avg_size;

    float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigens );	
	uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );

	for( i = 0; i < nObjects; i++ )
    {
        IplImage *img = arrInput[i];
        uchar *obj_data;

        cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
       
        objs[i] = obj_data;
    }
    for( i = 0; i < nEigens; i++ )
    {
        IplImage *eig = arrOutput[i];
        float *eig_data;

        cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );

        eigs[i] = eig_data;
    }    
//nada	

//---
	int* arr;
	arr = PCAgpu_i( nObjects, objs, obj_step, eig_step, obj_size.width,obj_size.height, avg_data, avg_step); 
    
    avg_step = arr[0];
	eig_step = avg_step;
	eig_size.width = arr[1];
	eig_size.height = arr[2];
	
/*	//Prueba

	float MA[3*3] = {3,-2,3,0,3,5,4,4,4};//{4,2,5,8,6,7,1,9,5};//{12,-51,4,6,167,-68,-4,24,-41};
	//float *MQ = (float *) cvAlloc( sizeof( float  ) * 3*3 );
	float MQ[9] = {0};
	
	for(int i=0; i<9; i++)
		printf("[%d] = %f\n",i,MA[i]);
	
	int r1 = householder2(MA, MQ, 3, 3);
	
	for(int i=0; i<9; i++)
		printf("[%d] = %f\n",i,MQ[i]);
	
	printf("r1 = %d -----------\n",r1);
//--------
	*/
	
	
	//printf("avg_size_width: %d \n avg_size_heigt: %d \n, size_width: %d \n, size_height: %d \n",avg_size.width,avg_size.height,arr[1],arr[2]);
    float* covarMatrix = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    PCAgpu_cov( nObjects, objs, avg_data, avg_step, arr[1],arr[2],covarMatrix );

//Medimos tiempo
    struct timeval start_jac, finish_jac, elapsed_jac;
    double ms;
    
	gettimeofday(&start_jac, NULL);
	 
	printf("Jacobi Inicio\n");
    float* ev = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    int err = PCAgpu_eig( covarMatrix, ev, eigVals, nObjects, 0.0f );
    printf("Jacobi Fin\n");

	gettimeofday(&finish_jac, NULL);
	timersub(&finish_jac, &start_jac, &elapsed_jac);
    
    ms = elapsed_jac.tv_usec / 1000.0 + elapsed_jac.tv_sec * 1000.0;
	printf("se tardó %.3lf ms en procesar-serial la Jacobi para nObjects = %d \n",ms,nObjects);	


//Fin tiempo
    
    assert(err != 1 && err != 2);
	cvFree( &covarMatrix );
	//printf("s %d, w %d, h %d\n",avg_step,arr[1],arr[2]);
	//printf("s %d, w %d, h %d\n",eig_step,eig_size.width,eig_size.height);
	
	PCAgpu_d(nObjects, objs,eigs,eig_step,eig_size.width,eig_size.height,avg_data,eigVals,ev);      
	printf("d Fin\n");
	cvFree( &ev );
    cvFree( &objs );
    cvFree( &eigs );
}



void impMatPun(float* Mat,int tam1,int tam2,char* nom)
{
	printf("%s\n",nom);
	for(int i = 0; i<tam1; i++)
		for(int j = 0; j<tam2; j++)
			printf(" [%d] = %f \n",i*tam2 + j,Mat[i*tam2 + j]);
}			

/*
int householder(float *A, float *Q, int n, int r, int eps)
{
	float* I = (float *) calloc( sizeof( float ) * n, sizeof( float ) * n );
	
	for(int i = 0; i < n; i++, I += n )
		I[i] = 1;
	
	Q = I;
	
	for(int k=0; k < (n/r); k++) //n multiplo de r poner ceil deps
	{
		float* V = (float *) calloc(n * r, sizeof( float ) );
		float* B = (float *) calloc(r,sizeof( float ) );

		
		float* Y = (float *) calloc( n * r, sizeof( float ) );
		float* W = (float *) calloc( n * r, sizeof( float ) );
		
		int s = (k-1)*r + 1; //filas
		for(int j = 0; j<r; j++) // j = columnas
		{
			int u = s + j - 1; // indice lineal 
			float* v = (float *) calloc( (n - u),sizeof(float)); //vector
			//float* vh;//vector
			float* b;//valor
			
			house(v,b,u,n); //creamos v, vh de tam n (con ceros inicialmente) y completamos d u a n
			
			float* VV = (float *) calloc((n - u) * (n - u), sizeof( float ));
 			multsec(b,v,v,VV,n-u,n-u,1); // VV = v * vh
 						
 			float* AA = (float *) calloc((n - u)* (s+r-1 - u), sizeof( float ) );
 			float* A_block  = (float *) calloc((n - u) * (s+r-1 - u), sizeof( float ) );
 			
 			for(int i=0; i<(n - u); i++)
				for(int j=0; j<(s+r-1 - u); j++)
					A_block[i*(s+r-1 - u) +j] = A[(i+u)*n + (u + j)];
 			
			multsec(1,VV,A_block,AA,(n-u),(s+r-1-u),n-u);//AA = (VV * A)  
			
			for(int q=u; q<=n;q++) {
				for(int p=u; p<s+r;p++)
					A[q*n+p] -= AA[(q-u)*(n-u)+(p-u)];  //
				V[q*n + j] = v[q];
			}
			 B[j] = b; 
		}
		//j = 0
		for(int i=0; i<n; i++) {
			Y[i*n + 1] = V[i*n + 1];
			W[i*n + 1] = -B[1]*V[i*n + 1];
		}
		for(int j=1; j<r; j++) {
			float *v = (float *) calloc(n, sizeof( float ));
			for(int q=0; q<n;q++)
				v[q] = V[q*n + j];
			float *z = (float *) calloc(n, sizeof( float ));
			float* WYh = (float *) calloc(n * n, sizeof( float ));
			multsecT(1,W, Y, WYh, n,n,r);//WYh = W*Yh ;
			float* WYhv = (float *) calloc(n, sizeof( float ));
			multsec2(1,WYh,v,WYhv,n,1,n);//WYh = WYh * v;
			for(int q=0; q<n;q++) {
				z[q] = -B[j]*v[q] - B[j]*WYhv[q];
				W[q*n + j] = z[q];
				Y[q*n + j] = v[q];
			}	 	
		}
		
		float* WYh = (float *) calloc(n * n, sizeof( float ));
		float* YWh = (float *) calloc(n * n, sizeof( float ));
		multsecT(1,Y, W, YWh, n,n,r);//YWh = Y*Wh ;
		multsecT(1,W, Y, WYh, n,n,r);//WYh = W*Yh ;
		
 		for(int i=0; i<(n - u); i++)
			for(int j=0; j<(s+r-1 - u); j++)
				A_block[i*(s+r-1 - u) +j] = A[(i+u)*n + u + j];		
				
		multsec(1,YWh,A_block,A,m-s,n-s+r,r); //YWh * A
		multsec(1,YWh,A_block,A,m-s,n-s+r,r); // Q * WYh
		//modificar estas mult para el bloque solamente ------> NO VA A SER FACIL	
		//mult por A a YWh solo el bloque

			 
	}
	return 0;
}
*/
/*
int householder2(float *A, float *output, int n, int r)
{
	memset(output,0,n*n* sizeof( float ));
	float* Q = output;
	for(int i = 0; i < n; i++, Q += n )
		Q[i] = 1;
	printf("1\n");
	Q=output;
	
	//definiciones
	float b;//valor
	float* v = (float *) calloc( n,sizeof(float)); //vector //calloc
	float* MM = (float *) malloc(n * n * sizeof( float )); //malloc
	float* AA = (float *) malloc(n* n * sizeof( float ) ); //malloc
	float* v2 = (float *) malloc(n * sizeof( float )); //malloc
	float* WYhv = (float *) malloc(n * sizeof( float )); //malloc
	float* QQ = (float *) malloc(n* n * sizeof( float ) );//malloc
	float* V = (float *) calloc(n * r , sizeof( float ) );//calloc
	float* B = (float *) malloc(r * sizeof( float ) );//malloc
	float* Y = (float *) malloc( n * r * sizeof( float ) ); //calloc
	float* W = (float *) malloc( n * r * sizeof( float ) );//calloc
	
	
	printf("2\n");
	for(int k=0; k < (n/r); k++) //n multiplo de r poner ceil deps
	{
		printf("2.2 -> k = %d\n",k);
		memset(Y,0,n * r * sizeof( float ));
		memset(W,0,n * r * sizeof( float )); 	
		printf("3\n");
		
		
		int s = k*r ; //filas
		for(int j = 0; j<r; j++) // j = columnas
		{
			int u = s + j; // indice lineal 
			printf("4\n");
			b = house(A,v,u,n); //creamos v, vh de tam n (con ceros inicialmente) y completamos d u a n
			impMatPun(v,1,n,"v");
			printf("beta = %f 5\n",b);
 			multsec(b,v,v,MM,n,n,1); // VV = v * vh
 			impMatPun(MM,n,n,"MM=v*v");
			multsec(1,MM,A,AA,n,n,n);//AA = (VV * A)  
			impMatPun(AA,n,n,"AA=MM*A");
			printf("6\n");
			//impMatPun(A,n,n,"A");
			for(int q=u; q<n;q++) {
				for(int p=u; p<s+r;p++)
					A[q*n+p] -= AA[q*n+p];  //
				V[q*n + j] = v[q];
			}
			impMatPun(A,n,n,"A");
			//impMatPun(V,n,r,"V");
			printf("7\n");
			B[j] = b; 
			v[u] = 0; //lugar no utilizado en la prox iter
		}
		//j = 0
		for(int i=0; i<n; i++) {
			Y[i*n + 0] = V[i*n + 0];
			W[i*n + 0] = -B[0]*V[i*n + 0];
		}
		for(int j=1; j<r; j++) {
			
			for(int q=0; q<n;q++)
				v2[q] = V[q*n + j];
			
			multsecT(1,W, Y, AA, n,n,r);//WYh(AA) = W*Yh ;
			multsec(1,AA,v2,WYhv,n,1,n);//WYhv = WYh(AA) * v;
	
			for(int q=0; q<n;q++) {
				W[q*n + j] = -B[j]*v2[q] - B[j]*WYhv[q];
				Y[q*n + j] = v2[q];
			}
				 	
		}
		printf("8\n");
		//impMatPun(W,n,r,"W");
		multsecT(1,W, Y, MM, n,n,r);//WYh = W*Yh ;
		multsec(1,Q,MM,QQ,n,n,n); // Q * WYh
		//impMatPun(QQ,n,n,"QQ = Q*MM");
		multsecT(1,Y, W, MM, n,n,r);//YWh = Y*Wh ;
		multsec(1,MM,A,AA,n,n,n); //YWh * A
		
		for(int q=s; q<n;q++) 
			for(int p=s+r; p<n;p++) 
				A[q*n+p] += AA[q*n+p]; 
		for(int q=0; q<n;q++) 
			for(int p=s; p<n;p++) 
				Q[q*n+p] += QQ[q*n+p]; 	
		//impMatPun(Q,n,n,"Q");	
		printf("8.2\n");			
	
		printf("8.3\n");			
	}
	printf("8.5\n");
	
	
	free(v);
	printf("a\n");
	free(MM);
	printf("b.5\n");
	free(AA);
	printf("c.5\n");
	free(v2);
	printf("d.5\n");
	free(WYhv);
	printf("e.5\n");
	free(QQ);
	printf("f.5\n");
	free(V);
	printf("g.5\n");
	free(B);
	printf("h.5\n");
	free(Y);
	free(W);	
	
	
	printf("9\n");
	
	
	return 0;
}

float prodEsc(float* v1, float* v2,int init, int n)
{
	float ret;
	for(int j=init;j<n;j++)
		ret += v1[j]*v2[j];
	return ret;
}		

float modulo(float *V, int n)
{
	return sqrt(prodEsc(V,V,0,n));
}	

float house(float *A,float *v,int u,int n)
{ //v -> vector, A -> Matriz n*n
	float b;
	float ti = 0;
	float mu;
	float x[n-u];  
	int len = n-u;
		
	for(int i=u; i<n; i++)
		x[i-u] =  A[i*n + u];
	impMatPun(x,len,1,"X");
	float modX = modulo(x,len);	

	//for(int i=u; i<n; i++)
		//x[i-u] /= modX;	
	//impMatPun(x,len,1,"X");
	
	//ti = prodEsc(x,x,0,len);
	//v[u] = 1;
	
	int sig;
	float* e = (float *) calloc( len,sizeof(float));
	e[0] = modX; 
	if(x[0] >= 0 )
		sig = 1;
	else
		sig = -1;
	
	for(int i=0; i<len; i++)
		v[i+u] = sig*e[i] + x[i]; 

	impMatPun(v,n,1,"v");
	
	modX = modulo(v,n);	
	
	for(int i=0; i<len; i++)
		v[i+u] /= modX ; 
	b=2;
	
	//ti = prodEsc(x,x,0,len);
	//if (ti ==0)
		//b = 0;
//	else
	//	b = 2/t1;
	
	
	//for(int i = u; i<n;i++)
		//v[i] = x[i-u];
	//impMatPun(v,n,1,"v");
	//if (ti ==0)
		//b = 0;
	//else
	//{
		//mu = sqrt(pow(x[0],2) + ti);
		//if(x[0] <= 0)
			//v[u] = x[0] - mu;
		//else
			//v[u] = -ti/(x[0]+mu);
		
		//float aux = pow(v[u],2);
		//b = 2*aux/(ti+ aux);
		//for(int k = u;k<n;k++)
			//v[k] /= v[u];
			
	//}	
		
	return b;
}


*/



//( covarMatrix,   ev,    eigVals, nObjects, 0.0f )
int PCAgpu_eig(float *A, float *V, float *E, int n, float eps) //metodo rotacion de Jacobi
{
    int i, j, k, ind;
    float *AA = A, *VV = V;
    double Amax, anorm = 0, ax;

    if( A == NULL || V == NULL || E == NULL )
        return 1;
    if( n <= 0 )
        return 2;
    if( eps < 1.0e-7f )
        eps = 1.0e-7f;
	
	//printf("A = %f\n",A[0]);
	
    /*-------- Prepare --------*/
    for( i = 0; i < n; i++, VV += n, AA += n )
    {
        for( j = 0; j < i; j++ )
        {
            double Am = AA[j];

            anorm += Am * Am;
        }
        for( j = 0; j < n; j++ )
            VV[j] = 0.f;
        VV[i] = 1.f;
    }

    anorm = sqrt( anorm + anorm );
    ax = anorm * eps / n;
    Amax = anorm;

    while( Amax > ax )
    {
        Amax /= n;
        do                      /* while (ind) */
        {
            int p, q;
            float *V1 = V, *A1 = A;

            ind = 0;
            for( p = 0; p < n - 1; p++, A1 += n, V1 += n )
            {
                float *A2 = A + n * (p + 1), *V2 = V + n * (p + 1);

                for( q = p + 1; q < n; q++, A2 += n, V2 += n )
                {
                    double x, y, c, s, c2, s2, a;
                    float *A3, Apq = A1[q], App, Aqq, Aip, Aiq, Vpi, Vqi;

                    if( fabs( Apq ) < Amax )
                        continue;

                    ind = 1;

                    /*---- Calculation of rotation angle's sine & cosine ----*/
                    App = A1[p];
                    Aqq = A2[q];
                    y = 5.0e-1 * (App - Aqq);
                    x = -Apq / sqrt( (double)Apq * Apq + (double)y * y );
                    if( y < 0.0 )
                        x = -x;
                    s = x / sqrt( 2.0 * (1.0 + sqrt( 1.0 - (double)x * x )));
                    s2 = s * s;
                    c = sqrt( 1.0 - s2 );
                    c2 = c * c;
                    a = 2.0 * Apq * c * s;

                    /*---- Apq annulation ----*/
                    A3 = A;
                    for( i = 0; i < p; i++, A3 += n )
                    {
                        Aip = A3[p];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A3[p] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < q; i++, A3 += n )
                    {
                        Aip = A1[i];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < n; i++ )
                    {
                        Aip = A1[i];
                        Aiq = A2[i];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A2[i] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    A1[p] = (float) (App * c2 + Aqq * s2 - a);
                    A2[q] = (float) (App * s2 + Aqq * c2 + a);
                    A1[q] = A2[p] = 0.0f;
                }               /*q */
            }                   /*p */
        }
        while( ind );
        Amax /= n;
    }                           /* while ( Amax > ax ) */

    for( i = 0, k = 0; i < n; i++, k += n + 1 ) {
        E[i] = A[k];
       // if(n == 0)
			//printf("Ev = %f\n",E[0]);
	}
 
    /*printf(" M = %d\n", M); */

    /* -------- ordering -------- */
    for( i = 0; i < n; i++ )
    {
        int m = i;
        float Em = (float) fabs( E[i] );

        for( j = i + 1; j < n; j++ )
        {
            float Ej = (float) fabs( E[j] );

            m = (Em < Ej) ? j : m;
            Em = (Em < Ej) ? Ej : Em;
        }
        if( m != i )
        {
            int l;
            float b = E[i];

            E[i] = E[m];
            E[m] = b;
            for( j = 0, k = i * n, l = m * n; j < n; j++, k++, l++ )
            {
                b = V[k];
                V[k] = V[l];
                V[l] = b;
            }
        }
    }

    return 0;
}


