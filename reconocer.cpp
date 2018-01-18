#include "funcs.h"



/*------------------RECONOCER------------------*/

void reconocer(char archivoNombresImg[512])
{
	printf("Reconociendo...\n");
	int i;
	int numCarasAReconocer = 0; //Cantidad de caras a reconocer
	int caraEntrenamientoCercana, cercana,cercana2;
	CvMat * trainPersonNumMat = 0; //The person numbers during training
	float * imgAReconocerProyectada = 0;
	CvSize tamanoImgCaras;
	

	/*Primer etapa:
	Cargar las caras de las cuales queremos reconocer.
	Es lo mismo que hicimos con las de entrenar*/
	
	numCarasAReconocer = cargarArrayImgCaras_reconocer(archivoNombresImg);
	
	printf("Caras a reconocer --> %d\n",numCarasAReconocer);
		
	/*Cargamos la informacion guardada del entrenamiento (el XML)*/
	if( !cargarDatosEntrenamiento( &trainPersonNumMat ) ) 
		return;

	/*Segunda etapa:
	Proyectamos las imágenes a reconocer en el subespacio de PCA
	y localiza la imágen de entrenamiento más cercana. (&)*/
	imgAReconocerProyectada = (float *)cvAlloc(numEigens*sizeof(float));
		
	for(i=0; i< numCarasAReconocer; i++) {
		/*(&)*/
		cvEigenDecomposite(
			arrReconocer[i],
			numEigens,
			arrAutoVectores,
			0, 0,
			pAvgTrainImg,
			imgAReconocerProyectada);
			
		
		caraEntrenamientoCercana = encontrarVecinoCercano(imgAReconocerProyectada);
		
		if(caraEntrenamientoCercana != 1000)
			cercana = trainPersonNumMat->data.i[caraEntrenamientoCercana];
		else
			cercana = -1;
		
		if(caraaux==1)
			cercana2 = trainPersonNumMat->data.i[cara2];
		else
			cercana2 = -1;
			
		Reconociendo = 1;
		
		/*cvDestroyWindow("Reconociendo imagen...");
		
		sleep(3);
		
		
		char strAux[99] = "La Persona es: ";
		
		
		cvNamedWindow("Exito");
		cvMoveWindow("Exito", 300, 0);
		
		//Colocar el texto
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 1.0f, 1.0f, 1.0f, 1, CV_AA);
		cvPutText(final, strAux, cvPoint(100, 100), &font, cvScalar(0, 0, 0, 0));
		if(cercana != -1)
			cvPutText(final, carpetas[cercana]->d_name, cvPoint(100, 150), &font, cvScalar(0, 0, 0, 0));
		else	
			cvPutText(final, "DESCONOCIDA", cvPoint(100, 150), &font, cvScalar(0, 0, 0, 0));
			
		if(cercana2 != -1)
			cvPutText(final, carpetas[cercana2]->d_name, cvPoint(100, 200), &font, cvScalar(0, 0, 0, 0));
		//else
			//cvPutText(final, "DESCONOCIDO", cvPoint(100, 200), &font, cvScalar(0, 0, 0, 0));
			
			
		cvShowImage("Exito", final);
		
		cvWaitKey(2222);
	
	
	
	cvDestroyWindow("Exito");
	*/
	
	if(cercana != -1)
		printf("La Persona es: %s\n",carpetas[cercana]->d_name); 
	else	
		printf("DESCONOCIDA\n");
			
	if(cercana2 != -1)
		printf("La Persona es: %s\n",carpetas[cercana2]->d_name);
	
	}	
	
	return;
}

