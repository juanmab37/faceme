#include "funcs.h"
/*------------------ENCONTRAR EL VECINO (punto) MAS CERCANO------------------*/

int encontrarVecinoCercano(float * imgAReconocerProyectada)
{
	/*Eigenface "recognizes" a face image by looking for the training 
	image that's closest to it in the PCA subspace. Finding the closest 
	training example in a learned subspace is a very common AI technique. 
	It's called Nearest Neighbor matching.*/
	
	/* It computes distance from the Projected Test Image to each projected
	training example. The distance basis here is "Squared Euclidean Distance."*/
 
	/*Trabajamos en un Espacio Euclideano. La distancia entre dos puntos es
	el módulo del vector mismo. El módulo es el usual: la raiz cuadrada de 
	la suma de las componentes del vector al cuadrado.
	Acá solamente tomamos la suma y obviamos la raíz ya que el módulo (distancia
	entre dos puntos) de un vector v es menor que el módulo de un vector u sii
	la suma de los compontentes de v al cuadrado es menor que las sumas de u.*/
	
	CvMat * trainPersonNumMat = 0;
	/*Cargamos la informacion guardada del entrenamiento (el XML)*/
	if( !cargarDatosEntrenamiento( &trainPersonNumMat ) ) 
		perror("cargarDatosEntrenamiento");
	
	double minDist = DBL_MAX;  //Max double
	double minDist2 = DBL_MAX;
	int i, nunCaraEntrenada, caraEntrenamientoCercana = 0,porcentaje;
	double dist = 0;
	float resta = 0;

	for (nunCaraEntrenada = 0; nunCaraEntrenada < numCarasEntrenamiento; nunCaraEntrenada++)
	{
		dist = 0;

		for (i = 0; i < numEigens; i++)
		{
			/*Hace el punto (en el subespacio PCA) proyectado de cada imágen
			a reconocer menos el punto proyectado (en el subespacio PCA) de cada
			imágen de entrenamiento. Luego, a esta resta, le aplica el cuadrado.
			Por lo tanto, es el módulo de la distancia entre el punto proyectado de 
			cada imágen a reconocer y el punto proyectado de cada imágen de entrenamiento.*/
			
			resta = imgAReconocerProyectada[i] - matImgEntrenamientoProyectadas->data.fl[nunCaraEntrenada*numEigens + i];
			dist += resta*resta; // Euclidean
			//dist += resta*resta / matAutoValores->data.fl[i];  // Mahalanobis
			
			//printf(" %f \n",dist);
			
		}
		dist = sqrt(dist);
		//printf(" DISTANCIA = %f \n",dist);
		
		/*Encuentra la mínima distancia:
		Nos dice qué cara de entrenamiento es la que mejor coincide con las caras
		a reconocer*/
		if (dist < minDist)
		{
			if (trainPersonNumMat->data.i[caraEntrenamientoCercana] != trainPersonNumMat->data.i[nunCaraEntrenada])
			{
				minDist2 = minDist;
				cara2 = caraEntrenamientoCercana;
			}
			minDist = dist;
			caraEntrenamientoCercana = nunCaraEntrenada;
			
		}
		
		
		/*double maxDist = DBL_MAX;
		if (dist > maxDist)
			maxDist = dist;
				
		printf(" Maxima = %f \n",maxDist); */
	}
	//printf(" cara: %d, cara2 %d \n", caraEntrenamientoCercana, cara2);
	//printf(" cara: %d, cara2 %d \n",trainPersonNumMat->data.i[caraEntrenamientoCercana], trainPersonNumMat->data.i[cara2]);
	if(minDist > 7500) //IMAGEN DESCONOCIDA
	{
		printf("DESCONOCIDO \n");
		return 1000; 	//caraEntrenamientoCercana = 1000
	}
	
	
	if(minDist < 1000) //LA IMAGEN ESTÁ EN LA BASE DE DATOS
		porcentaje = 100;
	else
	{
		if(minDist <1500)
			porcentaje = 90;
		else
			porcentaje = 100 - (minDist*90)/7500 ;
	}	
	printf(" MINIMA = %f \n",minDist);
	printf("Porcentaje:  %d \n",porcentaje);
	
	if(5000 < minDist && minDist < 7500)
	{
		caraaux = 1;
		printf("Al ser bajo el porcentaje es poca la probabilidad del acierto\n ");
	}
	
	
	/*Devuelve la cara de entrenamiento más cercana.*/
	return caraEntrenamientoCercana;
}

