
#include "funcs.h"


/*------------------ENTRENAR------------------*/

void entrenar()
{	
	printf("Entrenando...\n");
	
	int i, offset;

	/*Carga la entrada de entrenamiento.*/
	numCarasEntrenamiento = cargarArrayImgCaras();
	
	if (numCarasEntrenamiento < 2)
	{
		printf("Necesitamos dos o mas caras para el entrenamiento");
		return;
	}

	/*Aplicamos PCA para las caras de entrenamiento.*/
	PCA();
	
	/*Encontramos el subespacio. Ahora convertimos las imágenes de entrenamiento
	en puntos en este subespacio. Se utiliza cvEigenDecomposite().*/

	/*Proyectamos las caras de entrenamiento en el subespacio de PCA.
	Aquí se alojan las proyecciones de las imágenes de entrenamiento.*/
	matImgEntrenamientoProyectadas = cvCreateMat( numCarasEntrenamiento, numEigens, CV_32FC1 );
	
	offset = matImgEntrenamientoProyectadas->step / sizeof(float);
	printf("Eig Des INICIO\n");

	//for (i = 0; i < numCarasEntrenamiento; i++)
	//{
		/*Proyecta cada imágen de entrenamiento (transformada en un punto) en el
		subespacio creado por PCA.*/
		EigenDecomposite_gpu(
			arrCaras,       //(input)
			numEigens,         //(input)
			arrAutoVectores,   //(input)
			pAvgTrainImg,
			matImgEntrenamientoProyectadas->data.fl, //(output)
			offset);        

	//}
	printf("Eig Des FIN\n");
	/*The function cvEigenDecomposite calculates all decomposition coefficients 
	for the input object using the previously calculated eigen objects basis and 
	the averaged object. Depending on ioFlags parameter it may be used either in 
	direct access or callback mode.*/
	
	/*Guarda los datos de la función entrenar() en un archivo XML.*/
	guardarDatosEntrenamiento();
	
	Entrenando = 1;
	
	return;
}

/*------------------CARGAR DATOS DE ENTRENAMIENTO------------------*/

int cargarDatosEntrenamiento(CvMat ** pTrainPersonNumMat)
{
	/*Funcion para cargar la info de un archivo XML
	En este caso, necesitamos cargar la info de facedata.xml, donde
	tenemos los valores de numEigens, arrAutoVectores, etc...*/
		
	CvFileStorage * infoArchivo;
	int i;

	/*Abrimos "facedata.xml" con la opción de LEER (CV_STORAGE_READ).*/
	infoArchivo = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	
	if (!infoArchivo)
	{
		fprintf(stderr, "No se puede abrir facedata.xml\n");
		return 0;
	}

	/*Lee los datos:*/
	numEigens = cvReadIntByName(infoArchivo, 0, "numEigens", 0);  //Lee el valor titulado como "numEigens" (int nEigens).
	numCarasEntrenamiento = cvReadIntByName(infoArchivo, 0, "numCarasEntrenamiento", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(infoArchivo, 0, "trainPersonNumMat", 0);  //ID's
	matAutoValores  = (CvMat *)cvReadByName(infoArchivo, 0, "matAutoValores", 0);
	matImgEntrenamientoProyectadas = (CvMat *)cvReadByName(infoArchivo, 0, "matImgEntrenamientoProyectadas", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(infoArchivo, 0, "avgTrainImg", 0);
	arrAutoVectores = (IplImage **)cvAlloc(numCarasEntrenamiento*sizeof(IplImage *));
	for (i = 0; i < numEigens; i++)  //Lee lols AutoVectores
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		arrAutoVectores[i] = (IplImage *)cvReadByName(infoArchivo, 0, varname, 0);
	}

	cvReleaseFileStorage( &infoArchivo );

	return 1;
}

/*------------------GUARDAR DATOS DE ENTRENAMIENTO------------------*/

void guardarDatosEntrenamiento()
{
	CvFileStorage * infoArchivo;
	int i;

	/*Crea un "apuntador" al archivo XML.
	CV_STORAGE_WRITE: crear y/o escribir en el archivo que señalamos "facedata.xml".*/
	infoArchivo = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	/*Guardamos toda la info allí (cvWrite)*/
	cvWriteInt( infoArchivo, "numEigens", numEigens );  //"nEigens le ponemos como nombre a nEigens.
	cvWriteInt( infoArchivo, "numCarasEntrenamiento", numCarasEntrenamiento );
	cvWrite(infoArchivo, "trainPersonNumMat", numPersonaMat, cvAttrList(0,0));
	cvWrite(infoArchivo, "matAutoValores", matAutoValores, cvAttrList(0,0));
	cvWrite(infoArchivo, "matImgEntrenamientoProyectadas", matImgEntrenamientoProyectadas, cvAttrList(0,0));
	cvWrite(infoArchivo, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for (i = 0; i < numEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(infoArchivo, varname, arrAutoVectores[i], cvAttrList(0,0));
	}

	cvReleaseFileStorage(&infoArchivo);
	
	return;
}

/*------------------INICIALIZAR EL ARRAY DE CARAS (arrCaras)------------------*/

int cargarArrayImgCaras()
{
	/*Carga las imágenes y los ID de cada persona en los casos de "entrenar" y "reconocer".*/
	
	char cadena[MAXPATHLEN];
	int contar, contar2, contar3;
	int i, j, aux=0, numCaras = 0;
	//char archivoNombresImg[512];   //Buffer donde guarda caracteres
	
	if ( getcwd(trayectoria,TAM) == NULL )
    { 
		printf("Error obteniendo la trayectoria actual\n");
        exit(0);
    }
    
    contar = scandir(trayectoria, &carpetas, selecc_carpetas, alphasort);
	
	if (contar <= 0)
    { 
        printf("No hay archivos en este direntorio\n");
        exit(0);
    }
    
    for (i=0; i<contar; ++i)
    {
		chdir(carpetas[i]->d_name);
						
		if ( getcwd (cadena,TAM) == NULL )
		{ 
			printf("Error obteniendo la trayectoria actual\n");
			exit(0);
		}
		
		contar2 = scandir(cadena, &archivos, selecc_arch, alphasort);
		
		if (contar2 <= 0)
		{ 
			printf("No hay archivos en este direntorio\n");
			exit(0);
		}
		
		numCaras += contar2;
		chdir("..");
	}
	
	
	/*Inicializaciones*/
	arrCaras        = (IplImage **)cvAlloc( numCaras*sizeof(IplImage *) );
	numPersonaMat = cvCreateMat( 1, numCaras, CV_32SC1 );
	
	for (i=0; i<contar; ++i)
	{
		chdir(carpetas[i]->d_name);
						
		if ( getcwd (cadena,TAM) == NULL )
		{ 
			printf("Error obteniendo la trayectoria actual\n");
			exit(0);
		}
		
		contar3 = scandir(cadena, &archivos, selecc_arch, alphasort);
		
		/*Colocar cada imágen (cara) dentro de cada celda de arrCaras*/
		for (j=0; j < contar3; j++)
		{
			numPersonaMat->data.i[aux + j] = i;	 //data.i (ya que el ID es un int)
			
			//escalado(archivos[j]->d_name);// redimensiona la imagen de nombre archivoNombresImg
			
			arrCaras[aux + j] = cvLoadImage(archivos[j]->d_name, CV_LOAD_IMAGE_GRAYSCALE); //cargamos la imagen en escala de grises
			//arrCaras[aux + j] = cvLoadImage(archivos[j]->d_name, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
		
			//Por si no pudo cargar la imágen:
			if (!arrCaras[aux + j])
			{
				fprintf(stderr, "No se pudo cargar la imágen %s\n", archivos[j]->d_name);
				return 0;
			}
		
			//Enfoca
			cvEqualizeHist(arrCaras[aux + j],arrCaras[aux + j]);
		
		}
				
		aux += contar3;
					
		chdir("..");
	}
	
	
	return numCaras;
}

int cargarArrayImgCaras_reconocer(char archivoNombresImg[512]) /*archivoNombresImg tiene la ruta de la imágen a reconocer*/
{
	//Carga la imágen a reconocer
	int i = 0, numCaras = 1;
	char temp[] = "trash/temp.pgm";
		

	IplImage* img_aux_original = cvLoadImage(archivoNombresImg);
	
	//Por si no pudo cargar la imágen:
		if (!img_aux_original)
		{
			fprintf(stderr, "No se pudo cargar la imágen %s\n",archivoNombresImg);
			return 0;
		}
	
	cvSaveImage(temp,img_aux_original);
	cvReleaseImage(&img_aux_original);
	
	//IplImage* img_aux = cvLoadImage(temp);
	//numCaras = detectar(img_aux);
	
	if( numCaras == 0) {
		numCaras = 1;
		Aux = 0;
	}
		
	/*Inicializaciones*/
	arrReconocer = (IplImage **)cvAlloc( numCaras*sizeof(IplImage *) );
			
	/*Carga las imágenes de caras:
	arrCaras es un array donde en cada celda hay una imágen (cara).
	La transforma en GrayScale.*/
	
	if (numCaras > 0 && Aux == 1) //Esto pasa si usamos el detector de caras: detectFaces
	{
		for(i = 0; i < numCaras; i++) 
		{
			char* auxiliar;
			sprintf(auxiliar,"trash/imagenCaraRecortada%d.pgm",i);
			
			escalado(auxiliar);// redimensiona la imagen de nombre archivoNombresImg
			arrReconocer[i] = cvLoadImage(auxiliar, CV_LOAD_IMAGE_GRAYSCALE);  //cargamos la imagen en escala de grises
			//arrCaras[aux] = cvLoadImage(auxiliar, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
			
			//Por si no pudo cargar la imágen:
			if (!arrReconocer[i])
			{
				fprintf(stderr, "No se pudo cargar la imágen %s\n", auxiliar);
				return 0;
			} 
			
			cvEqualizeHist(arrReconocer[i],arrReconocer[i]);
			//ver_imagen(arrCaras[i]);
		
		}
	}	
	else
	{
		//escalado(temp);// redimensiona la imagen de nombre archivoNombresImg
		printf("Procedimiento normal\n");
		arrReconocer[0] = cvLoadImage(temp, CV_LOAD_IMAGE_GRAYSCALE); //cargamos la imagen en escala de grises
		//arrCaras[aux] = cvLoadImage(archivoNombresImg, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
		
		
		//Por si no pudo cargar la imágen:
		if (!arrReconocer[0])
		{
			fprintf(stderr, "No se pudo cargar la imágen %s\n", temp);
			return 0;
		}
			
		cvEqualizeHist(arrReconocer[0],arrReconocer[0]);
		//ver_imagen(arrReconocer[0]);	
	}
	
	return numCaras;
}



void ver_imagen(IplImage* imagen)
{
	cvNamedWindow( "test", 1); // representamos la imagen escalada 
                 // (con el 1 indicamos que la ventana se ajuste a los parámetros de la imagen)
 
	cvShowImage( "test", imagen); 
	cvWaitKey(0); // pulsamos cualquier tecla para terminar el programa
	cvDestroyAllWindows(); // destruimos todas las ventanas
}

void relacion()
{
	float rel;
	
	rel = ((float)ancho/(float)alto);
	//printf("relacion = %f \n", rel);
	
	if (rel < 0.81 or 0.83 < rel)
	{
		if(rel < 0.81)
		{
			alto = alto - 10;
			AltoImg = AltoImg + 10;
			relacion();
		}
		if(rel > 0.83)
		{
			ancho = ancho - 10;
			AnchoImg = AnchoImg + 10;
			relacion();
		}
	}

}

void escalado(char * name)
{
	IplImage* imagen; //Inicialización de "imagen" 
	IplImage* img = NULL;
	
	img = cvLoadImage(name, 0); //Cargamos la imagen
	
	/*Píxeles en el eje x de la imagen escalada, es decir, estamos definiendo la escala X*/
	int px = 92;
	/*Píxeles en el eje x de la imagen escalada, es decir, estamos definiendo la escala Y*/
	int py = 112;
	
	ancho = img->width;
	alto = img->height;
	
	if(0.81 > (float)ancho/(float)alto or 0.83 < (float)ancho/(float)alto )
	{ 
		relacion();
	
		cvSetImageROI( img , cvRect( AnchoImg/2 , 1 , ancho, alto) ) ;
	
		imagen = cvCreateImage( cvGetSize(img) ,img->depth, img->nChannels);
	
		cvCopy ( img , imagen , NULL ) ;

		cvResetImageROI(img);
		cvReleaseImage(&img);
 
	}
	else
		imagen = cvLoadImage(name, 0); //Cargamos la imagen
		
	/*Creamos la estructura donde ubicaremos la imagen escalada, 
    siendo px y py los píxeles de la imagen destino, es decir, 
    el propio factor de escala.*/
	IplImage *resized = cvCreateImage(cvSize(px, py), IPL_DEPTH_8U, 1); 
	
	cvResize(imagen, resized,CV_INTER_LINEAR); //Función escalado de imagen
	
	//cvEqualizeHist(resized, resized);
	
	cvSaveImage(name, resized);

	
	
	cvReleaseImage(&imagen);
	cvReleaseImage(&resized);

}

int detectar(IplImage * img)
{
	int cant=0;
	// Archivos de cascada de caracteristicas para ...
    const char *file1 = "haarcascade_frontalface_alt.xml"; // Deteccion de Rostros
   
    /* Cargar Clasificador de Rostros */
    cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
    /* Inicializar el Modulo de Memoria, Necesario para el Detector de Rostros */
    storage = cvCreateMemStorage(0);
 
	cant = detectFaces(img);
   
	cvReleaseImage( &img );
    
    return cant;  
}
 
int detectFaces(IplImage * img) 
{
	int cant=0;
	CvSeq* faces = cvHaarDetectObjects( img, cascade_f, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20) );
	printf("Total de caras a reconocer: %d \n",faces->total);
    for( int i = 0; i <(faces ? faces->total : 0); i++ ) 
    {
        CvRect *r = (CvRect*)cvGetSeqElem( faces, i );
        cvRectangle( img, cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),CV_RGB(255,0,0), 3 );
        cvSetImageROI (img, *r);
        
        /* Crear imagen de destino
		Tener en cuenta que cvGetSize devolverá la anchura y la altura del retorno de la inversión  */
		IplImage *img2 = cvCreateImage( cvGetSize(img) , img->depth, img->nChannels) ;
 
		/* Copia sub-imagen */
		cvCopy ( img , img2 , NULL ) ;
        
        if (img2->width < 91 && img2->height < 111)
        {
			cvResetImageROI(img);
			cvReleaseImage(&img2);
		}
        else
        {
			char *ubicacion;
			sprintf(ubicacion,"trash/imagenCaraRecortada%d.pgm",i);
			cvSaveImage( ubicacion ,img2,0); //Cambié de carpeta
			printf("creo la imagen \n");
			Aux = 1;
			cant++;
			/* Siempre restablecer la Región de Interés */
			cvResetImageROI(img);
			cvReleaseImage(&img2);
		}
    }
 
    cvClearMemStorage(storage);
    
    return cant;
}

int selecc_carpetas(const struct dirent *entry)
{ 
	if ((strcmp(entry->d_name, ".") == 0)  || (strcmp(entry->d_name, "..") == 0) || (strcmp(entry->d_name, "facemeprints") == 0) || (strcmp(entry->d_name, "trash") == 0) || (strcmp(entry->d_name, "Info proyecto") == 0) )
        return (FALSO);
        
    if(entry->d_type & DT_DIR)
		return (VERDADERO);
	else
		return (FALSO);
}
int selecc_arch(const struct dirent *entry)
{
	const char *ptr;
                
    if ((strcmp(entry->d_name, ".")== 0) || (strcmp(entry->d_name, "..") == 0))
        return (FALSO);
    
    
    ptr = rindex(entry->d_name, '.'); // Probar que tenga un punto 
    
    if ( (ptr != NULL )  && ( ((strcmp(ptr, ".pgm") == 0)) || ((strcmp(ptr, ".jpg") == 0)) || ((strcmp(ptr, ".JPG") == 0)) ) )//|| (strcmp(ptr, ".h") == 0) || (strcmp(ptr, ".o") == 0) ) )
        return (VERDADERO);
    else
        return(FALSO);

}

