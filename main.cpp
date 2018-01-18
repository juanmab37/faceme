
#include "funcs.h"

void presentacion()
{	
	
	cvNamedWindow("Bienvenido");
	cvMoveWindow("Bienvenido", 300, 0);
	
	cvShowImage("Bienvenido", bienvenida);

	
	while (1)
	{
		if (cvWaitKey(0) == 10)
			break;
		if (cvWaitKey(0) == 27)
			exit(0);
	}
	
	cvDestroyWindow("Bienvenido");
	
	return;
}	

/*------------------MAIN------------------*/

int main(int argc, char** argv)
{
	char* aux = argv[1];
	
	//presentacion();
	
	//cvNamedWindow("Cargando...");
	//cvMoveWindow("Cargando...", 300, 0);
	
	while (Entrenando == 0)
	{ 
		//cvShowImage("Cargando...", carga);
		//cvWaitKey(350);
		entrenar();
	}
	
	//cvDestroyWindow("Cargando...");
	
	//cvNamedWindow("Reconociendo imagen...");
	//cvMoveWindow("Reconociendo imagen...", 300, 0);
	
	while (Reconociendo == 0)
	{ 
		//cvShowImage("Reconociendo imagen...", reconociendo);
		//cvWaitKey(290);
		reconocer(aux);
		system("rm trash/temp.pgm");
	}
	
	
	//cvNamedWindow("¡Hasta pronto!");
	//cvMoveWindow("¡Hasta pronto!", 300, 0);
	//cvShowImage("¡Hasta pronto!", salida);
	
	//cvWaitKey(2500);
	
	return 0;
}

