#include "funcs.h"

struct dirent **archivos;
struct dirent **carpetas;

IplImage ** arrCaras;                    //Array de imágenes de caras
IplImage ** arrReconocer; 				 //Array de imágenes a reconocer
CvMat    *  numPersonaMat; 	         //Array de ID's de personas
int numCarasEntrenamiento;               //Cantidad de imágenes de entrenamiento
int numEigens;                           //Cantidad de AutoVectores y AutoValores (Eigens)
IplImage * pAvgTrainImg;                 //Imágen promedio
IplImage * pAvgTrainImg2;                 //Imágen promedio de prueba
IplImage ** arrAutoVectores;             //AutoVecotores
IplImage ** arrAutoVectores2;             //AutoVecotores de prueba
CvMat * matAutoValores;                  //AutoValores
CvMat * matAutoValores2;                  //AutoValores de prueba
CvMat * matImgEntrenamientoProyectadas;  //Imágenes de entrenamiento proyectadas en el subespacio PCA
CvHaarClassifierCascade *cascade_f;		 //Cascada de Clasificacion de Rostros
CvMemStorage            *storage;        //Almacenamiento de informacion
int Aux;                                 //variable auxiliar para saber si se crea una imagen
int Entrenando, Reconociendo, Comenzar;
extern int alphasort();					 //funcion que determina el orden alfabetico, usado para ordenar las fotos de entrenamiento mientras las carga
char trayectoria[MAXPATHLEN];			 // lugar donde se almacenara el directorio actual de trabajo
int AnchoImg;							// Auxiliar para la funcion de escalado
int AltoImg;							// Auxiliar para la funcion de escalado
int ancho;								// Auxiliar para la funcion de escalado
int alto;								// Auxiliar para la funcion de escalado
int caraaux,cara2;
IplImage* bienvenida   = cvLoadImage("facemeprints/faceme_presentacion.jpg");
IplImage* salida       = cvLoadImage("facemeprints/faceme_salida.jpg");
IplImage* carga        = cvLoadImage("facemeprints/faceme_cargando.jpg");
IplImage* reconociendo = cvLoadImage("facemeprints/faceme_reconociendo.jpg");
IplImage* final        = cvLoadImage("facemeprints/faceme_exito.jpg");
