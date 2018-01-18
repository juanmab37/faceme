#ifndef __FUNCS_H__
#define __FUNCS_H__

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>
#include <sys/param.h>
#include <sys/types.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#define FALSO 0
#define VERDADERO !FALSO
#define TAM 80
#define EPSILON 1

extern struct dirent **archivos;
extern struct dirent **carpetas;

int selecc_arch(const struct dirent *entry);
int selecc_carpetas(const struct dirent *entry);
void entrenar();
void PCA();
void reconocer(char archivoNombresImg[512]);
void guardarDatosEntrenamiento();
int  cargarDatosEntrenamiento(CvMat ** pTrainPersonNumMat);
int  encontrarVecinoCercano(float * projectedTestFace);
int  cargarArrayImgCaras();
int  cargarArrayImgCaras_reconocer(char archivoNombresImg[512]);
void Normalizacion(IplImage* imagen_scr,IplImage* imagen_dst);
void relacion();
void escalado(char * name);
int detectFaces(IplImage * img);
int detectar(IplImage * img);
void ver_imagen(IplImage* imagen);
void EigenDecomposite_gpu(void* objInput,int nEigObjs,void* eigInput,IplImage* avg,float* coeffs,int offset);


extern IplImage ** arrCaras;                    //Array de imágenes de caras
extern IplImage ** arrReconocer; 				 //Array de imágenes a reconocer
extern CvMat    *  numPersonaMat; 	         //Array de ID's de personas
extern int numCarasEntrenamiento;               //Cantidad de imágenes de entrenamiento
extern int numEigens;                           //Cantidad de AutoVectores y AutoValores (Eigens)
extern IplImage * pAvgTrainImg;                 //Imágen promedio
extern IplImage * pAvgTrainImg2;                 //Imágen promedio de prueba
extern IplImage ** arrAutoVectores;             //AutoVecotores
extern IplImage ** arrAutoVectores2;             //AutoVecotores de prueba
extern CvMat * matAutoValores;                  //AutoValores
extern CvMat * matAutoValores2;                  //AutoValores de prueba
extern CvMat * matImgEntrenamientoProyectadas;  //Imágenes de entrenamiento proyectadas en el subespacio PCA
extern CvHaarClassifierCascade *cascade_f;		 //Cascada de Clasificacion de Rostros
extern CvMemStorage            *storage;        //Almacenamiento de informacion
extern int Aux;                                 //variable auxiliar para saber si se crea una imagen
extern int Entrenando, Reconociendo, Comenzar;
extern int alphasort();					 //funcion que determina el orden alfabetico, usado para ordenar las fotos de entrenamiento mientras las carga
extern char trayectoria[MAXPATHLEN];			 // lugar donde se almacenara el directorio actual de trabajo
extern int AnchoImg;							// Auxiliar para la funcion de escalado
extern int AltoImg;							// Auxiliar para la funcion de escalado
extern int ancho;								// Auxiliar para la funcion de escalado
extern int alto;								// Auxiliar para la funcion de escalado
extern int caraaux,cara2;
/*Carga de entorno gráfico*/
extern IplImage* bienvenida;   
extern IplImage* salida;       
extern IplImage* carga;        
extern IplImage* reconociendo; 
extern IplImage* final;        

#endif
