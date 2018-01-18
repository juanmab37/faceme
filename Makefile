.PHONY : clean

CXXFLAGS=-g -ggdb -Wall -Wno-unused-function
LDFLAGS=-L/usr/local/cuda-5.0/lib64  -lm  `pkg-config opencv --cflags --libs` 

all : Faceme

clean :
	rm  *.o 

objects = reconocer.o entrenar.o distancia.o \
	pca.o main.o var.o pcaGPU.o eigenDecomp.o\
 	kernel_avg.o kernel_cov.o kernel_d.o kernel_decomp.o 

Faceme: $(objects)
	g++  -o Faceme $(objects) $(LDFLAGS)

main.o : main.cpp funcs.h 
	g++ -c main.cpp 
	
entrenar.o : entrenar.cpp pca.o main.o var.o eigenDecomp.o
	g++ -c  entrenar.cpp 

pca.o : pca.cpp pcaGPU.o pca.h 
	g++ -c  pca.cpp 	



CPPFLAGS=-I../common -I/usr/local/cuda-5.0/include
CUFLAGS=-g -G -arch=sm_20 `pkg-config opencv --cflags --libs`
LDFLAGS=-L/usr/local/cuda-5.0/lib64 -lcudart -lm -L../common `pkg-config opencv --cflags --libs`
CXXFLAGS=-g -ggdb -Wall -Wno-unused-function -D__AVG_PARALELO__  -D__COV_PARALELO__ -D__D_PARALELO__ -D__DECOMP_PARALELO__

kernel_avg.o: kernel_avg.cu
	/usr/local/cuda-5.0/bin/nvcc $(CPPFLAGS) $(CUFLAGS) -o $@ -c $<

kernel_cov.o: kernel_cov.cu
	/usr/local/cuda-5.0/bin/nvcc $(CPPFLAGS) $(CUFLAGS) -o $@ -c $<
	
kernel_d.o: kernel_d.cu
	/usr/local/cuda-5.0/bin/nvcc $(CPPFLAGS) $(CUFLAGS) -o $@ -c $<

kernel_decomp.o: kernel_decomp.cu
	/usr/local/cuda-5.0/bin/nvcc $(CPPFLAGS) $(CUFLAGS) -o $@ -c $<

pcaGPU: pcaGPU.o kernel_avg.o kernel_cov.o kernel_d.o
	make -C ../common
	g++ -c $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

eigenDecomp : eigenDecomp.o decomp.h kernel_decomp.o 
	make -C ../common
	g++ -c $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

distancia.o : distancia.cpp main.o
	g++ -c  distancia.cpp

reconocer.o : reconocer.cpp main.o var.o
	g++ -c reconocer.cpp
	 
var.o : funcs.h
	g++ -c var.cpp


