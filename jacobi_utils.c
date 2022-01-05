#include "jacobi_utils.h"
#include <stdio.h>
#include <stdlib.h>

double * allocate_1D(int dimX) {
	double * array=(double*)calloc(dimX,sizeof(double));
	if (array==NULL) {
		fprintf(stderr,"Error in allocation\n");
		exit(1);
	}
	return array;
}

unsigned char * allocate_compr(int dimX) {
	unsigned char * array=(unsigned char*)calloc(dimX,sizeof(unsigned char));
	if (array==NULL) {
		fprintf(stderr,"Error in allocation\n");
		exit(1);
	}
	return array;

}

double ** allocate_2D(int dimX,int dimY) {
	double ** array, * tmp;
	int i;
	tmp=(double*)calloc(dimX*dimY,sizeof(double));
	array=(double**)calloc(dimX,sizeof(double*));	
	for (i=0;i<dimX;i++)
		array[i]=tmp+i*dimY;
	if (array==NULL) {
		fprintf(stderr,"Error in allocation\n");
		exit(1);
	}
	return array;
}

double *** allocate_3D(int dimX,int dimY,int dimZ) {
	double *** array, * tmp;
	int i,j;
	tmp=(double*)calloc(dimX*dimY*dimZ,sizeof(double));
	array=(double***)calloc(dimX,sizeof(double**));
	for (i=0;i<dimX;i++) {
		array[i]=(double**)calloc(dimY,sizeof(double*));
		for (j=0;j<dimY;j++)
			array[i][j]=tmp+i*dimY*dimZ+j*dimZ;
	}
	if (array==NULL) {
		fprintf(stderr,"Error in allocation\n");
		exit(1);
	}
	return array;
}


void initialize_random_3D (double *** array, int dimX, int dimY, int dimZ) {
	int i,j,k;
	for (i=0;i<dimX;i++)
		for (j=0;j<dimY;j++)
			for (k=0;k<dimZ;k++)
				array[i][j][k]=rand()/1000000.0;
}

void initialize_constant_3D (double *** array, int dimX, int dimY, int dimZ) {
	int i,j,k;
	for (i=0;i<dimX;i++)
		for (j=0;j<dimY;j++)
			for (k=0;k<dimZ;k++) {
				if (i==0 || i==dimX-1 || j==0 || j==dimY-1 || k==0 || k==dimZ-1) 
					array[i][j][k]=10.0;
				else
					array[i][j][k]=0;
			}
}

void initialize_smooth_3D (double *** array, int dimX, int dimY, int dimZ) {
	int i,j,k;
	for (i=0;i<dimX;i++)
		for (j=0;j<dimY;j++)
			for (k=0;k<dimZ;k++)
				array[i][j][k]=0.1*(i*i*i+j*j+k);
}

void input_3D (double *** array, int dimX, int dimY, int dimZ) {
	int i,j,k;
	for (i=0;i<dimX;i++)
		for (j=0;j<dimY;j++)
			for (k=0;k<dimZ;k++)
				scanf("%lf",&array[i][j][k]);
}

void print_3D (double *** array, int dimX, int dimY, int dimZ) {
	int i,j,k;
	for (i=0;i<dimX;i++) {
		for (j=0;j<dimY;j++) {
			for (k=0;k<dimZ;k++) 
				printf("%.3lf ",array[i][j][k]);
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

