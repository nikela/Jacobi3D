#include <stdio.h>
#include "timers.h"
#include "jacobi_utils.h"
#include "omp.h"

void init_dimensions(dimensions * dim, int * T, int * threads,char ** argv);

int main(int argc,char * argv[]) {

	double *** current, *** previous;
	dimensions dim;
	timer_tt * total_time;
	int T,i,j,k,t,threads;
	
	total_time=timer_init();
	
	if (argc<6) {
		fprintf(stderr,"Usage: ./exec dimX dimY dimZ T threads\n");
		exit(1);
	}
 	else 
		init_dimensions(&dim,&T,&threads,argv);
	current=allocate_3D(dim.X,dim.Y,dim.Z);
	previous=allocate_3D(dim.X,dim.Y,dim.Z);	
	initialize_smooth_3D(previous,dim.X,dim.Y,dim.Z);
	initialize_smooth_3D(current,dim.X,dim.Y,dim.Z);

	timer_start(total_time);
	/*****Computational kernel of 3D Jacobi*****/
	#pragma omp parallel private(t,i,j,k) num_threads(threads)
	{
		for (t=0; t<T ; t++) {
			#pragma omp for
			for (i=1;i<dim.X-1;i++)
				for (j=1;j<dim.Y-1;j++)
					for (k=1;k<dim.Z-1;k++)
						current[i][j][k]=(previous[i-1][j][k]+previous[i+1][j][k] \
 									+previous[i][j-1][k]+previous[i][j+1][k]  \
									+previous[i][j][k-1]+previous[i][j][k+1])/6.0;
			#pragma omp single 
			{
				double *** middle=current;
				current=previous;
				previous=middle;
			}
		}
	}
	/*****************************************/
	timer_stop(total_time);
	double time=timer_report_sec(total_time);

	/*****Print Results*****/
	printf("Jacobi3d\tOpenMP\tX\t%d\tY\t%d\tZ\t%d\tT\t%d\tThreads\t%d\nTotal\t%lf\n",dim.X,dim.Y,dim.Z,T,threads,time);
	print_3D(current,dim.X,dim.Y,dim.Z);
	/**********************/

	return 0;
}

void init_dimensions(dimensions * dim, int * T, int * threads, char * argv[]) {
	dim->X=atoi(argv[1]);
	dim->Y=atoi(argv[2]);
	dim->Z=atoi(argv[3]);
	(*T)=atoi(argv[4]);
	(*threads)=atoi(argv[5]);
	if (dim->X<=0 || dim->Y<=0 || dim->Z<=0 || T<=0 || (threads<=0)) {
		fprintf(stderr,"Input error: all dimensions must be positive\n");
		exit(1);
	}	
}
	


