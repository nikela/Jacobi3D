#include <stdio.h>
#include "timers.h"
#include "jacobi_utils.h"

void init_dimensions(dimensions * dim, int * T, char ** argv);

int main(int argc,char * argv[]) {

	double *** current, *** previous;
	dimensions dim;
	timer_tt * total_time;
	int T,i,j,k,t;
	
	total_time=timer_init();
	
	if (argc<5) {
		fprintf(stderr,"Usage: ./exec dimX dimY dimZ T [error:optional]\n");
		exit(1);
	}
 	else 
		init_dimensions(&dim,&T,argv);
	current=allocate_3D(dim.X,dim.Y,dim.Z);
	previous=allocate_3D(dim.X,dim.Y,dim.Z);	
	initialize_smooth_3D(previous,dim.X,dim.Y,dim.Z);
	initialize_smooth_3D(current,dim.X,dim.Y,dim.Z);

	timer_start(total_time);
	/*****Computational kernel of 3D Jacobi*****/
	for (t=0; t<T ; t++) {
		for (i=1;i<dim.X-1;i++)
			for (j=1;j<dim.Y-1;j++)
				for (k=1;k<dim.Z-1;k++)
					current[i][j][k]=(previous[i-1][j][k]+previous[i+1][j][k] \
 									+previous[i][j-1][k]+previous[i][j+1][k]  \
									+previous[i][j][k-1]+previous[i][j][k+1])/6.0;
		double *** middle=current;
		current=previous;
		previous=middle;
	}
	/*****************************************/
	timer_stop(total_time);
	double time=timer_report_sec(total_time);

	/*****Print Results*****/
	printf("Jacobi3D\tSerial\tX\t%d\tY\t%d\tZ\t%d\tT\t%d\t\nTotal\t%lf\n",dim.X,dim.Y,dim.Z,T,time);
	print_3D(current,dim.X,dim.Y,dim.Z);
	/**********************/

	return 0;
}

void init_dimensions(dimensions * dim, int * T, char * argv[]) {
	dim->X=atoi(argv[1]);
	dim->Y=atoi(argv[2]);
	dim->Z=atoi(argv[3]);
	(*T)=atoi(argv[4]);
	if (dim->X<=0 || dim->Y<=0 || dim->Z<=0 || T<=0) {
		fprintf(stderr,"Input error: all dimensions must be positive\n");
		exit(1);
	}	
}
	


