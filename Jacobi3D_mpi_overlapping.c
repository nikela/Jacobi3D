#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "jacobi_utils.h"
#include "timers.h"

int size,rank;

void scatter_gather(double *** current, double *** local_current, double *** local_previous, dimensions ext_dim, dimensions local_dim, dimensions proc,int coords[], char pick, MPI_Comm cart_comm);
void compute_boundaries(dimensions dim,dimensions ext_dim, dimensions local_dim, dimensions * min, dimensions * max);
void compute_new_boundaries(dimensions * min_b, dimensions * max_b, dimensions proc,int coords[3]);
void pack(double * buf, double *** local, dimensions local_dim,char neighbour);
void unpack(double * buf, double *** local, dimensions local_dim,char neighbour);



int main(int argc, char * argv[]) {

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm cart_comm;
	
	dimensions dim,ext_dim,local_dim,min,max,proc,min_b,max_b;
	int T,up,down,left,right,back,front,i,j,k,t;
	double *** current, *** local_current, *** local_previous;
	timer_tt * total, * inner, * packing, * boundary;
	
	if (argc<8) {
		fprintf(stderr,"Usage: ./exec X Y Z T procX procY procZ\n");
		exit(1);
	}	
	else
		init_parallel_dimensions(&dim,&T,&proc,argv);

	compute_local_dimensions(dim,proc,&ext_dim,&local_dim);
	
	if (rank==0) {
		current=allocate_3D(ext_dim.X,ext_dim.Y,ext_dim.Z);
		initialize_smooth_3D(current,dim.X,dim.Y,dim.Z);
	//	print_3D(current,dim.X,dim.Y,dim.Z);
	}

	local_current=allocate_3D(local_dim.X+2,local_dim.Y+2,local_dim.Z+2);
	local_previous=allocate_3D(local_dim.X+2,local_dim.Y+2,local_dim.Z+2);

	/**** Set cartesian topology ****/
	int coords[3];
	int periods[3]={0,0,0};
	int dim_size[3]={proc.X,proc.Y,proc.Z};
	MPI_Cart_create(MPI_COMM_WORLD,3,dim_size,periods,0,&cart_comm);
	MPI_Cart_coords(cart_comm,rank,3,coords);	
	/*******************************/

	scatter_gather(current,local_current,local_previous,ext_dim,local_dim,proc,coords,'s',cart_comm);
	compute_boundaries(dim,ext_dim,local_dim,&min,&max);
	min_b.X=min.X; min_b.Y=min.Y; min_b.Z=min.Z;
	max_b.X=max.X; max_b.Y=max.Y; max_b.Z=max.Z;
	compute_new_boundaries(&min_b,&max_b,proc,coords);
	
	MPI_Cart_shift(cart_comm,0,1,&up,&down);
	MPI_Cart_shift(cart_comm,1,1,&back,&front);
	MPI_Cart_shift(cart_comm,2,1,&left,&right);

	if (up<0) 
		up=MPI_PROC_NULL;
	if (down<0) 
		down=MPI_PROC_NULL;
	if (left<0) 
		left=MPI_PROC_NULL;
	if (right<0) 
		right=MPI_PROC_NULL;
	if (back<0) 
		back=MPI_PROC_NULL;
	if (front<0) 
		front=MPI_PROC_NULL;
	
	double * upsend=allocate_1D(local_dim.Y*local_dim.Z);
	double * uprecv=allocate_1D(local_dim.Y*local_dim.Z);
	double * downsend=allocate_1D(local_dim.Y*local_dim.Z);
	double * downrecv=allocate_1D(local_dim.Y*local_dim.Z);
	double * leftsend=allocate_1D(local_dim.Y*local_dim.X);
	double * leftrecv=allocate_1D(local_dim.Y*local_dim.X);
	double * rightsend=allocate_1D(local_dim.Y*local_dim.X);
	double * rightrecv=allocate_1D(local_dim.Y*local_dim.X);
	double * backsend=allocate_1D(local_dim.X*local_dim.Z);
	double * backrecv=allocate_1D(local_dim.X*local_dim.Z);
	double * frontsend=allocate_1D(local_dim.X*local_dim.Z);
	double * frontrecv=allocate_1D(local_dim.X*local_dim.Z);

	MPI_Request request_up[2];
	MPI_Request request_down[2];
	MPI_Request request_left[2];
	MPI_Request request_right[2];
	MPI_Request request_back[2];
	MPI_Request request_front[2];
	MPI_Status status_up[2];
	MPI_Status status_down[2];
	MPI_Status status_left[2];
	MPI_Status status_right[2];
	MPI_Status status_back[2];
	MPI_Status status_front[2];
	
	int size_count[6];
	total=timer_init();
	inner=timer_init();
	boundary=timer_init();
	packing=timer_init();

	MPI_Barrier(MPI_COMM_WORLD);
	timer_start(total);
	for (t=0;t<T;t++) {

		if (up!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(upsend,local_previous,local_dim,'u');
			timer_stop(packing);
			MPI_Isend(upsend,local_dim.Y*local_dim.Z,MPI_DOUBLE,up,55,MPI_COMM_WORLD,&request_up[0]);
			MPI_Irecv(uprecv,local_dim.Y*local_dim.Z,MPI_DOUBLE,up,56,MPI_COMM_WORLD,&request_up[1]);
		}
		if (down!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(downsend,local_previous,local_dim,'d');
			timer_stop(packing);
			MPI_Isend(downsend,local_dim.Y*local_dim.Z,MPI_DOUBLE,down,56,MPI_COMM_WORLD,&request_down[0]);
			MPI_Irecv(downrecv,local_dim.Y*local_dim.Z,MPI_DOUBLE,down,55,MPI_COMM_WORLD,&request_down[1]);
		}
		if (left!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(leftsend,local_previous,local_dim,'l');
			timer_stop(packing);
			MPI_Isend(leftsend,local_dim.Y*local_dim.X,MPI_DOUBLE,left,57,MPI_COMM_WORLD,&request_left[0]);
			MPI_Irecv(leftrecv,local_dim.Y*local_dim.X,MPI_DOUBLE,left,58,MPI_COMM_WORLD,&request_left[1]);
		}
		if (right!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(rightsend,local_previous,local_dim,'r');
			timer_stop(packing);
			MPI_Isend(rightsend,local_dim.Y*local_dim.X,MPI_DOUBLE,right,58,MPI_COMM_WORLD,&request_right[0]);
			MPI_Irecv(rightrecv,local_dim.Y*local_dim.X,MPI_DOUBLE,right,57,MPI_COMM_WORLD,&request_right[1]);
		}
		if (back!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(backsend,local_previous,local_dim,'b');
			timer_stop(packing);
			MPI_Isend(backsend,local_dim.X*local_dim.Z,MPI_DOUBLE,back,59,MPI_COMM_WORLD,&request_back[0]);
			MPI_Irecv(backrecv,local_dim.X*local_dim.Z,MPI_DOUBLE,back,60,MPI_COMM_WORLD,&request_back[1]);
		}
		if (front!=MPI_PROC_NULL) {
			timer_start(packing);
			pack(frontsend,local_previous,local_dim,'f');
			timer_stop(packing);
			MPI_Isend(frontsend,local_dim.X*local_dim.Z,MPI_DOUBLE,front,60,MPI_COMM_WORLD,&request_front[0]);
			MPI_Irecv(frontrecv,local_dim.X*local_dim.Z,MPI_DOUBLE,front,59,MPI_COMM_WORLD,&request_front[1]);
		}

		timer_start(inner);
		for (i=min.X+1;i<=max.X-1;i++)
			for (j=min.Y+1;j<=max.Y-1;j++)
				for(k=min.Z+1;k<=max.Z-1;k++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
		timer_stop(inner);


		if (up!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_up,status_up);
			MPI_Get_count(&status_up[1],MPI_DOUBLE,&size_count[0]);
			timer_start(packing);
			unpack(uprecv,local_previous,local_dim,'u');
			timer_stop(packing);
			timer_start(boundary);
			i=min_b.X;
			for (j=min_b.Y;j<=max_b.Y;j++)
				for(k=min_b.Z;k<=max_b.Z;k++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
			timer_stop(boundary);
		}

		if (down!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_down,status_down);
			MPI_Get_count(&status_down[1],MPI_DOUBLE,&size_count[1]);
			timer_start(packing);
			unpack(downrecv,local_previous,local_dim,'d');
			timer_stop(packing);
			timer_start(boundary);
			i=max_b.X;
			for (j=min_b.Y;j<=max_b.Y;j++)
				for(k=min_b.Z;k<=max_b.Z;k++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
			timer_stop(boundary);

		}

		if (left!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_left,status_left);
			MPI_Get_count(&status_left[1],MPI_DOUBLE,&size_count[2]);
			timer_start(packing);
			unpack(leftrecv,local_previous,local_dim,'l');
			timer_stop(packing);
			timer_start(boundary);
			k=min_b.Z;
			for (i=min_b.X;i<=max_b.X;i++)
				for(j=min_b.Y;j<=max_b.Y;j++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
			timer_stop(boundary);

		}
		if (right!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_right,status_right);
			MPI_Get_count(&status_right[1],MPI_DOUBLE,&size_count[3]);
			timer_start(packing);
			unpack(rightrecv,local_previous,local_dim,'r');
			timer_stop(packing);
			timer_start(boundary);
			k=max_b.Z;
			for (i=min_b.X;i<=max_b.X;i++)
				for(j=min_b.Y;j<=max_b.Y;j++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;

			timer_stop(boundary);

		}
		if (back!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_back,status_back);
			MPI_Get_count(&status_back[1],MPI_DOUBLE,&size_count[4]);
			timer_start(packing);
			unpack(backrecv,local_previous,local_dim,'b');
			timer_stop(packing);
			timer_start(boundary);
			j=min_b.Y;
			for (i=min_b.X;i<=max_b.X;i++)
				for(k=min_b.Z;k<=max_b.Z;k++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
			timer_stop(boundary);

		}
		if (front!=MPI_PROC_NULL) {
			MPI_Waitall(2,request_front,status_front);
			MPI_Get_count(&status_front[1],MPI_DOUBLE,&size_count[5]);
			timer_start(packing);
			unpack(frontrecv,local_previous,local_dim,'f');
			timer_stop(packing);
			timer_start(boundary);
			j=max_b.Y;
			for (i=min_b.X;i<=max_b.X;i++)
				for(k=min_b.Z;k<=max_b.Z;k++)
					local_current[i][j][k]=(local_previous[i-1][j][k]+local_previous[i+1][j][k] 
								+local_previous[i][j-1][k]+local_previous[i][j+1][k] 
								+local_previous[i][j][k-1]+local_previous[i][j][k+1])/6.0;
			timer_stop(boundary);

		}

		double *** middle=local_current;
		local_current=local_previous;
		local_previous=middle; 
	}
	timer_stop(total);
	MPI_Barrier(MPI_COMM_WORLD);

	scatter_gather(current,local_current,local_previous,ext_dim,local_dim,proc,coords,'g',cart_comm);
	double total_time=timer_report_sec(total);
	double inner_time=timer_report_sec(inner);
	double boundary_time=timer_report_sec(boundary);
	double packing_time=timer_report_sec(packing);
	double avg_total,avg_comp,avg_pack,avg_boundary_comp;
	MPI_Reduce(&total_time,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&inner_time,&avg_comp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&packing_time,&avg_pack,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&boundary_time,&avg_boundary_comp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	if (rank==0) {
		avg_total=avg_total/(double)size;
		avg_comp=avg_comp/(double)size;
		avg_boundary_comp=avg_boundary_comp/(double)size;
		avg_pack=avg_pack/(double)size;

		printf("Jacobi3D\tMPI\tOverlapping\tX\t%d\tY\t%d\tZ\t%d\tT\t%d\n",dim.X,dim.Y,dim.Z,T);
		printf("Total\t%lf\tComputation\t%lf\tPack\t%lf\n",avg_total,avg_comp+avg_boundary_comp,avg_pack);
	//	print_3D(current,dim.X,dim.Y,dim.Z);	
	}

	MPI_Finalize(); 
	return 0;
}

void init_parallel_dimensions(dimensions * dim,int * T, dimensions * proc,char ** argv) {
	dim->X=atoi(argv[1]);
	dim->Y=atoi(argv[2]);
	dim->Z=atoi(argv[3]);
	(*T)=atoi(argv[4]);
	proc->X=atoi(argv[5]);
	proc->Y=atoi(argv[6]);
	proc->Z=atoi(argv[7]);
	if (dim->X<=0 || dim->Y<=0 || dim->Z<=0 || T<=0 || proc->X<=0 || proc->Y<=0 || proc->Z<=0) {
		fprintf(stderr,"Input error: all dimensions must be positive\n");
		exit(1);
	}
}

void compute_local_dimensions(dimensions dim, dimensions proc,dimensions * ext_dim, dimensions * local_dim) {
	if (dim.X%proc.X==0) {
		local_dim->X=dim.X/proc.X;
		ext_dim->X=dim.X;
	}
	else {
		local_dim->X=(dim.X+(proc.X-dim.X%proc.X))/proc.X;
		ext_dim->X=local_dim->X*proc.X;
	}
	if (dim.Y%proc.Y==0) {
		local_dim->Y=dim.Y/proc.Y;
		ext_dim->Y=dim.Y;
	}
	else {
		local_dim->Y=(dim.Y+(proc.Y-dim.Y%proc.Y))/proc.Y;
		ext_dim->Y=local_dim->Y*proc.Y;
	}	
	if (dim.Z%proc.Z==0) {
		local_dim->Z=dim.Z/proc.Z;
		ext_dim->Z=dim.Z;
	}
	else {
		local_dim->Z=(dim.Z+(proc.Z-dim.Z%proc.Z))/proc.Z;
		ext_dim->Z=local_dim->Z*proc.Z;
	}
}

void compute_boundaries(dimensions dim,dimensions ext_dim, dimensions local_dim, dimensions * min, dimensions * max) {
	min->X=1; min->Y=1; min->Z=1;
	max->X=local_dim.X; max->Y=local_dim.Y; max->Z=local_dim.Z;
	if (ext_dim.X!=dim.X)
		max->X=local_dim.X-(ext_dim.X-dim.X);
	if (ext_dim.X!=dim.X)
		max->Y=local_dim.Y-(ext_dim.Y-dim.Y);
	if (ext_dim.Z!=dim.Z)
		max->Z=local_dim.Z-(ext_dim.Z-dim.Z);
}

void scatter_gather(double *** current, double *** local_current, double *** local_previous, dimensions ext_dim, dimensions local_dim, dimensions proc,int coords[], char pick, MPI_Comm cart_comm) {
	
	int * scatter_offset;
	int * sendcounts;
	double ** sndbuf, ** rcvbuf1, ** rcvbuf2;

	int offset=local_dim.X*local_dim.Y*local_dim.Z*coords[0]*proc.Z*proc.Y+local_dim.Z*local_dim.Y*coords[1]*proc.Z+local_dim.Z*coords[2];
	int my_count=1;
	if (rank==0) {
		scatter_offset=(int *)malloc(size*sizeof(int));
		sendcounts=(int*)malloc(size*sizeof(int));
	}
	MPI_Gather(&offset,1,MPI_INT,scatter_offset,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Gather(&my_count,1,MPI_INT,sendcounts,1,MPI_INT,0,MPI_COMM_WORLD);
	
	MPI_Datatype init_a,init_array,d_a,d_array;
	
	int sizes[3]={ext_dim.X,ext_dim.Y,ext_dim.Z};
	int subsizes[3]={local_dim.X,local_dim.Y,local_dim.Z};
	int start_coords[3]={coords[0]*local_dim.X,coords[1]*local_dim.Y,coords[2]*local_dim.Z};
	MPI_Aint order=MPI_ORDER_C;
	
	MPI_Type_create_subarray(3,sizes,subsizes,start_coords,order,MPI_DOUBLE,&init_a);
	MPI_Type_commit(&init_a);
	MPI_Type_create_resized(init_a,0,sizeof(double),&init_array);
	MPI_Type_commit(&init_array);

	int d_sizes[3]={local_dim.X+2,local_dim.Y+2,local_dim.Z+2};
	int d_start_coords[3]={0,0,0};
	MPI_Type_create_subarray(3,d_sizes,subsizes,d_start_coords,order,MPI_DOUBLE,&d_a);
	MPI_Type_commit(&d_a);
	MPI_Type_create_resized(d_a,0,sizeof(double),&d_array);
	MPI_Type_commit(&d_array);

	int c[3]={0,0,0};
	int cart_root_rank;
	MPI_Cart_rank(cart_comm,c,&cart_root_rank);
	if (rank==0)
		sndbuf=&(current[0][0][0]);
	rcvbuf1=&(local_previous[1][1][1]);
	rcvbuf2=&(local_current[1][1][1]);

	switch (pick) {
		case 's': 
			MPI_Scatterv(sndbuf,sendcounts,scatter_offset,init_array,rcvbuf1,1,d_array,cart_root_rank,cart_comm);
			MPI_Scatterv(sndbuf,sendcounts,scatter_offset,init_array,rcvbuf2,1,d_array,cart_root_rank,cart_comm);
			break;
		
		case 'g': 
			MPI_Gatherv(rcvbuf2,1,d_array,sndbuf,sendcounts,scatter_offset,init_array,cart_root_rank,cart_comm);
			break;	
	}
}

void pack(double * buf, double *** local, dimensions local_dim,char neighbour) {
	int i,j,k;
	switch (neighbour) {
		case 'u':
			for (i=1;i<=local_dim.Y;i++)
				for (j=1;j<=local_dim.Z;j++)
					buf[(i-1)*local_dim.Z+j-1]=local[1][i][j];
			break;
		case 'd':
			for (i=1;i<=local_dim.Y;i++)
				for (j=1;j<=local_dim.Z;j++)
					buf[(i-1)*local_dim.Z+j-1]=local[local_dim.X][i][j];
			break;
		case 'l':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Y;j++)
					buf[(i-1)*local_dim.Y+j-1]=local[i][j][1];
			break;
		case 'r':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Y;j++)
					buf[(i-1)*local_dim.Y+j-1]=local[i][j][local_dim.Z];
			break;
		case 'b':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Z;j++)
					buf[(i-1)*local_dim.Z+j-1]=local[i][1][j];
			break;
		case 'f':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Z;j++)
					buf[(i-1)*local_dim.Z+j-1]=local[i][local_dim.Y][j];
			break;
	}
}

void unpack(double * buf, double *** local, dimensions local_dim,char neighbour) {
	int i,j,k;
	switch (neighbour) {
		case 'u':
			for (i=1;i<=local_dim.Y;i++)
				for (j=1;j<=local_dim.Z;j++)
					local[0][i][j]=buf[(i-1)*local_dim.Z+j-1];
			break;
		case 'd':
			for (i=1;i<=local_dim.Y;i++)
				for (j=1;j<=local_dim.Z;j++)
					local[local_dim.X+1][i][j]=buf[(i-1)*local_dim.Z+j-1];
			break;
		case 'l':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Y;j++)
					local[i][j][0]=buf[(i-1)*local_dim.Y+j-1];
			break;
		case 'r':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Y;j++)
					local[i][j][local_dim.Z+1]=buf[(i-1)*local_dim.Y+j-1];
			break;
		case 'b':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Z;j++)
					local[i][0][j]=buf[(i-1)*local_dim.Z+j-1];
			break;
		case 'f':
			for (i=1;i<=local_dim.X;i++)
				for (j=1;j<=local_dim.Z;j++)
					local[i][local_dim.Y+1][j]=buf[(i-1)*local_dim.Z+j-1];
			break;
	}
}

void compute_new_boundaries(dimensions * min_b, dimensions * max_b, dimensions proc, int coords[3]) {
	if (coords[0]==0)
		min_b->X+=1;
	if (coords[1]==0)
		min_b->Y+=1;
	if (coords[2]==0)
		min_b->Z+=1;
	if (coords[0]==proc.X-1) 
		max_b->X-=1;
	if (coords[1]==proc.Y-1)
		max_b->Y-=1;
	if (coords[2]==proc.Z-1)
		max_b->Z-=1;
}

