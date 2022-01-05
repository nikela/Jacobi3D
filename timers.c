#include "timers.h"
#include <stdlib.h>
//#include <mpi.h>
#include <stdio.h>

timer_tt *timer_init()
{
	timer_tt *timer;
	timer = (timer_tt *)malloc(sizeof(timer_tt));
	timer->duration = 0;
	return timer;
}

inline void timer_start(timer_tt *timer)
{
	gettimeofday(&timer->t1,0);
}

inline void timer_stop(timer_tt *timer)
{
	gettimeofday(&timer->t2,0);
	timer->duration += (double)((timer->t2.tv_sec-timer->t1.tv_sec)*1000000 \
								+ timer->t2.tv_usec - timer->t1.tv_usec)/1000000;
}

double timer_report_sec(timer_tt *timer)
{
	return timer->duration;
}

double average (double *arr, int n)
{
	int i;
	double sum = 0.0;

	for (i=0; i < n; i++) sum += arr[i];

	return sum / (double) n;

}

/*
double min (double *arr, int n)
{
	int i;
	double min = arr[0];

	for (i=1; i < n; i++) 
		if (arr[i] < min)
			min = arr[i];

	return min;

}

double max (double *arr, int n)
{
	int i;
	double max = arr[0];

	for (i=1; i < n; i++)
		if (arr[i] > max)
			max = arr[i];

	return max;

} */
/*
double timer_report_accumulated_sec(timer_tt *timer, timer_acc_t acc) {
	int size, rank;
	double *buffer, ret = 0;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)	buffer = (double *)malloc(size * sizeof(double));
	timer_report_all_sec(timer, buffer);

	if (rank == 0) {
		switch(acc) {
			case TIMER_ACC_TYPE_AVG:
				ret = average(buffer, size);
				break;
			case TIMER_ACC_TYPE_MIN:
				ret = min(buffer, size);
				break;
			case TIMER_ACC_TYPE_MAX:
				ret = max(buffer, size);
		}
		free(buffer);
	}

	return ret;
}

void timer_report_all_sec(timer_tt *timer, double *buffer) {
	int size, rank;
	double time;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	time = timer_report_sec(timer);
	MPI_Gather(&time, 1, MPI_DOUBLE, buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
*/
