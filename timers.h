#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

typedef struct timer_s {
    struct timeval t1;
    struct timeval t2;
    double duration;
} timer_tt;

enum timer_acc_s {
	TIMER_ACC_TYPE_AVG,
	TIMER_ACC_TYPE_MIN, 
	TIMER_ACC_TYPE_MAX
};
typedef enum timer_acc_s timer_acc_t;

timer_tt *cg_timer, *norm_timer;
timer_tt *sol_timer, *spmv_timer, *ddot_timer, *daxpy_timer, 
		 *mpi_reduce_timer, *mpi_gather_timer, *idle_timer;

timer_tt *timer_init();
inline void timer_start(timer_tt *timer);
inline void timer_stop(timer_tt *timer);
double timer_report_sec(timer_tt *timer);
double timer_report_accumulated_sec(timer_tt *timer, timer_acc_t acc);
void timer_report_all_sec(timer_tt *timer, double *buffer);

#endif
