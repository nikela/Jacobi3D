CC=gcc
MCC=mpicc
CFLAGS=-O3 -o
OMP=-fopenmp

all: jacobi_serial jacobi_omp jacobi_mpi jacobi_mpi_overlapping jacobi_hybrid 

jacobi_serial: Jacobi3D_serial.c jacobi_utils.c timers.c
	$(CC) $(CFLAGS) jacobi_serial Jacobi3D_serial.c jacobi_utils.c timers.c
jacobi_omp: Jacobi3D_omp.c jacobi_utils.c timers.c
	$(CC) $(OMP) $(CFLAGS) jacobi_omp Jacobi3D_omp.c jacobi_utils.c timers.c
jacobi_mpi: Jacobi3D_mpi.c jacobi_utils.c timers.c
	$(MCC) $(CFLAGS) jacobi_mpi Jacobi3D_mpi.c jacobi_utils.c timers.c
jacobi_mpi_overlapping: Jacobi3D_mpi_overlapping.c jacobi_utils.c timers.c
	$(MCC) $(CFLAGS) jacobi_mpi_overlapping Jacobi3D_mpi_overlapping.c jacobi_utils.c timers.c
jacobi_hybrid: Jacobi3D_hybrid.c jacobi_utils.c timers.c
	$(MCC) $(OMP) $(CFLAGS) jacobi_hybrid Jacobi3D_hybrid.c jacobi_utils.c timers.c

clean: 
	rm jacobi_serial jacobi_omp jacobi_mpi jacobi_mpi_overlapping jacobi_hybrid 

