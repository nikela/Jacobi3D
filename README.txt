Versions:
Serial			-- Jacobi3D_serial.c		--  Usage: ./jacobi_serial X Y Z T
OpenMP			-- Jacobi3D_omp.c		--  Usage: ./jacobi_omp X Y Z T threads
MPI			-- Jacobi3D_mpi.c		--  Usage: ./jacobi_mpi X Y Z T PROCS.X PROCS.Y PROCS.Z
MPI-Overlapping		-- Jacobi3D_mpi_overlapping.c	--  Usage: ./jacobi_mpi_overlapping X Y Z T PROCS.X PROCS.Y PROCS.Z
Hybrid MPI/OpenMP	-- Jacobi3D_hybrid.c		--  Usage: ./jacobi_hybrid X Y Z T PROCS.X PROCS.Y PROCS.Z threads

T: iterations of the Jacobi-3D kernel of computations
X, Y, Z: dimensions of the 3D-space 
PROCS.X, PROCS.Y, PROCS.Z: dimensions of the 3D-cartesian processor topology (processes = PROCS.X * PROCS.Y * PROCS.Z)
threads: number of OpenMP threads in OpenMP and hybrid version
