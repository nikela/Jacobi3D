#define ERR 0.00001

typedef struct {
	int X;
	int Y;
	int Z;
} dimensions;

int T;
double error;

int check_convergence(double *** current, double *** previous, int x_start,int x_end,int y_start,int y_end,int z_start, int z_end);
void compute_local_dimensions(dimensions dim, dimensions proc, dimensions * ext_dim, dimensions * local_dim);
double * allocate_1D(int dimX);
double ** allocate_2D(int dimX,int dimY);
double *** allocate_3D(int dimX,int dimY,int dimZ);
void initialize_random_3D (double *** array, int dimX, int dimY, int dimZ);
void initialize_constant_3D (double *** array, int dimX, int dimY, int dimZ);
void initialize_smooth_3D (double *** array, int dimX, int dimY, int dimZ);
void input_3D (double *** array, int dimX, int dimY, int dimZ);
void print_3D (double *** array, int dimX, int dimY, int dimZ);
