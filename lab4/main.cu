#include <stdio.h>
#include <algorithm>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)	

struct comparator {												
	__host__ __device__ bool operator()(double a, double b) {
		return abs(a) < abs(b);
	}
};

__global__ void swap_kernel(double *sub_matrix, int n, int m, int y, int x, int max_row_index){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (y+idx < m){
        double tmp = sub_matrix[(y+idx)*n+x];
        sub_matrix[(y+idx)*n+x] = sub_matrix[(y+idx)*n+max_row_index];
        sub_matrix[(y+idx)*n+max_row_index] = tmp;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void preparation_kernel(double* sub_matrix, int n, int m, int x, int y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (x+1+idx < n){
        sub_matrix[y*n+(x+1+idx)] /= sub_matrix[y*n+x];
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void kernel(double* sub_matrix, int n, int m, int x, int y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    while (y+1+idy < m){
        idx = blockDim.x * blockIdx.x + threadIdx.x;
        while (x+1+idx < n){
            sub_matrix[(y+1+idy)*n+(x+1+idx)] -= sub_matrix[(y+1+idy)*n+x]*sub_matrix[y*n+(x+1+idx)];
            idx += blockDim.x * gridDim.x;
        }
        idy += blockDim.y * gridDim.y;
    }
}

int main() {
    comparator comp;
    int n, m;
    scanf("%d%d", &n, &m);
    double* matrix = (double*)malloc(sizeof(double)*n*m);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j)
            scanf("%lf", &matrix[j*n+i]);
    }

    double *dev_matrix;
	CSC(cudaMalloc(&dev_matrix, sizeof(double)*n*m));
	CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double)*n*m, cudaMemcpyHostToDevice));

    int rank = m;
    

    cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));


    int j = 0;
    for (int i = 0; i < m and j <= n; ++i){
        if (j == n){
            --rank;
            continue;
        }
        thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(dev_matrix);
	    thrust::device_ptr<double> p_max_element = thrust::max_element(p_matrix+i*n+j, p_matrix+i*n+n, comp);
        double max_value = *p_max_element;
        if (abs(max_value) < 1e-7){
            --rank;
            
            continue;
        }

        int max_row_index = (int)(p_max_element-(p_matrix+i*n));
        if (j != max_row_index){
            swap_kernel<<< 32, 32 >>> (dev_matrix, n, m, i, j, max_row_index);
            CSC(cudaGetLastError());
        }
        preparation_kernel<<< 32, 32 >>> (dev_matrix, n, m, j, i);
        kernel<<< dim3(32, 32), dim3(32, 32) >>> (dev_matrix, n, m, j, i);
        CSC(cudaGetLastError());
        ++j;
    }

    CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float time;
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("time = %f ms \n", time);



    cudaFree(dev_matrix);
	free(matrix);

	return 0;
}