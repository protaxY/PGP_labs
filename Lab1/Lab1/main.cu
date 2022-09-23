#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CSC(call) \
	cudaError_t error = call; \
	if (error != cudaSuccess)
		printf("%s", cudaGetErrorString); \


//nvcc -gencode arch=compute_50,code=sm_50 main.cu
// 
//const int n = 10000;

__global__ void kernel(long long n, double* a, double* b, double* c) {
	//threadIdx.x
	//blockIdx.x
	//blockDim.x
	//gridDim.x

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < n) {
		c[idx] = a[idx] + b[idx];
		idx += gridDim.x * blockDim.x;
		printf("%lf", c[idx]);
	}
}

int main () {

	int deviceCount;
	cudaDeviceProp devProp;

	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	printf("%d\n", error);

	printf("dev_count = %d\n", deviceCount);

	///

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&devProp, i);
		printf("Device %d\n", i);
		printf("Name %s\n", devProp.name);
		printf("Max threads (%d %d %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("Max blocks (%d %d %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	}

	///

	long long n;
	scanf("%lld", &n);

	double* a = (double*)malloc(sizeof(double) * n);
	double* b = (double*)malloc(sizeof(double) * n);
	double* c = (double*)malloc(sizeof(double) * n);

	for (long long i = 0; i < n; ++i) {
		scanf("%lf", &a[i]);
	}
	for (long long i = 0; i < n; ++i) {
		scanf("%lf", &b[i]);
	}

	double* dev_a;
	double* dev_b;
	double* dev_c;

	error = cudaMalloc(&dev_a, sizeof(double) * n);
	printf("malloc %d\n", error);
	error = cudaMalloc(&dev_b, sizeof(double) * n);
	printf("malloc %d\n", error);
	error = cudaMalloc(&dev_c, sizeof(double) * n);
	printf("malloc %d\n", error);

	error = cudaMemcpy(dev_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);
	printf("memcpy %d\n", error);
	error = cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);
	printf("memcpy %d\n", error);

	kernel<<<256, 256>> (n, dev_a, dev_b, dev_c);


	error = cudaGetLastError();
	printf("%s\n", cudaGetErrorString(error));

	error = cudaMemcpy(c, dev_c, sizeof(double) * n, cudaMemcpyDeviceToHost);
	printf("memcpy %d\n", error);

	for (long long i = 0; i < n; ++i) {
		printf("%lf", c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);
	free(b);
	free(c);

	return 0;
}