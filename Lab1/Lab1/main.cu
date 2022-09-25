#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)																									\

__global__ void kernel(long long n, double* a, double* b, double* c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < n) {
		c[idx] = a[idx] + b[idx];
		idx += gridDim.x * blockDim.x;
	}
}

int main () {
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

	CSC(cudaMalloc(&dev_a, sizeof(double) * n));
	CSC(cudaMalloc(&dev_b, sizeof(double) * n));
	CSC(cudaMalloc(&dev_c, sizeof(double) * n));

	CSC(cudaMemcpy(dev_a, a, sizeof(double) * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));

	kernel<<<256, 256>>> (n, dev_a, dev_b, dev_c);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(c, dev_c, sizeof(double) * n, cudaMemcpyDeviceToHost));

	for (long long i = 0; i < n; ++i) {
		printf("%.10e ", c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);

	free(b);
	free(c);

	return 0;
}