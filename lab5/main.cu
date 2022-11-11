#include <stdio.h>
#include <algorithm>

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)

#define _index(i) ((i) + ((i) >> 5))

typedef unsigned int uint; 

const uint BLOCK_SIZE = 128;
const uint LOG2_BLOCK_SIZE = 7;
const uint GRID_SIZE = 64;

__global__ void scan_blocks_kernel(uint* data, uint* sums, uint sums_size) {
    int blockId = blockIdx.x;
    while (blockId < sums_size) {
        extern __shared__ uint temp[];
        temp[_index(threadIdx.x)] = data[blockDim.x * blockId + threadIdx.x];
        __syncthreads();

        uint stride = 1;
        for (uint d = 0; d < LOG2_BLOCK_SIZE; ++d) {
            if ((threadIdx.x + 1) % (stride<<1) == 0) {
                temp[_index(threadIdx.x)] += temp[_index(threadIdx.x-stride)];
            }
            stride <<= 1;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            sums[blockId] = temp[_index(BLOCK_SIZE - 1)];
            temp[_index(BLOCK_SIZE - 1)] = 0;
        }
        __syncthreads();

        stride = 1 << LOG2_BLOCK_SIZE-1;
        while (stride != 0) {
            if ((threadIdx.x + 1) % (stride<<1) == 0) {
                uint tmp = temp[_index(threadIdx.x)];
                temp[_index(threadIdx.x)] += temp[_index(threadIdx.x - stride)];
                temp[_index(threadIdx.x - stride)] = tmp;
            }
            stride >>= 1;
            __syncthreads();
        }

        data[blockDim.x * blockId + threadIdx.x] = temp[_index(threadIdx.x)];

        blockId += gridDim.x;
    }
}

__global__ void add_kernel(uint* data, uint* sums, uint sums_size) {
    uint blockId = blockIdx.x+1; 
    while (blockId < sums_size) {
        if (blockId != 0) {
            data[blockDim.x * blockId + threadIdx.x] += sums[blockId];
        }

        blockId += gridDim.x;
    }
}

__global__ void b_gen_kernel(uint* a, uint k, uint n, bool* b){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n){
        b[idx] = (a[idx]>>k)&1;

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void s_gen_kernel(bool* b, uint n, uint* s){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n){
        s[idx] = b[idx];

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void binary_digit_sort_kernel(uint* a, bool* b, uint* s, uint size, uint* res){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < size){
        if (b[idx])
            res[s[idx]+(size-s[size])] = a[idx];
        else 
            res[idx-s[idx]] = a[idx];
        
        idx += gridDim.x * blockDim.x;
    }
}

void scan(uint* dev_data, uint size) {
    if (size % BLOCK_SIZE != 0)
        size += BLOCK_SIZE - (size % BLOCK_SIZE);
    uint sums_size = size/BLOCK_SIZE;

    uint* dev_sums;
    CSC(cudaMalloc(&dev_sums, (sums_size * sizeof(uint))));
    scan_blocks_kernel << < GRID_SIZE, BLOCK_SIZE, _index(BLOCK_SIZE) * sizeof(uint) >> > (dev_data, dev_sums, sums_size);
    CSC(cudaGetLastError());

    if (size <= BLOCK_SIZE)
        return;
    scan(dev_sums, sums_size);

    add_kernel << < GRID_SIZE, BLOCK_SIZE >> > (dev_data, dev_sums, sums_size);
    CSC(cudaGetLastError());

    cudaFree(dev_sums);
}

void sort(uint* &dev_data, uint size){
    uint* dev_prev_res = dev_data;
    uint* dev_res;
    CSC(cudaMalloc(&dev_res, (size*sizeof(uint))));

    bool* dev_b;
    uint* dev_s;
    CSC(cudaMalloc(&dev_b, (size*sizeof(bool))));
    CSC(cudaMalloc(&dev_s, ((size+1)*sizeof(uint))));
    
    for (int k = 0; k < 32; ++k){
        b_gen_kernel<<< GRID_SIZE, BLOCK_SIZE >>> (dev_prev_res, k, size, dev_b);
        CSC(cudaGetLastError());
        s_gen_kernel<<< GRID_SIZE, BLOCK_SIZE >>> (dev_b, size, dev_s);
        CSC(cudaGetLastError());

        scan(dev_s, size+1);

        binary_digit_sort_kernel<<< GRID_SIZE, BLOCK_SIZE >>> (dev_prev_res, dev_b, dev_s, size, dev_res);
        CSC(cudaGetLastError());

        std::swap(dev_prev_res, dev_res);
    }
    cudaFree(dev_b);
    cudaFree(dev_s);
    cudaFree(dev_res);
}

int main() {
    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);

    uint n;
    fread(&n, sizeof(uint), 1, stdin);

    uint* data = (uint*)malloc(n * sizeof(uint));
    fread(data, sizeof(uint), n, stdin);

    uint* dev_data;
    CSC(cudaMalloc(&dev_data, n * sizeof(uint)));
    CSC(cudaMemcpy(dev_data, data, n * sizeof(uint), cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));

    sort(dev_data, n);
    
    CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float time;
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("time = %f ms \n", time);


    CSC(cudaMemcpy(data, dev_data, (n) * sizeof(uint), cudaMemcpyDeviceToHost));

    // fwrite(data, sizeof(uint), n, stdout);

    cudaFree(dev_data);
    free(data);

    return 0;
}