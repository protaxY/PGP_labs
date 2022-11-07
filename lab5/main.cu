#include <stdio.h>

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

const uint BLOCK_SIZE = 8;
const uint LOG2_BLOCK_SIZE = 3;
const uint GRID_SIZE = 4;

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
            // printf("%d\n", temp[_index(BLOCK_SIZE - 1)]);
            temp[_index(BLOCK_SIZE - 1)] = 0;
        }

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
    uint blockId = blockIdx.x;
    while (blockId < sums_size) {
        if (blockId != 0) {
            data[blockDim.x * blockId + threadIdx.x] += sums[blockId];
        }

        blockId += gridDim.x;
    }
}

void scan(uint* dev_data, uint size) {
    if (size % BLOCK_SIZE != 0)
        size += BLOCK_SIZE - (size % BLOCK_SIZE);

    uint sums_size = size/BLOCK_SIZE;

    uint* dev_sums;
    CSC(cudaMalloc(&dev_sums, (sums_size * sizeof(uint))));
    scan_blocks_kernel << < GRID_SIZE, BLOCK_SIZE, _index(BLOCK_SIZE) * sizeof(uint) >> > (dev_data, dev_sums, sums_size);

    if (size <= BLOCK_SIZE)
        return;
    scan(dev_sums, sums_size);
    add_kernel << < GRID_SIZE, BLOCK_SIZE >> > (dev_data, dev_sums, size);
}

int main() {
    uint n;
    scanf("%u", &n);
    uint* data = (uint*)malloc(n * sizeof(uint));
    for (uint i = 0; i < n; ++i) {
        scanf("%u", &data[i]);
    }
    uint* dev_data;
    CSC(cudaMalloc(&dev_data, (n+1) * sizeof(uint)));
    CSC(cudaMemcpy(dev_data, data, n * sizeof(uint), cudaMemcpyHostToDevice));

    scan(dev_data, n+1);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_data, (n+1) * sizeof(uint), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n+1; ++i) {
        printf("%u ", data[i]);
    }

    return 0;
}