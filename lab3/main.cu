#include <stdio.h>
#include <iostream> 
#include <string>

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)	

__constant__ float dev_avg[32][3];

__global__ void kernel(uchar4* data, int nc, int w, int h) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < w*h) {
        char predicion;
        float best_distance = -INFINITY;
        for (char i = 0; i < nc; ++i){
            float current_distance = -(data[idx].x-dev_avg[i][0])*(data[idx].x-dev_avg[i][0])
                                     -(data[idx].y-dev_avg[i][1])*(data[idx].y-dev_avg[i][1])
                                     -(data[idx].z-dev_avg[i][2])*(data[idx].z-dev_avg[i][2]);
            if (current_distance > best_distance){
                predicion = i;
                best_distance = current_distance;
            }
        }
        data[idx].w = predicion;
		idx += gridDim.x * blockDim.x;
	}
}

int main()
{
	int w, h;
    int nc;

    std::string in_path, out_path;
    std::cin >> in_path >> out_path;

    std::cin >> nc;
    long long np[nc];
    float avg[32][3];
    for (int i = 0; i < nc; ++i)
        for (int j = 0; j < 3; ++j)
            avg[i][j] = 0;

	FILE* fp = fopen(in_path.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);

	uchar4* data = (uchar4*)malloc(sizeof(uchar4)*w*h);
	fread(data, sizeof(uchar4), w*h, fp);
	fclose(fp);

    for (int i = 0; i < nc; ++i){
        std::cin >> np[i];
        for (int j = 0; j < np[i]; ++j){
            int x, y;
            std::cin >> x >> y;
            avg[i][0] += data[y*w+x].x;
            avg[i][1] += data[y*w+x].y;
            avg[i][2] += data[y*w+x].z;
        }
        avg[i][0] /= np[i];
        avg[i][1] /= np[i];
        avg[i][2] /= np[i];
    }

	uchar4* dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(uchar4)*w*h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4)*w*h, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(dev_avg, &avg, 32*3*sizeof(float)));

	kernel <<<256, 256>>> (dev_data, nc, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_data, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));

	fp = fopen(out_path.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w*h, fp);
	fclose(fp);

	free(data);

    return 0;
}