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

__global__ void kernel(uchar4 * out, cudaTextureObject_t texObj, int w, int h, int r) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 p;
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			int left_boundry = max(0, x-r);
			int right_boundry = min(w-1, x+r);
			int top_boundry = max(0, y-r);
			int bottom_boundry = min(h-1, y+r);

			ushort counts[3][256];
			ushort* counts_x = counts[0];
			ushort* counts_y = counts[1];
			ushort* counts_z = counts[2];
			ushort n = (right_boundry-left_boundry+1)*(bottom_boundry-top_boundry+1);

			for (int i = 0; i < 256; ++i) {
				counts_x[i] = 0;
				counts_y[i] = 0;
				counts_z[i] = 0;
			}

			for (int i = top_boundry; i <= bottom_boundry; ++i) {
				for (int j = left_boundry; j <= right_boundry; ++j) {
                    p = tex2D<uchar4>(texObj, j, i);
					++counts_x[p.x];
					++counts_y[p.y];
					++counts_z[p.z];
				}
			}

			uchar4 median_p;

			for (int k = 0; k < 3; ++k){
					for (int i = 0; i < 256; ++i) {
					if (i > 0)
						counts[k][i] += counts[k][i-1];
					if (counts[k][i] > n/2) {
						if (k == 0)
							median_p.x = i;
						if (k == 1)
							median_p.y = i;
						if (k == 2)
							median_p.z = i;
						break;
					}
				}
			}

			out[y * w + x] = make_uchar4(median_p.x, median_p.y, median_p.z, p.w);
		}
}

int main()
{
	int w, h;
    int r;

    std::string in_path, out_path;
    std::cin >> in_path >> out_path >> r;

	FILE* fp = fopen(in_path.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);

	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	uchar4* dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));


	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));

	kernel <<< dim3(16, 16), dim3(32, 32) >>> (dev_out, texObj, w, h, r);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float time;
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("time = %f ms \n", time);

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(out_path.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);

    return 0;
}