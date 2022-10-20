#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cmath>
float dev_avg[32][3];

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

	int* data = (int*)malloc(sizeof(int)*w*h);
	fread(data, sizeof(int), w*h, fp);
	fclose(fp);

    for (int i = 0; i < nc; ++i){
        std::cin >> np[i];
        for (int j = 0; j < np[i]; ++j){
            int _x, _y;
            std::cin >> _x >> _y;

            int a = data[_y*w+_x];
            uint8_t x = a & 0xFF;
            uint8_t y = (a >> 8) & 0xFF;
            uint8_t z = (a >> 16) & 0xFF;

            avg[i][0] += x;
            avg[i][1] += y;
            avg[i][2] += z;
        }
        avg[i][0] /= np[i];
        avg[i][1] /= np[i];
        avg[i][2] /= np[i];
    }

    auto start_time = std::chrono::high_resolution_clock::now();

	for (int k = 0; k < w*h; ++k){
        int a = data[k];
        uint8_t x = a & 0xFF;
        uint8_t y = (a >> 8) & 0xFF;
        uint8_t z = (a >> 16) & 0xFF;

        char prediction;
        float best_distance = -INFINITY;
        for (char i = 0; i < nc; ++i){
            
            float current_distance = -(x-dev_avg[i][0])*(x-dev_avg[i][0])
                                     -(y-dev_avg[i][1])*(y-dev_avg[i][1])
                                     -(z-dev_avg[i][2])*(z-dev_avg[i][2]);
            if (current_distance > best_distance){
                prediction = i;
                best_distance = current_distance;
            }
        }
        int out = (x)|(y << 8)|(z << 16)|(prediction << 24);
        data[k] = out;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "Elapsed time in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " µs" << std::endl;
	
	fp = fopen(out_path.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(int), w*h, fp);
	fclose(fp);

	free(data);

    return 0;
}