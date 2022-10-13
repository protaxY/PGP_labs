#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

struct pixel{
    ushort x;
    ushort y;
    ushort z;
    ushort w;
};

int median_filter(int &y, int &x, int &h, int &w, int* data, int &r){
    int left_boundry = std::max(0, x-r);
    int right_boundry = std::min(w-1, x+r);
    int top_boundry = std::max(0, y-r);
    int bottom_boundry = std::min(h-1, y+r);

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
            // std::cout << i << ' ' << j << '\n';
            int a = data[i*w+j];
            uint8_t x = a & 0xFF;
            uint8_t y = (a >> 8) & 0xFF;
            uint8_t z = (a >> 16) & 0xFF;
            uint8_t w = (a >> 24) & 0xFF;
            ++counts_x[x];
            ++counts_y[y];
            ++counts_z[z];
        }
    }

    
    uint8_t x_out, y_out, z_out, w_out = 0;
    for (int k = 0; k < 3; ++k){
            for (int i = 0; i < 256; ++i) {
            if (i > 0)
                counts[k][i] += counts[k][i-1];
            if (counts[k][i] > n/2) {
                if (k == 0)
                    x_out = i;
                if (k == 1)
                    y_out = i;
                if (k == 2)
                    z_out = i;
                break;
            }
        }
    }

    // std::cout << (uint)x_out << (uint)y_out << (uint)z_out << '\n';
    int median_p = (x_out)|(y_out << 8)|(z_out << 16)|(w_out << 24);
    return median_p;
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

	int* data = (int*)malloc(sizeof(int) * w * h);
	fread(data, sizeof(int), w * h, fp);
	fclose(fp);

    int* processed_data = (int*)malloc(sizeof(int) * w * h);

    auto start_time = std::chrono::high_resolution_clock::now();


    // for (int y = 0; y < h; ++y){
    //     for (int x = 0; x < w; ++x){
    //         std::cout << data[y*w+x] << ' ';
    //     }
    //     std::cout << '\n';
    // }

    for (int y = 0; y < h; ++y){
        for (int x = 0; x < w; ++x){
            processed_data[y*w+x] = median_filter(y, x, h, w, data, r);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "Elapsed time in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " µs" << std::endl;

	fp = fopen(out_path.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(processed_data, sizeof(int), w * h, fp);
	fclose(fp);

	free(data);

    return 0;
}