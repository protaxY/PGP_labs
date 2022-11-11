#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>

typedef unsigned int uint; 

void gen_b(uint* a, uint k, uint n, bool* b){
    for (uint i = 0; i < n; ++i){
        b[i] = (a[i]>>k)&1;
    }
}

void gen_s(bool* b, uint n, uint* s){

    for (uint i = 0; i < n; ++i){
        s[i] = b[i];
    }
}

void scan(uint* s, uint n){
    for (int i = 1; i < n; ++i){
        s[i] += s[i-1];
    }
}

void binary_digit_sort(uint* a, bool* b, uint* s, uint size, uint* res){

    for(int i = 0; i < size; ++i){
        if (b[i])
            res[s[i]+(size-s[size])] = a[i];
        else
            res[i-s[i]] = a[i];
    }
}

void sort(uint* data, uint size){
    uint* prev_res = data;

    uint* res = (uint*)malloc(size*sizeof(uint));
    bool* b = (bool*)malloc(size*sizeof(bool));
    uint* s = (uint*)malloc((size+1)*sizeof(uint));

    for (int k = 0; k < 32; ++k){
        gen_b(res, k, size, b);
        gen_s(b, size+1, s);
        scan(s, size+1);
        binary_digit_sort(prev_res, b, s, size, res);
        std::swap(prev_res, res);
    }
    free(b);
    free(s);
    free(res);
}

int main() {
    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);

    uint n;
    fread(&n, sizeof(uint), 1, stdin);

    uint* data = (uint*)malloc(n*sizeof(uint));
    fread(data, sizeof(uint), n, stdin);

    auto start_time = std::chrono::high_resolution_clock::now();

    sort(data, n);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "Elapsed time in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " µs" << std::endl;

    free(data);

    return 0;
}