#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

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

    auto start_time = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < n; ++i){
        c[i] = a[i] + b[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "Elapsed time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " µs" << std::endl;

	free(a);
	free(b);
	free(c);

	return 0;
}