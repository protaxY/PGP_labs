#include <stdio.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

// using namespace std;

int main(){

    const double EPS = 1E-7;

    int n, m;
    scanf("%d %d", &n, &m);
    
    std::vector<std::vector<double>> a(n, std::vector<double> (m));
    // double a[n][m];
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j)
            scanf("%lf", &a[i][j]);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();

    int rank = std::max(n,m);
    std::vector<char> line_used (n);
    for (int i=0; i<m; ++i) {
        int j;
        for (j=0; j<n; ++j)
            if (!line_used[j] && abs(a[j][i]) > EPS)
                break;
        if (j == n)
            --rank;
        else {
            line_used[j] = true;
            for (int p=i+1; p<m; ++p)
                a[j][p] /= a[j][i];
            for (int k=0; k<n; ++k)
                if (k != j && abs(a[k][i]) > EPS)
                    for (int p=i+1; p<m; ++p)
                        a[k][p] -= a[j][p] * a[k][i];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "Elapsed time in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " µs" << std::endl;

    return 0;
}