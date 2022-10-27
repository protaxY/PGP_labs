#include <stdio.h>
#include <vector>
#include <cmath>

using namespace std;

int main(){

    const double EPS = 1E-7;

    int n, m;
    scanf("%d %d", &n, &m);
    double a[n][m];
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j)
            scanf("%lf", &a[i][j]);
    }

    int rank = max(n,m);
    vector<char> line_used (n);
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
    printf("%d\n", rank);

    return 0;
}