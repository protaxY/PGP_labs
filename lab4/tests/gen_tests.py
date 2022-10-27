import random
import numpy as np

# N = [10, 10**2, 10**3, 10**4]
N = [20, 30, 50]

# for i, n in enumerate(N):
#     matrix = (100*np.random.random((n, n))).astype(int)
#     with open(f'{i+1}.txt', 'w') as fp:
#         fp.write(str(n)+' ')
#         fp.write(str(n)+' ')
#         for x in range(matrix.shape[0]):
#             for y in range(matrix.shape[1]):
#                 fp.write(str(matrix[x, y])+' ')
#             fp.write('\n')
#     with open(f'{i+1}.ans', 'w') as fp:
#         fp.write(str(np.linalg.matrix_rank(matrix, 1e-7)))

for i, n in enumerate(N):
    matrix = np.empty((n, n+10))
    for i in range(n):
        for j in range(n+10):
            matrix[i, j] = i*n+j
    with open(f'{i+1}.txt', 'w') as fp:
        fp.write(str(n)+' ')
        fp.write(str(n+10)+' ')
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                fp.write(str(matrix[x, y])+' ')
            fp.write('\n')
    with open(f'{i+1}.ans', 'w') as fp:
        fp.write(str(np.linalg.matrix_rank(matrix, 1e-7)))