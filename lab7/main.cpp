#include <iostream>
#include <algorithm>
#include <limits>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define _i(x, y, z) ((z+1)*((ny+2)*(nx+2))+(y+1)*(nx+2)+(x+1))
#define _ib(x, y, z) ((z)*(ynb*xnb)+(y)*xnb+(x))

// #define _i(x, y, z) ((x+1)*((ny+2)*(nz+2))+(y+1)*(nz+2)+(z+1))
// #define _ib(x, y, z) ((x)*(ynb*znb)+(y)*znb+(z))

// параметры отдельноо процесса
int id;
int xb, yb, zb;

// общий параметры
int xnb, ynb, znb;
int nx, ny, nz;
std::string save_path;
double eps;
double lx, ly, lz;
double bc_down, bc_up, bc_left, bc_right, bc_front, bc_back;
double u0;

// указатели на данные
double* data;
double* next_data;

// статус
MPI_Status status;

void broadcast_params(){
    MPI_Bcast(&xnb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ynb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&znb, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    if (id == 0){
        std::cin >> xnb >> ynb >> znb;
        std::cin >> nx >> ny >> nz;
        std::cin >> save_path;
        std::cin >> eps;
        std::cin >> lx >> ly >> lz;
        std::cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back;
        std::cin >> u0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    broadcast_params();

    xb = id%(xnb*ynb)%xnb;
    yb = id%(xnb*ynb)/xnb;
    zb = id/(xnb*ynb);

    double hx, hy, hz;
    hx = lx/(xnb*nx);
    hy = ly/(ynb*ny);
    hz = lz/(znb*nz);

    next_data = (double*)malloc((nx+2)*(ny+2)*(nz+2)*sizeof(double));
    data = (double*)malloc((nx+2)*(ny+2)*(nz+2)*sizeof(double));
    double* eps_buff = (double*)malloc(xnb*ynb*znb*sizeof(double));
    double max_eps = 10000000.0;

    // инициализация начального состояния блока
    for (int z = -1; z <= nz; ++z){   
        for (int y = -1; y <= ny; ++y){
            for (int x = -1; x <= nx; ++x){
                data[_i(x, y, z)] = u0;
                // next_data[_i(x, y, z)] = u0;
            }
        }
    }

    double* send_up_down_buff = (double*)malloc(ny*nx*sizeof(double));
    double* send_left_right_buff = (double*)malloc(nz*ny*sizeof(double));
    double* send_front_back_buff = (double*)malloc(nz*nx*sizeof(double));

    double* recv_up_down_buff = (double*)malloc(ny*nx*sizeof(double));
    double* recv_left_right_buff = (double*)malloc(nz*ny*sizeof(double));
    double* recv_front_back_buff = (double*)malloc(nz*nx*sizeof(double));

    while(max_eps > eps){
        MPI_Barrier(MPI_COMM_WORLD);
        // отправка (right. back, up)
            // right
            if (xb < xnb-1){
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        send_left_right_buff[z*ny+y] = data[_i(nx-1, y, z)];
                    }
                }
                MPI_Send(send_left_right_buff, nz*ny, MPI_DOUBLE, _ib(xb+1, yb, zb), id, MPI_COMM_WORLD);
            }
            // back
            if (yb < ynb-1){
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        send_front_back_buff[z*nx+x] = data[_i(x, ny-1, z)];
                    }
                }
                MPI_Send(send_front_back_buff, nz*nx, MPI_DOUBLE, _ib(xb, yb+1, zb), id, MPI_COMM_WORLD);
            }
            // up
            if (zb < znb-1){
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        send_up_down_buff[y*nx+x] = data[_i(x, y, nz-1)];
                    }
                }
                MPI_Send(send_up_down_buff, ny*nx, MPI_DOUBLE, _ib(xb, yb, zb+1), id, MPI_COMM_WORLD);
            }
        // прием (left, front, down)
            // left
            if (xb > 0){
                // std::cout << id << "left\n";
                MPI_Recv(recv_left_right_buff, nz*ny, MPI_DOUBLE, _ib(xb-1, yb, zb), _ib(xb-1, yb, zb), MPI_COMM_WORLD, &status);
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        data[_i(-1, y, z)] = recv_left_right_buff[z*ny+y];
                    }
                }
            } else {
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        data[_i(-1, y, z)] = bc_left;
                    }
                }
            } 
            // front
            if (yb > 0){
                // std::cout << id << "front\n";
                MPI_Recv(recv_front_back_buff, nz*nx, MPI_DOUBLE, _ib(xb, yb-1, zb), _ib(xb, yb-1, zb), MPI_COMM_WORLD, &status);
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, -1, z)] = recv_front_back_buff[z*nx+x];
                    }
                }
            } else {
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, -1, z)] = bc_front;
                    }
                }
            }
            // down
            if (zb > 0){
                // std::cout << id << "down\n";
                MPI_Recv(recv_up_down_buff, ny*nx, MPI_DOUBLE, _ib(xb, yb, zb-1), _ib(xb, yb, zb-1), MPI_COMM_WORLD, &status);
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, y, -1)] = recv_up_down_buff[y*nx+x];
                    }
                }
            } else {
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, y, -1)] = bc_down;
                    }
                }
            }
        MPI_Barrier(MPI_COMM_WORLD);
        // отправка (left, front, down)
            // left
            if (xb > 0){
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        send_left_right_buff[z*ny+y] = data[_i(0, y, z)];
                    }
                }
                MPI_Send(send_left_right_buff, nz*ny, MPI_DOUBLE, _ib(xb-1, yb, zb), id, MPI_COMM_WORLD);
            }
            // front
            if (yb > 0){
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        send_front_back_buff[z*nx+x] = data[_i(x, 0, z)];
                    }
                }
                MPI_Send(send_front_back_buff, nz*nx, MPI_DOUBLE, _ib(xb, yb-1, zb), id, MPI_COMM_WORLD);
            }
            // down
            if (zb > 0){
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        send_up_down_buff[y*nx+x] = data[_i(x, y, 0)];
                    }
                }
                MPI_Send(send_up_down_buff, ny*nx, MPI_DOUBLE, _ib(xb, yb, zb-1), id, MPI_COMM_WORLD);
            }
        // прием (right, back, up)
            // right
            if (xb < xnb-1){
                MPI_Recv(recv_left_right_buff, nz*ny, MPI_DOUBLE, _ib(xb+1, yb, zb), _ib(xb+1, yb, zb), MPI_COMM_WORLD, &status);
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        data[_i(nx, y, z)] = recv_left_right_buff[z*ny+y];
                    }
                }
            } else {
                for (int z = 0; z < nz; ++z){
                    for (int y = 0; y < ny; ++y){
                        data[_i(nx, y, z)] = bc_right;
                    }
                }
            }
            // back
            if (yb < ynb-1){
                MPI_Recv(recv_front_back_buff, nz*nx, MPI_DOUBLE, _ib(xb, yb+1, zb), _ib(xb, yb+1, zb), MPI_COMM_WORLD, &status);
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, ny, z)] = recv_front_back_buff[z*nx+x];
                    }
                }
            } else {
                for (int z = 0; z < nz; ++z){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, ny, z)] = bc_back;
                    }
                }
            }
            // up
            if (zb < znb-1){
                MPI_Recv(recv_up_down_buff, ny*nx, MPI_DOUBLE, _ib(xb, yb, zb+1), _ib(xb, yb, zb+1), MPI_COMM_WORLD, &status);
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, y, nz)] = recv_up_down_buff[y*nx+x];
                    }
                }
            } else {
                for (int y = 0; y < ny; ++y){
                    for (int x = 0; x < nx; ++x){
                        data[_i(x, y, nz)] = bc_up;
                    }
                }
            }
        MPI_Barrier(MPI_COMM_WORLD);

        // итерация
        double eps = -10000000.0;
        for (int z = 0; z < nz; ++z){
            for (int y = 0; y < ny; ++y){
                for (int x = 0; x < nx; ++x){
                    next_data[_i(x, y, z)] = ((data[_i(x+1, y, z)]+data[_i(x-1, y, z)])/(hx*hx) +
                                              (data[_i(x, y+1, z)]+data[_i(x, y-1, z)])/(hy*hy) + 
                                              (data[_i(x, y, z+1)]+data[_i(x, y, z-1)])/(hz*hz)) /
                                              (2.0*((1/(hx*hx))+(1/(hy*hy))+(1/(hz*hz))));
                    eps = std::max(eps, std::abs(next_data[_i(x, y, z)]-data[_i(x, y, z)]));                                       
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&eps, 1, MPI_DOUBLE, eps_buff, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        max_eps = -10000000.0;
        for (int i = 0; i < znb*ynb*xnb; ++i){
            max_eps = std::max(max_eps, eps_buff[i]);
        }
            
        // обмен указателями
        double* tmp = data;
        data = next_data;
        next_data = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // сборка результата
    double* buff = (double*)malloc(nx*sizeof(double));
    if (id != 0){
        for (int z = 0; z < nz; ++z){
            for (int y = 0; y < ny; ++y){
                for (int x = 0; x < nx; ++x){
                    buff[x] = data[_i(x, y, z)];
                }
                MPI_Send(buff, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    }
    else {
        FILE* fptr = fopen(save_path.c_str(), "w");
        for (int z_b = 0; z_b < znb; ++z_b){
            for (int z = 0; z < nz; ++z){
                for (int y_b = 0; y_b < ynb; ++y_b){
                    for (int y = 0; y < ny; ++y){
                        for (int x_b = 0; x_b < xnb; ++x_b){
                            if (_ib(x_b, y_b, z_b) == 0){
                                for (int x = 0; x < nx; ++x){
                                    buff[x] = data[_i(x, y, z)];
                                }
                            }
                            else {
                                MPI_Recv(buff, nx, MPI_DOUBLE, _ib(x_b, y_b, z_b), _ib(x_b, y_b, z_b), MPI_COMM_WORLD, &status);
                            }

                            for (int x = 0; x < nx; ++x){
                                fprintf(fptr, "%.6e ", buff[x]);
                            }
                        }
                        fprintf(fptr, "\n");
                    }
                }
                fprintf(fptr, "\n");
            }
        }
        fclose(fptr);
    }

    MPI_Finalize();

    free(buff);

    free(send_up_down_buff);
    free(send_left_right_buff);
    free(send_front_back_buff);

    free(recv_up_down_buff);
    free(recv_left_right_buff);
    free(recv_front_back_buff);

    free(data);
    free(next_data);
    free(eps_buff);

    return 0;
}