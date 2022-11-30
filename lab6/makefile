CC = /usr/local/cuda-7.0/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -lcublas -lcurand -lGL -lGLU -lglut -lGLEW
SOURSES = main.cu
BIN = lab6
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)