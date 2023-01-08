CC = /usr/local/cuda/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
BIN = kp
all:
	$(CC) $(CFLAGS) cp.cu -o $(BIN)