
CC=g++
NC=/usr/local/cuda/bin/nvcc
CFLAGS=-std=c++14 -O3
NFLAGS=-arch=sm_86

SHARED=../shared

all: converter

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS)
			
clean:
	rm -f converter
