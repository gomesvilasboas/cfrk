#NVCCFLAGS=-ccbin=icpc -lm -m64 -arch=sm_35 --use_fast_math --default-stream per-thread --ptxas-options=-O3
#INTELFLAGS=-Xcompiler="-qopenmp -Ofast -pthread"
NVCCFLAGS=-lm -m64 -arch=sm_50 --default-stream per-thread --ptxas-options=-O3 -lineinfo#--use_fast_math
CXXFLAGS=-Xcompiler -fopenmp -lpthread -O3 -g -v
default: cfrk

cfrk:
	nvcc $(CXXFLAGS) $(NVCCFLAGS) kmer_kernel.cu -c
	nvcc $(CXXFLAGS) $(NVCCFLAGS) kmer_main.cu -c
	nvcc $(CXXFLAGS) $(NVCCFLAGS) fastaIO.cu -c
	nvcc main.cu $(INTELFLAGS) $(NVCCFLAGS) -c
	nvcc main.o fastaIO.o kmer_kernel.o kmer_main.o $(NVCCFLAGS) $(CXXFLAGS) -o ../bin/cfrk
	rm main.o kmer_main.o kmer_kernel.o fastaIO.o
clean:
	rm -f ../bin/* ./*.o
