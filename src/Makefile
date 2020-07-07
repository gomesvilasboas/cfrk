#NVCCFLAGS=-ccbin=icpc -lm -m64 -arch=sm_35 --use_fast_math --default-stream per-thread --ptxas-options=-O3
#INTELFLAGS=-Xcompiler="-qopenmp -Ofast -pthread"
NVCCFLAGS=-lm -m64 -arch=sm_50 --default-stream per-thread -lgomp --ptxas-options=-O3 -O3# --use_fast_math
CXXFLAGS=-Xcompiler -fopenmp
#default: cfrk

all:
	nvcc $(NVCCFLAGS) $(CXXFLAGS) kmer_kernel.cu -c
	nvcc $(NVCCFLAGS) $(CXXFLAGS) kmer_main.cu -c
	nvcc $(NVCCFLAGS) $(CXXFLAGS) fastaIO.cu -c
	nvcc main.cu $(NVCCFLAGS) $(CXXFLAGS) -c
	nvcc main.o fastaIO.o kmer_kernel.o kmer_main.o $(NVCCFLAGS) $(CXXFLAGS) -o ../bin/cfrk
	rm main.o kmer_main.o kmer_kernel.o fastaIO.o
clean:
	rm -f ../bin/* ./*.o
