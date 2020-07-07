#ifndef _kmer_cuh
#define _kmer_cuh

#include "tipos.h"

void ReadFASTASequences(char *file, lint *nN, lint *nS, struct read *rd, ushort flag);

void kmer_main(Read *rd, lint nN, lint nS, int k, ushort device, cudaStream_t stream);

__global__ void ResetFreq(Freq *freq, ushort offset, int val, int nF);

__global__ void ResetKmer(lint *kmer, ushort offset, int val, int nF);

__global__ void ComputeKmer(char *seq, lint *kmer, const int k, lint nN, ushort offset);

__global__ void ComputeFreq(lint *kmer, Freq *freq, lint *start, int *length, lint nS, int k, int nF);

//__global__ void ComputeFreqNew(int *Index, int *Freq, lint *start, int *length, ushort offset, int fourk, lint nS);

#endif
