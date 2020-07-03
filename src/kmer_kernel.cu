#include <stdio.h>
#include <cuda.h>
#include "tipos.h"

__global__ void ResetKmer(lint *kmer, ushort offset, int val, int nF)
{
  lint idx = threadIdx.x + (blockDim.x * blockIdx.x);

  lint start = idx * offset;
  lint end   = start + offset;

  for(lint id = start; id < end; id++)
  {
    if (id < nF)
    kmer[idx] = val;
  }
}

__global__ void ResetFreq(Freq *freq, ushort offset, int val, int nF)
{
  lint idx = threadIdx.x + (blockDim.x * blockIdx.x);

  lint start = idx * offset;
  lint end   = start + offset;

  for(lint id = start; id < end; id++)
  {
    if (id < nF)
    {
      freq[id].kmer = val;
      freq[id].count = val;
    }
  }
}

//Compute k-mer index
__global__ void ComputeKmer(char *seq, lint *kmer, const int k, lint nN, ushort offset)
{
  lint idx = threadIdx.x + (blockDim.x * blockIdx.x);

  lint start = idx * offset;
  lint end   = start + offset;

  for(lint id = start; id < end; id++)
  {
    lint index = 0;
    if (id < nN)
    {
      for( lint i = 0; i < k; i++ )
      {
        char nuc = seq[i + id];
        if (nuc != -1) //Verifica se ha alguem que nao deve ser processado
        {
          index += nuc * powf(4, ((k-1)-i));
        }
        else
        {
          index = -1;
          break;
        }
      }//End for i
      kmer[id] = index;// Value of the combination
    }
  }//End for id
}

//Compute k-mer frequency
__global__ void ComputeFreq(lint *kmer, Freq *freq, lint *start, int *length, lint nS, int k, int nF)
{

  int idx = threadIdx.x + (blockDim.x * blockIdx.x);

  if (idx < nS)
  {
    int end = start[idx] + (length[idx] + 1);
    for (int i = start[idx]; i < end; i++)// Each kmer
    {
      for (int j = start[idx]; j < (start[idx] + (length[idx]-k-1)) && j < nF; j++)
      {
        if (freq[j].kmer == -1)
        {
          freq[j].kmer = kmer[i];
          freq[j].count = 1;
          break;
        }
        if (freq[j].kmer == kmer[i])
        {
          freq[j].count++;
          break;
        }
      }
    }
  }
}

//New way to compute k-mer frequency
//__global__ void ComputeFreqNew(int *Index, Freq *Freq, lint *start, int *length, ushort offset, int fourk, lint nS)
//{
//
//  int blx = blockIdx.x;
//
//  int st = blx * offset;
//  int nd = st + offset;
//
//  for (int i = st; i < nd; i++)
//  {
//    int idx = start[i] + threadIdx.x;
//    int id_freq = (fourk * i) + Index[idx];
//    if (threadIdx.x < length[i]-1)
//    {
//      atomicAdd(&Freq[id_freq], 1);
//    }
//  }
//}
