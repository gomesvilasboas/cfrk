#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "tipos.h"
#include "kmer.cuh"

void GetDeviceProp(uint8_t device, lint *maxGridSize, lint *maxThreadDim, lint *deviceMemory)
{
  cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, device);

  *maxThreadDim = prop.maxThreadsDim[0];
  *maxGridSize = prop.maxGridSize[0];
  *deviceMemory = prop.totalGlobalMem;
}

void kmer_main(Read *rd, lint nN, lint nS, int k, ushort device, cudaStream_t stream)
{
  lint *dKmer;// Index vector
  char *dSeq;// Seq matrix
  Freq *dFreq;
  lint *dStart;
  int *dLength;// The beggining and the length of each sequence
  lint block[4], grid[4];// Grid config; 0:nN, 1:nS
  lint maxGridSize, maxThreadDim, deviceMemory;// Device config
  ushort offset[4] = {1,1,1,1};
  size_t size[5], totalsize;

  dKmer =  NULL;
  dSeq = NULL;

  cudaSetDevice(device);
  GetDeviceProp(device, &maxGridSize, &maxThreadDim, &deviceMemory);

  //printf("omp_in_parallel: %d\n", omp_in_parallel());
  //printf("threadId: %d\n", omp_get_thread_num());
  //printf("nN: %d, nS: %d, k: %d\n", nN, nS, k);

  //---------------------------------------------------------------------------
  size[0] = nN * sizeof(char);// dSeq and Seq size
  size[1] = nN * sizeof(lint); // dKmer and kmer size
  size[2] = nS * sizeof(int);  // dLength
  size[3] = nN * sizeof(Freq);// Freq
  size[4] = nS * sizeof(lint); // dStart
  totalsize = size[0] + size[1] + (size[2] * 2) + size[3];

  if (totalsize > deviceMemory)
  {
    printf("\n\n\t\t\t[Error] There is no enough space on GPU memory\n");
    printf("\t\t\t[Error] Required memory: %ld; Available memory: %ld\n", totalsize, deviceMemory);
    exit(1);
  }
  //---------------------------------------------------------------------------

  if ( cudaMalloc (&dSeq, size[0])    != cudaSuccess ) printf("\n[Error 1] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMalloc (&dKmer, size[1])  != cudaSuccess ) printf("\n[Error 2] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMalloc (&dStart, size[4])  != cudaSuccess ) printf("\n[Error 3] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMalloc (&dLength, size[2]) != cudaSuccess ) printf("\n[Error 4] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMalloc (&dFreq, size[3])   != cudaSuccess ) printf("\n[Error 5] %s\n", cudaGetErrorString(cudaGetLastError()));

  //************************************************
  block[0] = maxThreadDim;
  grid[0] = floor(nN / block[0]) + 1;
  if (grid[0] > maxGridSize)
  {
    grid[0] = maxGridSize;
    offset[0] = (nN / (grid[0] * block[0])) + 1;
  }

  block[1] = maxThreadDim;
  grid[1] = (nS / block[1]) + 1;
  if (grid[1] > maxGridSize)
  {
    grid[1] = maxGridSize;
    offset[1] = (nS / (grid[1] * block[1])) + 1;
  }

  block[2] = maxThreadDim;
  grid[2] = nS;
  if (nS > maxGridSize)
  {
    grid[2] = maxGridSize;
    offset[2] = (nS / grid[2]) + 1;
  }

  int nF = nN;// - (nS * (k - 1));
  //printf("\nnS: %ld, nN: %ld, nF: %d\n", nS, nN, nF);
  block[3] = maxThreadDim;
  grid[3] = ceil(nF/1024)+1;
  if (grid[3] > maxGridSize)
  {
    grid[3] = maxGridSize;
    offset[3] = (nF / (grid[3] * block[3])) + 1;
  }

  //************************************************

  if ( cudaMemcpyAsync(dSeq, rd->data, size[0], cudaMemcpyHostToDevice, stream) != cudaSuccess) printf("[Error 6] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMemcpyAsync(dStart, rd->start, size[4], cudaMemcpyHostToDevice, stream) != cudaSuccess) printf("[Error 7] %s\n", cudaGetErrorString(cudaGetLastError()));
  if ( cudaMemcpyAsync(dLength, rd->length, size[2], cudaMemcpyHostToDevice, stream) != cudaSuccess) printf("[Error 8] %s\n", cudaGetErrorString(cudaGetLastError()));

  //************************************************
  ResetKmer<<<grid[0], block[0], 0, stream>>>(dKmer, offset[0], -1, nN);
  ResetFreq<<<grid[3], block[3], 0, stream>>>(dFreq, offset[3], -1, nF);
  ComputeKmer<<<grid[0], block[0], 0, stream>>>(dSeq, dKmer, k, nN, offset[0]);
  ComputeFreq<<<grid[1], block[1], 0, stream>>>(dKmer, dFreq, dStart, dLength, nS, k, nF);
  //ComputeFreqNew<<<grid[2],block[2]>>>(d_Index, d_Freq, d_start, d_length, offset[2], fourk, nS);

  //cudaFree(rd);

//puts("Foi");
//  if ( cudaMallocHost(&rd->freq, size[3]) != cudaSuccess) printf("\n[Error 9] %s\n", cudaGetErrorString(cudaGetLastError()));
//puts("Voltou");
  if ( cudaMemcpyAsync(rd->freq, dFreq, size[3], cudaMemcpyDeviceToHost, stream) != cudaSuccess) printf("\n[Error 10] %s\n", cudaGetErrorString(cudaGetLastError()));
  //cudaStreamSynchronize(stream);
  //for (int i = 0 ; i < nF; i++)
  //  printf("%d:%d, ", rd->freq[i].kmer, rd->freq[i].count);

  //************************************************
  cudaFree(dSeq);
  cudaFree(dFreq);
  cudaFree(dKmer);
  cudaFree(dStart);
  cudaFree(dLength);
  //---------------------------------------------------------------------------

  //printf("\nFim kmer_main\n");
}
