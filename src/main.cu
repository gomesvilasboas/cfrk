/*
CFRK-MT - Contabilizador da Frequencia de Repetica de kmer (Multi GPU version)
Developer: Fabricio Gomes Vilasboas
Istitution: National Laboratory for Scientific Computing
*/

#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include "kmer.cuh"
#include "tipos.h"

void PrintFreqRemain(Read *chunk, int nChunk, int chunkSize, int k, char *fileOut)
{
  FILE *output;
  //char str[256];

  output = fopen(fileOut, "a");
  for (int i = 0; i < chunkSize; i++)
  {
    int end = chunk->start[i] + chunk->length[i];
    for (int j = chunk->start[i]; j < end; j++)
    {
      if (chunk->freq[j].kmer != -1)
      fprintf(output, "%ld:%d ", chunk->freq[j].kmer, chunk->freq[j].count);
    }
    fprintf(output, "\n");
  }
  fclose(output);
}

void PrintFreq(Read *chunk, int nChunk, int chunkSize, int k, char *fileOut)
{
  FILE *output;
  //char str[256];

  output = fopen(fileOut, "w");
  for (int i = 0; i < nChunk; i++)
  {
    for (int j = 0; j < chunkSize; j++)
    {
      int end = chunk[i].start[j] + chunk[i].length[j];
      for (int k = chunk[i].start[j]; k < end; k++)
      {
        if (chunk[i].freq[k].kmer != -1)
          fprintf(output, "%ld:%d ", chunk[i].freq[k].kmer, chunk[i].freq[k].count);
      }
      fprintf(output, "\n");
    }
  }
  fclose(output);
}

void DeviceInfo(uint8_t device)
{
  cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, device);

  printf("\n\n***** Device information *****\n\n");

  printf("\tId: %d\n", device);
  printf("\tName: %s\n", prop.name);
  printf("\tTotal global memory: %ld\n", prop.totalGlobalMem);
  printf("\tMax grid size: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("\tMax thread dim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("\tWarp size: %d\n", prop.warpSize);
  printf("\tMax threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);

  printf("\n************************************\n\n");
}

int SelectDevice(int devCount)
{

  int i, device = 0;
  cudaDeviceProp prop[devCount];

  if (devCount > 0)
  {
    for (i = 0; i < devCount; i++)
    {
      cudaGetDeviceProperties(&prop[i], i);
    }

    for (i = 0; i < devCount; i++)
    {
      if (prop[i].totalGlobalMem > prop[device].totalGlobalMem)
      {
        device = i;
      }
    }
  }
  else
  return 0;

  return device;
}

struct read* SelectChunkRemain(struct read *rd, ushort chunkSize, ushort it, lint max, lint gnS, lint *nS, lint gnN, lint *nN, int nt, int k)
{
  struct read *chunk;
  lint i;
  lint j;
  lint length = 0;

  // Size to be allocated
  for (i = 0; i < max; i++)
  {
    lint id = chunkSize*it + i;
    if (id > gnS-1)
    {
      break;
    }
    length += rd->length[id]+1;
  }

  cudaMallocHost((void**)&chunk, sizeof(struct read));
  cudaMallocHost((void**)&chunk->data, sizeof(char)*length);
  cudaMallocHost((void**)&chunk->length, sizeof(int)*chunkSize);
  cudaMallocHost((void**)&chunk->start, sizeof(lint)*chunkSize);

  // Copy rd->data to chunk->data
  lint start = rd->start[chunkSize*it];
  lint end = start + (lint)length;

  #pragma omp parallel for num_threads(nt)
  for (j = start; j < end; j++)
  {
    chunk->data[j-start] = rd->data[j];
  }

  chunk->length[0] = rd->length[chunkSize*it];
  chunk->start[0] = 0;

  // Copy start and length
  for (i = 1; i < max; i++)
  {
    lint id = chunkSize*it + i;
    chunk->length[i] = rd->length[id];
    chunk->start[i] = chunk->start[i-1]+(chunk->length[i-1]+1);
  }

  cudaMallocHost(&chunk->freq, length * sizeof(Freq));

  *nN = length;
  *nS = max;
  return chunk;
}


void SelectChunk(struct read *chunk, const int nChunk, struct read *rd, ushort chunkSize, lint max, lint gnS, lint *nS, lint gnN, lint *nN, int nt, int k)
{
  lint i, j, it;

  for (it = 0; it < nChunk; it++)
  {
    lint length = 0;

    // Size to be allocated
    for (i = 0; i < max; i++)
    {
      lint id = chunkSize*it + i;
      if (id > gnS-1)
      {
        break;
      }
      length += rd->length[id]+1;
    }

    cudaMallocHost(&chunk[it].data, sizeof(char)*length);
    cudaMallocHost(&chunk[it].length, sizeof(int)*chunkSize);
    cudaMallocHost(&chunk[it].start, sizeof(lint)*chunkSize);

    // Copy rd->data to chunk->data
    lint start = rd->start[chunkSize*it];
    lint end = start + (lint)length;
    #pragma omp parallel for num_threads(nt)
    for (j = start; j < end; j++)
    {
      chunk[it].data[j-start] = rd->data[j];
    }

    chunk[it].length[0] = rd->length[chunkSize*it];
    chunk[it].start[0] = 0;

    // Copy start and length
    for (i = 1; i < max; i++)
    {
      lint id = chunkSize*it + i;
      chunk[it].length[i] = rd->length[id];
      chunk[it].start[i] = chunk[it].start[i-1]+(chunk[it].length[i-1]+1);
    }

    cudaMallocHost(&chunk[it].freq, length * sizeof(Freq));

    nN[it] = length;
    nS[it] = max;
  }
}

int main(int argc, char* argv[])
{
  struct read *chunk;
  lint *nS, *nN;
  int device;
  int k;
  char fileOut[512];
  lint gnN, gnS, chunkSize = 8192;
  int devCount;
  int nt = 12;

  if ( argc < 4)
  {
    printf("Usage: ./cfrk [dataset.fasta] [file_out.cfrk] [k] <number of threads: Default 12> <chunkSize: Default 8192>");
    return 1;
  }
  cudaDeviceReset();

  k = atoi(argv[3]);
  if (argc >= 5) nt = atoi(argv[4]);
  if (argc >= 6) chunkSize = atoi(argv[5]);

  //printf("nt: %d, chunkSize: %d\n", nt, chunkSize);

  cudaGetDeviceCount(&devCount);
  //DeviceInfo(device);

  strcpy(fileOut, argv[2]);

  //printf("\ndataset: %s, out: %s, k: %d, chunkSize: %d\n", argv[1], file_out, k, chunkSize);

  lint st = time(NULL);
  //puts("\n\n\t\tReading seqs!!!");
  struct read *rd;
  cudaMallocHost((void**)&rd, sizeof(struct read));
  // rd = (struct read*)malloc(sizeof(struct read));
  ReadFASTASequences(argv[1], &gnN, &gnS, rd, 1);
  //printf("\nnS: %ld, nN: %ld\n", gnS, gnN);
  lint et = time(NULL);

  //printf("\n\t\tReading time: %ld\n", (et-st));

  int nChunk = floor(gnS/chunkSize);

  cudaMallocHost((void**)&chunk, sizeof(struct read)*nChunk);
  cudaMallocHost((void**)&nS, sizeof(lint)*nChunk);
  cudaMallocHost((void**)&nN, sizeof(lint)*nChunk);
  SelectChunk(chunk, nChunk, rd, chunkSize, chunkSize, gnS, nS, gnN, nN, nt, k);
  int chunkRemain = abs(gnS - (nChunk*chunkSize));
  lint rnS, rnN;
  struct read *chunk_remain = SelectChunkRemain(rd, chunkSize, nChunk, chunkRemain, gnS, &rnS, gnN, &rnN, nt, k);
  //cudaFree(rd);

  device = SelectDevice(devCount);

  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
  cudaStream_t stream[nChunk];
  for (int i = 0; i < nChunk; i++)
  {
    cudaStreamCreate(&stream[i]);
  }

  omp_set_num_threads(nt);
  #pragma omp parallel
  {
    #pragma omp for firstprivate (chunk, nN, nS, stream)
    for (int i = 0; i < nChunk; i++)
    {
      kmer_main(&chunk[i], nN[i], nS[i], k, device, stream[i]);
    }
  }

  //cudaDeviceSynchronize();

  cudaStream_t streamRemain;
  cudaStreamCreate(&streamRemain);
  //puts("Remain");
  kmer_main(chunk_remain, rnN, rnS, k, device, streamRemain);

  // st = time(NULL);
  PrintFreq(chunk, nChunk, chunkSize, k, fileOut);
  // et = time(NULL);
  //puts("\n\nPrintFreqRemain");
  PrintFreqRemain(chunk_remain, 1, chunkRemain, k, fileOut);
  //printf("\n");
  // printf("\n\t\tWriting time: %ld\n", (et-st));

  return 0;
}
