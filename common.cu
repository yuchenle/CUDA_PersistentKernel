#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <stdbool.h>

#include "common.h"
#include "cudaErr.h"
#include "const.h"

void init_com (int **com)
{
  printf ("init communicator\n");
  int *com_t;
  gpuErrchk (cudaMallocHost (&com_t, sizeof(int) * (NB_BLOCKS + 1)));  // last byte to notify PK to exit

  for (int i = 0; i < NB_BLOCKS; i++)
    com_t[i] = 0;

  *com = com_t;
}

void startGPU (int *com)
{
  // printf ("startGPU\n");
  // memset (com, 1, NB_BLOCKS);
  for (int i=0; i < NB_BLOCKS;i++)
    com[i] = 1;
}

void waitGPU (int *com)
{
  // printf ("waitGPU\n");
  int sum;
  do
  {
    sum = 0;
    asm volatile ("" ::: "memory");
    for (int i = 0; i < NB_BLOCKS; i++)
      sum |= com[i];
  }while (sum != 0);
}

void endGPU (int *com)
{
  printf ("cpu is ending GPU\n");
  com [NB_BLOCKS] = 1;
}
