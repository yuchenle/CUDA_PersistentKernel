  #include "persistent_mult.cuh"
  #include <stdio.h>

  __device__ void waitCPU (int *com)
  {
    int block_id = blockIdx.x;
    // printf ("waitCPU by PK_mult, from block %d\n", block_id);

    while (com[block_id] == 0 && com[NB_BLOCKS] != 1)
      __threadfence();
  }

  __device__ void work_complete (int *com)
  {
    int block_id = blockIdx.x;
    // printf ("work complete from PK_mult\n");

    com [block_id] = 0;
  }

  __global__ void persistent_mult (GPU_DATA_TYPE *data, GPU_DATA_TYPE value, int *com)
  {
    int local_id = threadIdx.x;
    int iter = 0;
    while (com[NB_BLOCKS] != 1)
    {
      iter++;
      if (local_id == 0)
        waitCPU (com);

      __syncthreads();

      // cancelling point
      if (com [NB_BLOCKS] == 1)
        return;

      int global_id = blockDim.x * blockIdx.x + threadIdx.x;

      // for now, just vecDec
      for (; global_id < GPU_SIZE && global_id > 0; global_id += blockDim.x * gridDim.x)
      {
        data[global_id] -= value;
      }
     
      if (local_id == 0)
        // inform next CUDA kernel
        work_complete (com);
    }
  }
