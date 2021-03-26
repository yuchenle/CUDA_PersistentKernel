  #include "persistent_add.cuh"
  #include <stdio.h>

  __device__ void waitCPU (int *com)
  {
    int block_id = blockIdx.x;
    // printf ("waitCPU by PK_add\n");

    while (com[block_id] != 1 && com [NB_BLOCKS] != 1)
    {
      __threadfence();
      // printf ("waitMult by PK_add, data[0] = %d, block %d\n", data[0], block_id);
    }
  }

  __device__ void work_complete (int *com)
  {
    int block_id = blockIdx.x;
    //  printf ("work_complete by PK_add\n");

    com[block_id] = 0;
  }

  __global__ void persistent_add (GPU_DATA_TYPE *data, GPU_DATA_TYPE value, int *com)
  {
    int local_id = threadIdx.x;
    while (com[NB_BLOCKS] != 1)
    {
      if (local_id == 0)
        waitCPU (com);

      __syncthreads();

      // cancelling point
      if (com[NB_BLOCKS]==1)
        return;

      int global_id = blockDim.x * blockIdx.x + threadIdx.x;

      // for now, just vecAdd
      for (; global_id < GPU_SIZE && global_id > 0; global_id += blockDim.x * gridDim.x)
        data[global_id] += value;
      
      if (local_id == 0)
        work_complete (com);
    }
  }
