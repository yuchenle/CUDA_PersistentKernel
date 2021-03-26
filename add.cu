#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

#include "cudaErr.h"
#include "const.h"
#include "persistent_add.cuh"
#include "common.h"

// global variables to be initialized
int shmid;
TYPE *ptr;

void init ()
{
  // FILE to key
  key_t key = ftok (FILENAME, FILEID);
  if (key == -1) 
  {
    printf ("ftok failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // getting SHM id
  printf (" size of shm is %zu\n", sizeof(TYPE));
  shmid = shmget (key, sizeof (TYPE), 0666|IPC_CREAT);
  if (shmid == -1) 
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0);
}

void notifyNext ()
{
  // printf ("notifyNext by add\n");
  ptr->ready[SIZE-1] = 0;
}

void waitPrevStage ()
{
  while (ptr->ready[SIZE-1] != 1)
  {
    asm volatile ("":::"memory");
  }
}

// void init_com (int **com)
// {
//   printf ("init communicator\n");
// 
//   int *com_t;
//   gpuErrchk (cudaMallocHost (&com_t, sizeof(int) * (NB_BLOCKS + 1))); //last element to stop GPU
// 
//   for (int i = 0; i < NB_BLOCKS; i++)
//     com_t[i] = 0;
// 
//   *com = com_t;
// }

// void startGPU (int *com)
// {
//   // printf ("startGPU\n");
//   for (int i=0; i < NB_BLOCKS;i++)
//     com[i] = 1;
// }

// void waitGPU (int *com)
// {
//   // printf ("waitGPU\n");
//   int sum;
//   do  
//   {
//     sum = 0;
//     asm volatile ("" ::: "memory");
//     for (int i = 0; i < NB_BLOCKS; i++)
//       sum |= com[i];
//   }while (sum != 0);
// }
 
// void endGPU (int *com)
// {
//   printf ("cpu is ending GPU\n");
//   com [NB_BLOCKS] = 1;
// }

int main()
{
  // allocating shared memory, inter-process communication
  init ();

  // establish intra-process GPU & CPU communication method
  int *com;
  init_com (&com);

  // setting GPU data pointer
  gpuErrchk (cudaHostRegister ((void *)ptr->data, sizeof (GPU_DATA_TYPE) * GPU_SIZE, cudaHostRegisterMapped|cudaHostRegisterPortable));
  gpuErrchk (cudaHostGetDevicePointer (&(ptr->d_data), (void *)ptr->data, 0));

  // launching PK
  persistent_add<<<NB_BLOCKS, NB_TH>>> (ptr->d_data, 1, com);

  for (int i=0; i<NUM_ITER; i++)
  {
    // printf ("add begins one iter, data[GPU_SIZE-1] = %.2f\n", ptr->data[GPU_SIZE-1]);

    waitPrevStage ();

    startGPU (com);

    /* generate/receive new data for next iteration? */

    waitGPU (com);
    /* use of the result, print it for instance */
    notifyNext();
    // printf ("add ends one iter, data[GPU_SIZE-1] = %.2f\n", ptr->data[GPU_SIZE-1]);
  }
  endGPU (com);
  gpuErrchk (cudaHostUnregister (ptr->data));
  gpuErrchk (cudaFreeHost (com));
  return 0;
}
