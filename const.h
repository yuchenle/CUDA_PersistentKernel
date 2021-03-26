#ifndef CONST_H
#define CONST_H

#define GPU_SIZE 1024*24
#define GPU_DATA_TYPE float

#define NUM_ITER 1e5
#define SIZE 1  // number of communication canal needed
#define WAIT_VALUE false // false: data not ready, true: data ready to be read
#define TYPE struct SharedData
#define FILENAME "/tmp"
#define FILEID 666
#define NB_BLOCKS 24
#define NB_TH 128

struct SharedData
{
  bool ready[SIZE];
  // cudaStream_t RT_stream;
  // cudaStream_t callBackStream;
  // cudaIpcMemHandle_t memHandle;
  GPU_DATA_TYPE data[GPU_SIZE];
  GPU_DATA_TYPE *d_data;
  // cudaIpcEventHandle_t eventHandle;
};

#if 0 // Not used by now
#if defined(__CUDACC__) //NVCC
  #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) //GCC
  #define MY_ALIGN(n) __attribute__((aligned(n))
#else
  #error "Define memory alignment for your compiled"
#endif
#endif //0

#endif
