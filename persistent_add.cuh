#ifndef P_ADD_CUH
#define P_ADD_CUH

extern "C"
{
  #include "const.h" 
  __global__ void persistent_add (GPU_DATA_TYPE *, GPU_DATA_TYPE, int *);
}

#endif // P_ADD_CUH

