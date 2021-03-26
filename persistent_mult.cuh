#ifndef P_MULT_CUH
#define P_MULT_CUH

extern "C"
{
  #include "const.h" 
  __global__ void persistent_mult (GPU_DATA_TYPE *, GPU_DATA_TYPE, int *);
}

#endif // P_MULT_CUH

