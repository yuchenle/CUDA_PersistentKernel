/*
 * CUDA error checking macro from @talonmies,
 * https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#include <stdio.h>
#include <stdlib.h>
#ifndef cudaErrChecking
#define cudaErrChecking

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: code(%d): %s, at %s: %d\n", (unsigned) code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif
