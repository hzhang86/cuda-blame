/*
 * Copyright 2010-2017 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler event values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include "libunwind.h"
#include <sys/time.h>

#include "inst_sampling.h"
#define COMPUTE_N 50000
// Define some global variables

// Gpu wait
__device__ void gpu_sleep(clock_t cycles)
{ 
  clock_t start = clock64();
  clock_t elapsed = 0;
  while (elapsed < cycles) {
    elapsed = clock64() - start;
  }
}

// Kernels
__global__ void 
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
  gpu_sleep(N);
}

__global__ void 
VecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
  //gpu_sleep(N);
}

static void
initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

static void
do_pass(cudaStream_t stream)
{
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;
  
  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  
  // Initialize input vectors
  initVec(h_A, COMPUTE_N);
  initVec(h_B, COMPUTE_N);
  memset(h_C, 0, size);

  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

  RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  //VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  VecSub<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
    
  RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

  if (stream == 0)
    RUNTIME_API_CALL(cudaDeviceSynchronize());
  else
    RUNTIME_API_CALL(cudaStreamSynchronize(stream));

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


int main(int argc, char *argv[])
{
  timeval start;
  gettimeofday(&start, NULL) ;
  //initialize sampling
  initTrace();

  do_pass(0);
  //finish sampling
  finiTrace();
  
  timeval end;
  gettimeofday(&end, NULL) ;
  double elapsed_time = (double)(end.tv_sec - start.tv_sec) + 
                        ((double)(end.tv_usec - start.tv_usec))/1000000 ;
  printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
  return 0;
}

