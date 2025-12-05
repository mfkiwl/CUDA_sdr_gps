// src/cuda_lib.cu
#include "cuda_lib.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// __global__ void addKernel(const int* a, const int* b, int* c, unsigned int size) {
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         c[i] = a[i] + b[i];
//     }
// }

// CUDA Kernel function that runs on the GPU
// __global__ specifies that this function is a kernel and can be called from the CPU
__global__ void add_vectors(int *a, int *b, int *c, int size) {
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we are within the bounds of the array
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

void add_gpu(const int* a, const int* b, int* c, unsigned int size) {
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    printf("Launching kernel with %u blocks of 256 threads\n", (size + 255) / 256);

    // addKernel<<size/256 + 1, 256>>(d_a, d_b, d_c, size);
    add_vectors<<<(size / 256) + 1, 256>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}