#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 1000
#define BASE_TYPE int

__device__ bool IsApropriate(double* x, double* y)
{
    bool result = (*x) * (*x) + (*y) * (*y) <= 1;
    return result;
}

__global__ void CalculatePI(BASE_TYPE *d_a)
{
    double x = (double)blockIdx.x / N;
    double y = (double)threadIdx.x / N;
    IsApropriate(&x, &y) ? atomicAdd(d_a, 1) : 0;
}

int main()
{
    BASE_TYPE size = N * N;
    BASE_TYPE a = 0;
    BASE_TYPE *d_a;

    cudaMalloc((void**)&d_a, sizeof(BASE_TYPE));
    cudaMemcpy(&a, d_a, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

    CalculatePI <<< N, N >>> (d_a);

    cudaMemcpy(&a, d_a, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
    printf("pi = %f\n",  (double)a * 4 / size);

    cudaFree(d_a);
    return 0;
}

