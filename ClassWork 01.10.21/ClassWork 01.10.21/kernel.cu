#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 1000
#define BASE_TYPE short

__device__ bool IsApropriate(double *x, double *y)
{
    bool result = (*x) * (*x) + (*y) * (*y) <= 1;
    return result;
}

__global__ void CalculatePI(BASE_TYPE *d_a) 
{
    double x = (double)blockIdx.x / N;
    double y = (double)threadIdx.x / N;
    d_a[blockIdx.x * N + threadIdx.x] = IsApropriate(&x, &y);
}

int main()
{
    int size = N * N;
    BASE_TYPE *a = new BASE_TYPE[size];
    BASE_TYPE *d_a;
    int counter = 0;

    cudaMalloc((void**)&d_a, size * sizeof(BASE_TYPE));

    CalculatePI <<< N, N >>>(d_a);
    
    cudaMemcpy(a, d_a, size * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) 
    {
        counter += a[i];
    }
    printf("pi = %f\n", (double)counter * 4 / size);

    cudaFree(d_a);
    return 0;
}