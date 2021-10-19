#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>
#include <vector>
#include<iostream>



__global__ void CalculatePi(int *niter, int *count, double *x, double *y)
{
    printf("niter = %d\n", *niter);
    for (int i = 0; i < *niter; ++i)
    {
        double vectorLength = (x[i] * x[i]) + (y[i] * y[i]);

        if (vectorLength <= 1)
            ++ *count;
    }
    printf("count = %d\n", *count);
}

int main()
{
    const int niter = 50000;
    int i;
    int count = 0;
    double pi;
    int *dev_niter, *dev_count;
    int size = sizeof(int);
    double x[niter];
    double *dev_x;
    double y[niter];
    double *dev_y;
    
    const size_t x_size = sizeof(double) * size_t(niter);

    srand(time(NULL));

    for (int i = 0; i < niter; i++)
    {
        x[i] = (double)rand() / RAND_MAX;
        y[i] = (double)rand() / RAND_MAX;

    }

    cudaMalloc((void**)&dev_niter, size);
    cudaMalloc((void**)&dev_count, size);

    cudaMalloc((void**)&dev_x, x_size);
    cudaMalloc((void**)&dev_y, x_size);

    cudaMemcpy(dev_niter, &niter, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, &count, size, cudaMemcpyHostToDevice);

    cudaMemcpy(dev_x, x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, x_size, cudaMemcpyHostToDevice);

    CalculatePi<<< 1, 1 >>> (dev_niter, dev_count, dev_x, dev_y);

    cudaMemcpy(&count, dev_count, size, cudaMemcpyDeviceToHost);

    pi = ((double)count / (double)niter) * 4.0;
    std::cout << "Pi number: " << pi << std::endl;

    return 0;
}