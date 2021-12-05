#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 100 // количество членов ряда

__global__ void ZFunction(float *a, float *b)
{
    int i = threadIdx.x; //индексация
    a[i] = 1.f / powf(float(i + 1), *b); //вычисление i-ого члена
}


int main()
{
    float s = 2; //степень
    float a[N]; //массив членов ряда
    float *d_s = 0;
    float *d_a = 0;
    float sum = 0; //частная сумма ряда

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_s, sizeof(float));
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, &s, sizeof(float), cudaMemcpyHostToDevice);

    ZFunction <<<1, N >>>(d_a, d_s);

    cudaMemcpy(a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) 
    {
        sum += a[i];
    }
    printf("%f\n", sum);

    cudaFree(d_a);
    cudaFree(d_s);
    return 0;
}