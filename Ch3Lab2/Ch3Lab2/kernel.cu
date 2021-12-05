#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define n 1000 // количество элементов в массиве

__global__ void CalculatePI(double* a)
{
    int i = threadIdx.x; // индексация
    a[i] = std::sqrtf(1.0 - double(i * i) / double(n * n)); // вычисление значения подинтегральной функции
}

int main()
{
    double a[n]; // массив для значений функции
    double* d_a;

    cudaMalloc((void**)&d_a, n * sizeof(double)); // выделяем память на GPU
    CalculatePI <<<1, n >>>(d_a);

    // получение сведений о последней ошибке
    cudaError_t err = cudaGetLastError();

    cudaMemcpy(a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

    // проверка на ошибки
    if (err != cudaSuccess)
        printf("%s ", cudaGetErrorString(err)); //вывод ошибки
    else
    {
        // подсчёт и вывод интегральной суммы, если не было ошибки
        double q = 0;
        for (int i = 0; i < n; ++i) {
            q += a[i];
        }
        printf("pi = %f\n", q * 4 / n);
    }

    cudaFree(d_a);
    return 0;
}