#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 100 //количество элементов массива
#define BASE_TYPE float
#define BLOCK_SIZE 32

// Ядро
__global__ void Prod(const BASE_TYPE* xs, BASE_TYPE* h, BASE_TYPE* F)
{
	// Создание массивов в разделяемой памяти
	__shared__ BASE_TYPE fsh[BLOCK_SIZE];
	// Копирование из глобальной памяти
	fsh[threadIdx.x] = *h * __expf(xs[blockIdx.x * blockDim.x + threadIdx.x]);
	// Синхронизация нитей
	__syncthreads();
	// Вычисление скалярного произведения
	if (blockIdx.x * blockDim.x + threadIdx.x < N)
	{
		F[blockIdx.x * blockDim.x + threadIdx.x] = fsh[threadIdx.x];
		printf("thread %d with %f\n", threadIdx.x, F[blockIdx.x * blockDim.x + threadIdx.x]);
	}

}

int main()
{
	// выделение памяти на CPU
	BASE_TYPE xs[N], F[N], h;
	// выделение памяти для копирования на GPU
	BASE_TYPE* dev_xs, * dev_F, *dev_h;
	
	BASE_TYPE a = 0; // нижний предел интегрирования
	BASE_TYPE b = 1; // верхний предел интегрирования
	h = (b - a) / N; //шаг

	// заполнение вектора узлов сетки
	for (int i = 0; i < N - 1; i++)
	{
		xs[i] = a + (2 * i + 1) * h / 2;
		//printf("%f\n", xs[i]);
	}

	// выделение памяти
	cudaMalloc((void**)&dev_xs, N * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_F, N * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_h, sizeof(BASE_TYPE));

	// копирование данных в память GPU
	cudaMemcpy(dev_xs, xs, N * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, &h, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

	int GRID_SIZE = N / BLOCK_SIZE + 1;
	Prod <<<GRID_SIZE, BLOCK_SIZE >>> (dev_xs, dev_h, dev_F);

	cudaMemcpy(F, dev_F, N * sizeof(float), cudaMemcpyDeviceToHost);

	//вывод результата
	float sum = 0;
	for (int i = 0; i < N; i++)
		sum += F[i];
	printf("sum = %f", sum);

	cudaFree(dev_xs);
	cudaFree(dev_h);
	cudaFree(dev_F);
	return 0;
}