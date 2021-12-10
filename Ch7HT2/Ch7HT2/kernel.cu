#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 64 //количество элементов массива
#define BASE_TYPE int
#define BLOCK_SIZE 32

// Ядро
__global__ void SelfProd(const BASE_TYPE* A, BASE_TYPE* C)
{
	// Переменная для хранения суммы элементов
	BASE_TYPE sum = 0.0;
	// Создание массивов в разделяемой памяти
	__shared__ BASE_TYPE ash[BLOCK_SIZE];
	// Копирование из глобальной памяти
	ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x];
	// Синхронизация нитей
	__syncthreads();
	// Вычисление скалярного произведения
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		for (int j = 0; j < BLOCK_SIZE; j++)
			sum += ash[j] * ash[j];
		C[blockIdx.x] = sum;
	}
}

int main()
{
	// выделение памяти под массивы на CPU
	BASE_TYPE a[N], c[N];
	// выделение памяти под массивы для копирования
	// на GPU
	BASE_TYPE* dev_a, * dev_c;

	// заполнение массивов
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
	}

	// выделение памяти под массивы на GPU
	cudaMalloc((void**)&dev_a, N * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_c, N * sizeof(BASE_TYPE));

	// копирование данных в память GPU
	cudaMemcpy(dev_a, a, N * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

	int GRID_SIZE = N / BLOCK_SIZE + 1;
	SelfProd << <GRID_SIZE, BLOCK_SIZE >> > (dev_a, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	//вывод результата
	int prod = 0;
	for (int i = 0; i < GRID_SIZE; i++)
		prod += c[i];
	printf("Euclid distance = %f", sqrt(prod));

	cudaFree(dev_a);
	cudaFree(dev_c);
	return 0;
}