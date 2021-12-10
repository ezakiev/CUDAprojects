#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 100 //количество разбиений промежутка интегрирования
#define BASE_TYPE float
#define BLOCK_SIZE 32

// расчет по формуле прямоугольников
__global__ void ExpIntegralByRectangle(const BASE_TYPE* xs, BASE_TYPE* h, BASE_TYPE* F)
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
	}

}

//функция для расчета по формулам трапеции и Симпсона
__global__ void ExpIntegralByTrapezoidAndSimpson(const BASE_TYPE* x, BASE_TYPE* h, BASE_TYPE* F)
{
	// Создание массивов в разделяемой памяти
	__shared__ BASE_TYPE fsh[BLOCK_SIZE];

	//индексация
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i - 100;

	//проверка на выход за массив
	if (i >= 2 * N) return;
	// Копирование из глобальной памяти
	if (i < N)
		fsh[threadIdx.x] = *h * (__expf(x[i + 1]) + __expf(x[i])) / 2;
	else
	{
		fsh[threadIdx.x] = *h / 6 * (__expf(x[j]) + 4 * __expf((x[j] + x[j + 1]) / 2) + __expf(x[j + 1]));
	}

	// Синхронизация нитей
	__syncthreads();

	F[i] = fsh[threadIdx.x];
}

int main()
{
	// выделение памяти на CPU
	BASE_TYPE xR[N], xTS[N + 1], FR[N], FTS[N * 2], h;
	// выделение памяти для копирования на GPU
	BASE_TYPE* dev_xR, * dev_xTS, * dev_FR, * dev_FTS, * dev_h;

	BASE_TYPE a = 0; // нижний предел интегрирования
	BASE_TYPE b = 1; // верхний предел интегрирования
	h = (b - a) / N; //шаг

	// заполнение вектора узлов сетки
	for (int i = 0; i < N + 1; i++)
	{
		if (i < N)
			xR[i] = a + (2 * i + 1) * h / 2;
		xTS[i] = a + i * h;
	}

	// выделение памяти
	cudaMalloc((void**)&dev_xR, N * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_FR, N * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_xTS, (N + 1) * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_FTS, N * 2 * sizeof(BASE_TYPE));
	cudaMalloc((void**)&dev_h, sizeof(BASE_TYPE));

	// копирование данных в память GPU
	cudaMemcpy(dev_xR, xR, N * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xTS, xTS, (N + 1) * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, &h, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

	int GRID_SIZE_R = N / BLOCK_SIZE + 1;
	int GRID_SIZE_TS = N * 2 / BLOCK_SIZE + 1;
	ExpIntegralByRectangle <<<GRID_SIZE_R, BLOCK_SIZE >>> (dev_xR, dev_h, dev_FR);

	cudaMemcpy(FR, dev_FR, N * sizeof(float), cudaMemcpyDeviceToHost);

	ExpIntegralByTrapezoidAndSimpson <<< GRID_SIZE_TS, BLOCK_SIZE >>> (dev_xTS, dev_h, dev_FTS);

	cudaMemcpy(FTS, dev_FTS, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);

	//вывод результата
	float sumR = 0;
	float sumT = 0;
	float sumS = 0;
	for (int i = 0; i < N * 2; i++)
	{
		if (i < N)
		{
			sumR += FR[i];
			sumT += FTS[i];
		}
		else
			sumS += FTS[i];
	}

	printf("Integral of exp by rectangle = %f\n", sumR);
	printf("Integral of exp by trapezoid = %f\n", sumT);
	printf("Integral of exp by Simpson = %f\n", sumS);

	cudaFree(dev_xR);
	cudaFree(dev_FR);
	cudaFree(dev_xTS);
	cudaFree(dev_FTS);
	cudaFree(dev_h);
	return 0;
}