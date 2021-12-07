#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 10 //количество элементов массива

// Ядро
__global__ void ProdD(double* a, double* b, double* c)
{
	int i = threadIdx.x; //индексирование
	if (i > N - 1) return; 	//проверка на выход за пределы массива
	//поэлементное умножение массивов
	c[i] = __dmul_rn(a[i], b[i]);
}

__global__ void ProdF(float* a, float* b, float* c)
{
	int i = threadIdx.x; //индексирование
	if (i > N - 1) return; 	//проверка на выход за пределы массива
	//поэлементное умножение массивов
	c[i] = __fmul_rn(a[i], b[i]);
}

int main()
{
	//переменные для замера времени работы
	cudaEvent_t start_f, stop_f, start_d, stop_d;
	cudaEventCreate(&start_f);
	cudaEventCreate(&stop_f);
	cudaEventCreate(&start_d);
	cudaEventCreate(&stop_d);

	// выделение памяти под массивы на CPU
	float a_f[N], b_f[N], c_f[N];
	double a_d[N], b_d[N], c_d[N];
	// выделение памяти под массивы для копирования
	// на GPU
	float *dev_a_f, *dev_b_f, *dev_c_f;
	double *dev_a_d, *dev_b_d, *dev_c_d;

	// заполнение массивов
	for (int i = 0; i < N; i++)
	{
		a_f[i] = a_d[i] = i + 1;
		b_f[i] = b_d[i] = -2;
	}

	// выделение памяти под массивы на GPU
	cudaMalloc((void**)&dev_a_f, N * sizeof(float));
	cudaMalloc((void**)&dev_b_f, N * sizeof(float));
	cudaMalloc((void**)&dev_c_f, N * sizeof(float));
	cudaMalloc((void**)&dev_a_d, N * sizeof(double));
	cudaMalloc((void**)&dev_b_d, N * sizeof(double));
	cudaMalloc((void**)&dev_c_d, N * sizeof(double));

	// копирование данных в память GPU
	cudaMemcpy(dev_a_f, a_f, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b_f, b_f, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a_d, a_d, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b_d, b_d, N * sizeof(double), cudaMemcpyHostToDevice);

	cudaEventRecord(start_f, 0); //начало замера ядра float
	ProdF <<<1, N >>> (dev_a_f, dev_b_f, dev_c_f);
	cudaEventRecord(stop_f, 0); //конец замера
	cudaEventSynchronize(stop_f);

	float kernelTime_f;
	cudaEventElapsedTime(&kernelTime_f, start_f, stop_f);
	printf("Float kernel time = %f ms\n", kernelTime_f);

	cudaEventRecord(start_d, 0); //начало замера ядра double
	ProdD <<<1, N >>> (dev_a_d, dev_b_d, dev_c_d);
	cudaEventRecord(stop_d, 0); //конец замера
	cudaEventSynchronize(stop_d);

	float kernelTime_d;
	cudaEventElapsedTime(&kernelTime_d, start_d, stop_d);
	printf("Double kernel time = %f ms\n", kernelTime_d);

	cudaMemcpy(c_f, dev_c_f, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_d, dev_c_d, N * sizeof(double), cudaMemcpyDeviceToHost);

	//переменные для результата
	float prod_f = 0;
	double prod_d = 0;

	//подсчёт скалярного произведения
	for (int i = 0; i < N; i++)
	{
		prod_f += c_f[i];
		prod_d += c_d[i];
	}

	//вывод результата
	printf("prod_f = %f\nprod_d = %f\n", prod_f, prod_d);

	cudaFree(dev_a_f);
	cudaFree(dev_b_f);
	cudaFree(dev_c_f);
	cudaFree(dev_a_d);
	cudaFree(dev_b_d);
	cudaFree(dev_c_d);
	return 0;
}