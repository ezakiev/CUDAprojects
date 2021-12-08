#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16 
#define BASE_TYPE int // тип, который будут иметь элементы матриц 

// функция перемножения матриц
__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C0, BASE_TYPE *C1, int Acols, int Bcols)
{
	int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
	int j0 = blockDim.x * blockIdx.x + threadIdx.x;
	int i1 = Bcols * (blockDim.y * blockIdx.y + threadIdx.y);
	int j1 = j0;
	BASE_TYPE sum0 = 0;
	BASE_TYPE sum1 = 0;
	for (int k = 0; k < Acols; k++)
		sum0 += A[i0 + k] * B[k * Bcols + j0];
	for (int k = 0; k < Bcols; k++)
		sum1 += B[i1 + k] * A[k * Acols + j1];
	int ind0 = Bcols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	int ind1 = Acols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C0[ind0] = sum0;
	C1[ind1] = sum1;
}

int toMultiple(int a, int b) {
	int mod = a % b;
	if (mod != 0) {
		mod = b - mod;
		return a + mod;
	}
	return a;
}

int main()
{
	// количество строк и столбцов матрицы
	int Arows = 5;
	int Acols = 6;
	int Brows = Acols;
	int Bcols = 17;

	Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);

	Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);
	Brows = toMultiple(Brows, BLOCK_SIZE);
	printf("Brows = %d\n", Brows);

	Bcols = toMultiple(Bcols, BLOCK_SIZE);
	printf("Bcols = %d\n", Bcols);

	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
	size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

	BASE_TYPE* h_A = (BASE_TYPE*)malloc(Asize);
	BASE_TYPE* h_B = (BASE_TYPE*)malloc(Bsize);
	BASE_TYPE* h_C0 = (BASE_TYPE*)malloc(Csize);
	BASE_TYPE* h_C1 = (BASE_TYPE*)malloc(Csize);

	srand(time(0));
	// рассмотрен случай двух диагональных матриц (такие матрицы всегда коммутирующие)
	for (int i = 0; i < Arows; i++) 
	{
		for (int j = 0; j < Acols; j++)
		{
			if (i == j)
				h_A[i * Acols + j] = rand() % 10 + 1;
			else
				h_A[i * Acols + j] = 0;
			printf("%d ", h_A[i * Acols + j]);
		}
		printf("\n");
	}
	printf("\n");

	// для случайного эксперимента закомментировать этот цикл
	for (int i = 0; i < Brows; i++)
	{
		for (int j = 0; j < Bcols; j++)
		{
			if (i == j)
				h_B[i * Bcols + j] = rand() % 10 + 1;
			else
				h_B[i * Bcols + j] = 0;
			printf("%d ", h_B[i * Bcols + j]);
		}
		printf("\n");
	}

	//для случайного эксперимента раскомментировать эту часть
	/*for (int i = 0; i < Brows * Bcols; ++i) {
		h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
	}*/
	BASE_TYPE* d_A = NULL;
	cudaMalloc((void**)&d_A, Asize);

	BASE_TYPE* d_B = NULL;
	cudaMalloc((void**)&d_B, Bsize);

	BASE_TYPE* d_C0 = NULL;
	cudaMalloc((void**)&d_C0, Csize);

	BASE_TYPE* d_C1 = NULL;
	cudaMalloc((void**)&d_C1, Csize);

	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);

	matrixMult <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C0, d_C1, Acols, Bcols);

	cudaMemcpy(h_C0, d_C0, Csize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C1, d_C1, Csize, cudaMemcpyDeviceToHost);

	if (*h_C0 == *h_C1)
		printf("Success!");
	else
		printf("FAIL!");


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C0);
	cudaFree(d_C1);
	free(h_A);
	free(h_B);
	free(h_C0);
	free(h_C1);
	return 0;
}