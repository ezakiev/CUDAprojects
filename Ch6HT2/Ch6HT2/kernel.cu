#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE int 
#define rows 16
#define cols 16

// Ядро
__global__ void matrixAdd(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C)
{
	int i = cols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[i] = A[i] + B[i];
}
// Функция вычисление числа, которое больше
// числа а и кратное числу b
int toMultiple(int a, int b)
{
	int mod = a % b;
	if (mod != 0)
	{
		mod = b - mod;
		return a + mod;
	}
	return a;
}

int main()
{
	size_t size = rows * cols * sizeof(BASE_TYPE);
	cudaError_t cudaStatus;

	BASE_TYPE h_A[rows][cols] = {0};
	BASE_TYPE h_B[rows][cols] = {0};
	BASE_TYPE h_C[rows][cols] = {0};

	srand(time(0));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			h_A[i][j] = rand() % 10 + 1;
			h_B[i][j] = rand() % 10 + 1;
		}
	}

	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				k == 0 ? printf("%d ", h_A[i][j]) : printf("%d ", h_B[i][j]);
			printf("\n");
		}
		printf("\n");
	}

	BASE_TYPE* d_A = NULL;
	cudaMalloc((void**)&d_A, size);

	BASE_TYPE* d_B = NULL;
	cudaMalloc((void**)&d_B, size);

	BASE_TYPE* d_C = NULL;
	cudaMalloc((void**)&d_C, size);

	cudaStatus = cudaMemcpy(d_A, &h_A, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMempcyA failed!");
		return 1;
	}
	cudaStatus = cudaMemcpy(d_B, &h_B, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMempcyB failed!");
		return 2;
	}

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE, rows / BLOCK_SIZE);

	matrixAdd <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C);

	cudaStatus = cudaMemcpy(&h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "memcpyC failed!");
		return 4;
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf("%d ", h_C[i][j]);
		printf("\n");
	}
	printf("\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}