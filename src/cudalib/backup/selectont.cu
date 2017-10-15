#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "gettime.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__); \
     exit(-1);}cudaThreadSynchronize();} while(0)

void test() {
	printf("\nfor a test");
}

template<typename Real>
__device__
static void __insert_sort(Real *__first, Real *__last)
{
	if (__first == __last)
		return;
	Real *p;
	for (Real *iter = __first + 1; iter != __last; ++iter)
	{
		Real tmp = *iter;
		for (p = iter; p != __first && tmp < *(p - 1); --p)
			*p = *(p - 1);
		*p = tmp;
	}
}

template<typename Real>
__device__
static Real* __partition(Real *__first, Real *__last, Real __pivot)
{
	while(true)
	{
		while (*__first < __pivot)
			++__first;
		--__last;
		while (__pivot < *__last)
			--__last;
		if(!(__first < __last))
			return __first;
		//swap two number
		{
			*__first += *__last;
			*__last = *__first - *__last;
			*__first -= *__last;
		}
		++__first;
	}
}

template<typename Real>
__device__
static void _partition(Real *__first, Real *__nth, Real *__last)
{
	while(__last - __first > 3)
	{
		Real *__cut = __partition(__first, __last, *(__first + (__last - __first) / 2));
		if (__cut <= __nth)
			__first = __cut;
		else
			__last = __cut;
	}
	__insert_sort(__first, __last);
}

template<typename Real>
__global__
static void _gmm_select(Real *data, int rows, int cols, int num_ceps, int *gmm_selected)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < rows)
	{
		Real *dataCopy = (Real *)malloc(cols * sizeof(Real));
		//copy data
		for(int n = 0; n < cols; n++)
			dataCopy[n] = data[row * cols + n];
		//sort copy data to get greater 20 numbers;
		_partition(dataCopy, dataCopy + cols - num_ceps, dataCopy + cols);
		Real thresh = dataCopy[cols - num_ceps];
		printf("thread %d thresh is %f\toffset is %d\n", row, thresh, cols - num_ceps);
		for (int j = 0; j < cols; j++)
			if(*(data + row * cols + j) >= thresh)
			{
				*(gmm_selected + row * num_ceps) = j;
				gmm_selected++;
			}
	}
}


template<typename Real>
__host__
static int *_my_cuda_gmm_select(Real *data, int rows, int cols, int num_ceps)
{
	int threadsPerBlock = 256;
	int blockPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
	int *selected_gauss;
	int *host_selected_gauss = (int *)malloc(rows * num_ceps * sizeof(int));
	Real *devdata;
	CUDA_CALL(cudaMalloc((void **)&selected_gauss, rows * num_ceps * sizeof(int)));
	CUDA_CALL(cudaMalloc((void **)&devdata, rows * cols * sizeof(float)));
	my_time t;
	t.start();
	CUDA_CALL(cudaMemcpy(devdata, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	_gmm_select<<<blockPerGrid, threadsPerBlock>>>(devdata, rows, cols, num_ceps, selected_gauss);
//	_gmm_select<<<1, 1>>>(devdata, dim, num_ceps, selected_gauss);
	CUDA_CALL(cudaMemcpy(host_selected_gauss, selected_gauss, rows * num_ceps * sizeof(int), cudaMemcpyDeviceToHost));
	t.end();
	printf("gpu gmm select used time is:%lld", t.used_time());
	CUDA_CALL(cudaFree(selected_gauss));
	CUDA_CALL(cudaFree(devdata));
	return host_selected_gauss;
}

int main()
{
	float data[4096];
	for (int i = 0; i < 4096; i++)
		data[i] = static_cast<float>(i);
	int num_gselect = 20;
	int *p = _my_cuda_gmm_select(data, 64, 64, num_gselect);
	printf("\n");
	for (int j = 0; j < 64; j++)
	{
		for (int i = 0;i < num_gselect; i++)
			printf("p[%d]=%d  ", i, p[i]);
		printf("\n");
		p += num_gselect;
	}
	return 0;
}


