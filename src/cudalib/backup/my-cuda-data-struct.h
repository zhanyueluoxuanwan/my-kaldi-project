#ifndef MY_CUDA_DATA_STRUCT
#define MY_CUDA_DATA_STRUCT

#if HAVE_CUDA == 1
#include "base/kaldi-common.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrixdim.h"
#include <stdio.h>
#include <stdlib.h>

using namespace kaldi;

class CuMatrixInt
{
	public:
		CuMatrixInt() {}
		CuMatrixInt(int32_cuda rows, int32_cuda cols)
		{
			Resize(rows, cols);
		}

		~CuMatrixInt(){}
		
		void Resize(int32_cuda rows, int32_cuda cols);

		inline int32_cuda operator() (int32_cuda row, int32_cuda col) const
		{
			KALDI_ASSERT(row < rows_ && col < cols_ );
			return *(data_ + row * cols_ + col);
		}


		inline int32_cuda NumRows() const {return rows_;}
		inline int32_cuda NumCols() const {return cols_;}
		inline int32_cuda* Data() const {return data_;}

	private:
		int32_cuda *data_;
		int32_cuda rows_;
		int32_cuda cols_;
};

inline void printCuMatInt(CuMatrixInt const mat)
{
	for (int i = 0; i < mat.NumRows(); i++)
	{
		std::cout << "[";
		for (int j = 0; j < mat.NumCols(); j++)
			std::cout << " " << mat(i, j);
		std::cout << " ]\n";
	}
}

#endif

#endif
