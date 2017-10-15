#include "cudalib/my-cuda-data-struct.h"

void CuMatrixInt::Resize(int32_cuda rows, int32_cuda cols)
{
	KALDI_ASSERT(rows > 0 || cols > 0);
	int32_cuda dataSize = rows * cols * sizeof(int32_cuda);
	this->data_ = (int32_cuda *)malloc(dataSize);
	this->rows_ = rows;
	this->cols_ = cols;
}
