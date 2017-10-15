#ifndef MY_CUDA_TOOL
#define MY_CUDA_TOOL

#include "cudalib/my-cuda-data-struct.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/kaldi-io.h"
#include "mylib/gettime.h"
#include "cudamatrix/cu-vector.h"

template <class T>
void out_to_file(T mat, std::string filename)
{
	Output out_(filename, false, false);
	if (out_.IsOpen())
	{
		for (int i = 0; i < mat.NumRows(); i++)
		{
			out_.Stream() << "[";
			for (int j = 0; j < mat.NumCols(); j++)
				out_.Stream() << " " << mat(i, j);
			out_.Stream() << " ]\n";
		}
	}
}

template <class T>
bool out_vec_to_file(T vec, std::string filename)
{
	Output out_(filename, false, false);
	if (out_.IsOpen())
	{
		out_.Stream() << "[";
		for (int i = 0; i < vec.Dim(); i++)
				out_.Stream() << " " << *(vec.Data() + i);
		out_.Stream() << " ]\n";
		return true;
	}
	else return false;
	return true;
}

inline bool cuda_ivector_normalize_length(CuVector<double> &ivector)
{
    long long start, end;
    start = getSystemTime();
    double norm = ivector.Norm(2.0);
    double ratio = norm / sqrt(ivector.Dim());
    KALDI_LOG << "Ratio is " << ratio << std::endl;
    if (ratio == 0.0)
    {
        std::cout << "Zero iVector" << std::endl;
        return false;
    }
    else ivector.Scale(1.0 / ratio);
    end = getSystemTime();
    KALDI_LOG << "ivector normalize length time: " << end - start << "ms";
    return true;
}

inline bool cuda_ivector_mean(CuVector<double> &ivector)
{
	long long start, end;
    start = getSystemTime();
    if (ivector.Dim() < 1)
    {
        std::cout << "empty ivector" << std::endl;
        return false;
    }
    ivector.Scale(1.0 / 1);
    end = getSystemTime();
    KALDI_LOG << "ivector mean time: " << end - start << "ms";
    return true;
}


#endif
