package com.pingan.jni;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Properties(value = {
	@Platform(define={"HAVE_POSIX_MEMALIGN", "KALDI_DOUBLEPRECISION 0", "HAVE_EXECINFO_H 1", "HAVE_CXXABI_H 1", "HAVE_ATLAS 1", "HAVE_OPENFST_GE_10400 1", "HAVE_CUDA 1"},
			  include={"mylib/score.h"},
			  includepath={"/home/hanyu/SRE-GPU/src", "/home/hanyu/SRE-GPU/tools/ATLAS/include", "/home/hanyu/SRE-GPU/tools/openfst/include", "/home/hanyu/SRE-GPU/tools/openfst/lib",  "/usr/local/cuda/include"},
			  linkpath={"/home/hanyu/SRE-GPU/src/lib", "/usr/local/cuda/lib64", "/home/hanyu/SRE-GPU/tools/openfst/lib", "/usr/lib"},
			  link={"mylib", "cudalib", "kaldi-feat", "kaldi-ivector", "kaldi-transform", "kaldi-base", "kaldi-gmm", "kaldi-tree", "kaldi-cudamatrix", "kaldi-hmm", "kaldi-matrix", "kaldi-thread", "kaldi-util", "m", "dl", "fst", "cublas", "cudart", "cufft"}
			 )
})

public class ScoreDotProduct extends Pointer
{
	static { Loader.load(); }
	public ScoreDotProduct() { allocate(); }
	private native void allocate();
	public native float score_dot_product(@StdString String file1, @StdString String file2);
}

