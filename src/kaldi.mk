# This file was generated using the following command:
# ./configure --shared

# Rules that enable valgrind debugging ("make valgrind")

valgrind: .valgrind

.valgrind:
	echo -n > valgrind.out
	for x in $(TESTFILES); do echo $$x>>valgrind.out; valgrind ./$$x >/dev/null 2>> valgrind.out; done
	! ( grep 'ERROR SUMMARY' valgrind.out | grep -v '0 errors' )
	! ( grep 'definitely lost' valgrind.out | grep -v -w 0 )
	rm valgrind.out
	touch .valgrind


KALDI_FLAVOR := dynamic
KALDILIBDIR := /home/hanyu/SRE-GPU/src/lib
CONFIGURE_VERSION := 2
FSTROOT = /home/hanyu/SRE-GPU/tools/openfst
OPENFST_VER = 1.4.1
OPENFST_GE_10400 = 1
EXTRA_CXXFLAGS += -DHAVE_OPENFST_GE_10400 -std=c++0x
OPENFSTLIBS = -L/home/hanyu/SRE-GPU/tools/openfst/lib -lfst
OPENFSTLDFLAGS = -Wl,-rpath=/home/hanyu/SRE-GPU/tools/openfst/lib
ATLASINC = /home/hanyu/SRE-GPU/tools/ATLAS/include
ATLASLIBS = /usr/lib/libatlas.so.3 /usr/lib/libf77blas.so.3 /usr/lib/libcblas.so.3 /usr/lib/liblapack_atlas.so.3
# You have to make sure ATLASLIBS is set...

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef ATLASINC
$(error ATLASINC not defined.)
endif

ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif


CXXFLAGS = -msse -msse2 -Wall -I.. \
	   -pthread \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I$(ATLASINC) \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -rdynamic $(OPENFSTLDFLAGS)
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(ATLASLIBS) -lm -lpthread -ldl
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib

#Next section enables CUDA for compilation
CUDA = true
CUDATKDIR = /usr/local/cuda

CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
UNAME := $(shell uname)
#aware of fact in cuda60 there is no lib64, just lib.
ifeq ($(UNAME), Darwin)
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
else
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
endif
CUDA_LDLIBS += -lcublas -lcudart -lcufft -lcurand #LDLIBS : The libs are loaded later than static libs in implicit rule

