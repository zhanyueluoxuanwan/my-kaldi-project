#ifndef OPTIONS_STRUCT
#define OPTIONS_STRUCT

#include <string>
#include <vector>
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "cudamatrix/cu-vector.h"
#include "mylib/conf.h"

using namespace kaldi;
using kaldi::int32;
using kaldi::int64;
using std::vector;

struct CudaMelBanksOptions
{
	int32 num_bins;
	BaseFloat low_freq;
	BaseFloat high_freq;
	BaseFloat vtln_low;
	BaseFloat vtln_high;
	bool debug_mel;
	bool htk_mode;
	explicit CudaMelBanksOptions(int num_bins=numBins):
		num_bins(num_bins), low_freq(lowFreq), high_freq(highFreq), vtln_low(vtlnLow),
		vtln_high(vtlnHigh), debug_mel(false), htk_mode(false) {}
};

struct CudaFrameExtractionOptions
{
	BaseFloat samp_freq;
	BaseFloat frame_shift_ms;
	BaseFloat frame_length_ms;
	BaseFloat dither;
	BaseFloat preemph_coeff;
	bool remove_dc_offset;
	std::string window_type;
	bool round_to_power_of_two;
	bool snip_edges;

	CudaFrameExtractionOptions():
		samp_freq(sampFreq), frame_shift_ms(frameLength / 2), frame_length_ms(frameLength),
		dither(1.0), preemph_coeff(0.97), remove_dc_offset(true),
		window_type("povey"), round_to_power_of_two(true), snip_edges(true) {}

	int32 WindowShift() const
	{
		return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
	}

	int32 WindowSize() const
	{
		return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
	}

	int32 PaddedWindowSize() const
	{
		return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) : WindowSize());
	}
};

struct CudaFeatureWindowFunction
{
	CudaFeatureWindowFunction() {}
	explicit CudaFeatureWindowFunction(const CudaFrameExtractionOptions &opts);
	Vector<BaseFloat> window;
};

struct CudaDeltaFeatureOptions
{
	int32 order;
	int32 window;
	CudaDeltaFeatureOptions(int32 order = deltaOrder, int32 window = deltaWindow):
		order(order), window(window){}
};

struct CudaShiftedDeltaFeaturesOptions
{
	int32 window,
		  num_blocks,
		  block_shift;

	CudaShiftedDeltaFeaturesOptions():
		window(1), num_blocks(7), block_shift(3){}
};

struct CudaSlidingWindowCmnOptions
{
	int32 cmn_window;
	int32 min_window;
	bool normalize_variance;
	bool center;

	CudaSlidingWindowCmnOptions():
		cmn_window(cmnWindow), min_window(100), normalize_variance(false),
		center(false) {}
};


#endif
