#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#include "dlib/stft_norm.hpp"
#include "energy_stft_segmentation.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"

#ifdef USE_CUDA
#include "cuda/negpos_def.h"
#endif  // USE_CUDA

FxLoadCustomOpFactory LoadCustomOpClasses_Math = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(CustomCpuFuncV2("NegPos", neg_pos),
#ifdef USE_CUDA
                               CustomCudaFuncV2("NegPos", neg_pos_cuda),
#endif
#ifdef ENABLE_DLIB
                               CustomCpuFuncV2("Inverse", inverse), CustomCpuStructV2("StftNorm", StftNormal),
                               CustomCpuFuncV2("DetectEnergySegments", detect_energy_segments),
                               CustomCpuFuncV2("MergeAndFilterAudioSegments", merge_and_filter_segments),
#endif
                               CustomCpuFuncV2("SegmentExtraction", segment_extraction),
                               CustomCpuFuncV2("SegmentSum", segment_sum));

#if defined(USE_CUDA)
  // CustomCudaFunc("NegPos", neg_pos_cuda),
#endif  // USE_CUDA
  return op_loader.GetCustomOps();
};
