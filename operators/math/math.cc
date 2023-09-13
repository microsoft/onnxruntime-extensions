#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#include "dlib/stft_norm.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_Math = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(CustomCpuFuncV2("NegPos", neg_pos),
#ifdef ENABLE_DLIB
                               CustomCpuFuncV2("Inverse", inverse),
                               CustomCpuStructV2("StftNorm", StftNormal),
#endif
                               CustomCpuFuncV2("SegmentExtraction", segment_extraction),
                               CustomCpuFuncV2("SegmentSum", segment_sum));
  return op_loader.GetCustomOps();
};
