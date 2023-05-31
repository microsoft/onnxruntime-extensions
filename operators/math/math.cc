#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#include "dlib/stft_norm.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"

const std::vector<const OrtCustomOp*>& MathLoader() {
  static OrtOpLoader op_loader(CustomCpuFunc("NegPos", neg_pos),
#ifdef ENABLE_DLIB
                               CustomCpuFunc("Inverse", inverse),
                               CustomCpuStruct("STFT", STFT),
                               CustomCpuStruct("StftNorm", StftNormal),
#endif
                               CustomCpuFunc("SegmentExtraction", segment_extraction),
                               CustomCpuFunc("SegmentSum", segment_sum));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Math = MathLoader;