#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"

const std::vector<const OrtCustomOp*>& MathLoader() {
  static OrtOpLoader op_loader("NegPos", neg_pos,
#ifdef ENABLE_DLIB
                               "Inverse", inverse,
#endif
                               "SegmentExtraction", segment_extraction,
                               "SegmentSum", segment_sum);
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Math = MathLoader;