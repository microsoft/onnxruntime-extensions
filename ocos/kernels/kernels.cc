// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <functional>

#include "kernels.h"
#include "matrix.hpp"


static FxGetSchemaInstance custom_op_list[] = {

    CustomOpMatrixDiag::GetInstance,
    CustomOpMatrixBandPart::GetInstance,

    // Indicating the end of the list, don't remove it.
    nullptr
};


FxGetSchemaInstance const* GetCustomOpSchemaList()
{
  return custom_op_list;
}
