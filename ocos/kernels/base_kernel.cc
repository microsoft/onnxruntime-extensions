// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <functional>

#include "base_kernel.hpp"

BaseKernel::BaseKernel(OrtApi api) : api_(api), ort_(api_) {
}