// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

void bias_gelu_cuda(const ortc::Tensor<float>& X,
                    const ortc::Tensor<float>& B,
                    ortc::Tensor<float>& Y);