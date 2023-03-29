// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_hash(const ortc::TensorT<std::string>& input,
                 int64_t num_buckets,
                 ortc::TensorT<int64_t>& output);
void string_hash_fast(const ortc::TensorT<std::string>& input,
                      int64_t num_buckets,
                      ortc::TensorT<int64_t>& output);
