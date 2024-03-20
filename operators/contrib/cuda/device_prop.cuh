// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

struct DeviceProp {
  static cudaDeviceProp& GetCudaDeviceProp() {
    static DeviceProp device_prop;
    return device_prop.prop_;
  }
  static int GetCapability() {
    return GetCudaDeviceProp().major;
  }

 private:
  DeviceProp() {
    auto err = cudaGetDeviceProperties(&prop_, 0);
    if (err != cudaError::cudaSuccess) {
      throw std::runtime_error((std::string{"Failed to get device property, err code: "} + std::to_string(err)).c_str());
    }
  }
  cudaDeviceProp prop_;
};

