// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include "narrow.h"

// https://en.wikipedia.org/wiki/Butterworth_filter
class ButterworthLowpass {
 public:
  ButterworthLowpass(float sample_rate, float cutoff_frequency)
      : x_prev_(0.0f), y_prev_(0.0f) {
    float RC = 1.0f / (2.0f * 3.14159265359f * cutoff_frequency);
    float dt = 1.0f / sample_rate;
    float alpha = dt / (RC + dt);
    a0_ = alpha;
    a1_ = alpha;
    b1_ = 1 - alpha;
  }

  float Process(float input) {
    float output = a0_ * input + a1_ * x_prev_ - b1_ * y_prev_;
    x_prev_ = input;
    y_prev_ = output;
    return output;
  }

  std::vector<float> Process(const std::vector<float>& inputSignal) {
    std::vector<float> outputSignal(inputSignal.size());
    for (size_t i = 0; i < inputSignal.size(); ++i) {
      outputSignal[i] = Process(inputSignal[i]);
    }
    return outputSignal;
  }

 private:
  float x_prev_, y_prev_;
  float a0_, a1_, b1_;
};

// https://ccrma.stanford.edu/~jos/sasp/Kaiser_Window.html
class KaiserWindowInterpolation {
 private:
  // Kaiser window parameters, empirically
  constexpr static double kBeta = 6.0; // Beta controls the width of the transition band

 public:
  static void Process(const std::vector<float>& input, std::vector<float>& output, float inputSampleRate, float outputSampleRate) {
    // Downsampling factor
    float factor = outputSampleRate / inputSampleRate;
    const double MY_PI = 3.14159265359;

    // Calculate the number of output samples
    int outputSize = static_cast<int>(std::ceil(static_cast<float>(input.size()) * factor));
    output.resize(outputSize);

    for (int i = 0; i < outputSize; i++) {
      float index = i / factor;  // Fractional index for interpolation

      // Calculate the integer and fractional parts of the index
      int integerPart = static_cast<int>(index);
      float fractionalPart = index - integerPart;

      // Calculate the range of input samples for interpolation
      int range = static_cast<int>(std::ceil(kBeta / (2.0f * factor)));
      int startSample = std::max(0, integerPart - range);
      int endSample = std::min(static_cast<int>(input.size()) - 1, integerPart + range);

      // Calculate the Kaiser window weights for the input samples
      std::vector<double> weights = KaiserWin(static_cast<size_t>(endSample - startSample + 1));
      for (int j = startSample; j <= endSample; j++) {
        double distance = std::abs(j - index);
        double sincValue = (distance < 1e-6f) ? 1.0f : std::sin(MY_PI * distance) / (MY_PI * distance);
        weights[j - startSample] *= sincValue;
      }

      // Perform the interpolation
      double interpolatedValue = 0.0f;
      for (int j = startSample; j <= endSample; j++) {
        interpolatedValue += input[j] * weights[j - startSample];
      }

      output[i] = static_cast<float>(interpolatedValue);
    }
  }

 private:
  // std::cyl_bessel_i is not available for every platform.
  static double cyl_bessel_i0(double x) {
    double sum = 0.0;
    double term = 1.0;
    double x_squared = x * x / 4.0;
    int n = 0;
    double tolerance = 1e-8;

    while (term > tolerance * sum) {
      sum += term;
      n += 1;
      term *= x_squared / (n * n);
    }

    return sum;
  }

  // Kaiser Window function
  static std::vector<double> KaiserWin(size_t window_length) {
    std::vector<double> window(window_length);
    static const double i0_beta = cyl_bessel_i0(kBeta);

    for (size_t i = 0; i < window_length; i++) {
      double x = 2.0 * i / (window_length - 1.0) - 1.0;
      double bessel_value = cyl_bessel_i0(kBeta * std::sqrt(1 - x * x));
      window[i] = bessel_value / i0_beta;
    }

    return window;
  }
};
