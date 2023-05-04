// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include "narrow.h"

namespace OrtX {
constexpr float M_PI = 3.14159265f;  // std::numbers is ready until C+ 20
}  // namespace OrtX

// https://en.wikipedia.org/wiki/Butterworth_filter
class ButterworthLowpassFilter {
 private:
  int order;                        // Filter order
  float cutoffFreq;                 // Cutoff frequency
  std::vector<float> coefficients;  // Filter coefficients

 public:
  ButterworthLowpassFilter(int filterOrder = 4, float cutoffFrequency = 0.2f)
      : order(filterOrder), cutoffFreq(cutoffFrequency) {
    calculateCoefficients();
  }

  // Apply the filter to the input signal with gain adjustment
  std::vector<float> Process(const std::vector<float>& inputSignal, float gain) {
    std::vector<float> outputSignal(inputSignal.size());

    for (int i = 0; i < inputSignal.size(); i++) {
      float output = 0.0f;
      for (int j = 0; j <= order; j++) {
        if (i - j >= 0) {
          output += coefficients[j] * inputSignal[i - j];
        }
      }
      outputSignal[i] = output * gain;  // Apply gain adjustment
    }

    return outputSignal;
  }

 private:
  // Calculate the filter coefficients using Butterworth filter design
  void calculateCoefficients() {
    coefficients.resize(order + 1);

    // Pre-warp the cutoff frequency
    float wc = 2.0f * OrtX::M_PI * cutoffFreq;

    // Calculate analog prototype filter poles
    std::vector<std::complex<float>> poles(order);
    for (int i = 0; i < order; i++) {
      float theta = OrtX::M_PI * (0.5f + (2.0f * i + 1.0f) / (2.0f * order));
      float realPart = -sinh(logf(2.0f) / (2.0f * order) * sinh(theta));
      float imagPart = cosh(logf(2.0f) / (2.0f * order) * sinh(theta));
      poles[i] = std::complex<float>(realPart, imagPart);
    }

    // Apply bilinear transform to get digital filter coefficients
    for (int i = 0; i <= order; i++) {
      std::complex<float> s = poles[i];
      std::complex<float> z = (2.0f + s) / (2.0f - s);
      coefficients[i] = z.real();
    }
  }

 private:
  std::vector<std::complex<float>> poles_;
};

// https://en.wikipedia.org/wiki/Kaiser_window
class KaiserWindowInterpolation {
 private:
  // Kaiser window parameters, empirically
  constexpr static double kBeta = 6.0;

 public:
  static void Process(const std::vector<float>& input, std::vector<float>& output, float inputSampleRate, float outputSampleRate) {
    // Downsampling factor
    float factor = outputSampleRate / inputSampleRate;

    // Calculate the number of output samples
    int outputSize = static_cast<int>(std::ceil(static_cast<float>(input.size()) * factor));
    output.resize(outputSize);

    float beta = 6.0f;  // Beta controls the width of the transition band

    // Interpolation loop
    for (int i = 0; i < outputSize; i++) {
      float index = i / factor;  // Fractional index for interpolation

      // Calculate the integer and fractional parts of the index
      int integerPart = static_cast<int>(index);
      float fractionalPart = index - integerPart;

      // Calculate the range of input samples for interpolation
      int range = static_cast<int>(std::ceil(beta / (2.0f * factor)));
      int startSample = std::max(0, integerPart - range);
      int endSample = std::min(static_cast<int>(input.size()) - 1, integerPart + range);

      // Calculate the Kaiser window weights for the input samples
      std::vector<double> weights = KaiserWin(static_cast<size_t>(endSample - startSample + 1));
      for (int j = startSample; j <= endSample; j++) {
        double distance = std::abs(j - index);
        double sincValue = (distance < 1e-6f) ? 1.0f : std::sin(OrtX::M_PI * distance) / (OrtX::M_PI * distance);
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
  // Kaiser Window function
  static std::vector<double> KaiserWin(size_t window_length) {
    std::vector<double> window(window_length);
    static double i0_beta = std::cyl_bessel_i(0, kBeta);

    for (size_t i = 0; i < window_length; i++) {
      double x = 2.0 * i / (window_length - 1.0) - 1.0;
      double bessel_value = std::cyl_bessel_i(0, kBeta * std::sqrt(1 - x * x));
      window[i] = bessel_value / i0_beta;
    }

    return window;
  }
};
