// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>

class ButterworthLowpassFilter {
 public:
  ButterworthLowpassFilter(double sample_rate, double cutoff_frequency) {
    double nyquist_frequency = sample_rate / 2;
    double normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency;

    static const float PI = 3.14159265f;
    // Compute filter coefficients using bilinear transformation
    double c = std::tan(PI * normalized_cutoff_frequency);
    double a1 = 1 / (1 + std::sqrt(2) * c + c * c);
    double a2 = 2 * a1;
    double a3 = a1;
    double b1 = 2 * (c * c - 1) * a1;
    double b2 = 1 - std::sqrt(2) * c + c * c;

    // Store filter coefficients
    b_ = {b1, b2, b1};
    a_ = {a1, a2, a3};
  }

  std::vector<double> process(const std::vector<float>& a_x) {
    std::vector<double> output;
    double z1 = 0;
    double z2 = 0;

    output.resize(a_x.size());
    for (size_t n = 0; n < a_x.size(); ++n) {
      auto x = a_x[n];
      // Compute filter output using Direct Form II transposed structure
      double y = b_[0] * x + z1;
      z1 = b_[1] * x - a_[1] * y + z2;
      z2 = b_[2] * x - a_[2] * y;
      output[n] = y;
    }

    return output;
  }

 private:
  std::vector<double> b_;
  std::vector<double> a_;
};

class SincInterpolator {
 public:
  static double process(std::vector<double>& input_audio, double t, double sample_rate_ratio, int filter_length = 101) {
    static const float PI = 3.14159265f;
    // Compute the integer and fractional parts of the input sample index
    int n = (int)std::floor(t);
    double delta = t - n;

    // Compute the sinc function values for the fractional part of the index
    std::vector<double> sinc_coeffs;
    for (int i = 0; i < filter_length; i++) {
      double x = (i - filter_length / 2 + delta) * sample_rate_ratio;
      if (x == 0) {
        sinc_coeffs.push_back(1);
      } else {
        sinc_coeffs.push_back(std::sin(PI * x) / (PI * x));
      }
    }

    // Compute the output sample value by summing the product of input samples and sinc coefficients
    double output_sample = 0;
    for (int i = 0; i < filter_length; i++) {
      int k = n - filter_length / 2 + i;
      if (k >= 0 && k < input_audio.size()) {
        output_sample += sinc_coeffs[i] * input_audio[k];
      }
    }
    return output_sample;
  }
};
