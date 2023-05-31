// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include "narrow.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// http://www.dspguide.com/CH33.PDF, p625
class ButterworthLowpass {
 private:
  static constexpr int kFilterOrder = 4;

  // public for test
 public:
  static void OnePoleCoefs(double pole_n, double np, double fc, double a[3], double b[3]) {
    double rp, ip, t, w, m, d, x0, x1, x2, y1, y2, k;

    // Calculate pole location on unit circle
    rp = -cos(M_PI / (np * 2.0) + (pole_n - 1.0) * M_PI / np);
    ip = sin(M_PI / (np * 2.0) + (pole_n - 1.0) * M_PI / np);

    // s-domain to z-domain conversion
    t = 2.0 * tan(0.5);
    w = 2.0 * M_PI * fc;
    m = rp * rp + ip * ip;
    d = 4.0 - 4.0 * rp * t + m * t * t;
    x0 = t * t / d;
    x1 = 2.0 * t * t / d;
    x2 = t * t / d;
    y1 = (8.0 - 2.0 * m * t * t) / d;
    y2 = (-4.0 - 4.0 * rp * t - m * t * t) / d;

    // LP TO LP, or LP TO HP
    k = sin(0.5 - w / 2.0) / sin(0.5 + w / 2.0);

    d = 1.0 + y1 * k - y2 * k * k;

    a[0] = (x0 - x1 * k + x2 * k * k) / d;
    a[1] = (-2.0 * x0 * k + x1 + x1 * k * k - 2.0 * x2 * k) / d;
    a[2] = (x0 * k * k - x1 * k + x2) / d;
    b[1] = (2.0 * k + y1 + y1 * k * k - 2.0 * y2 * k) / d;
    b[2] = (-k * k - y1 * k + y2) / d;
  }

  void CalculateCoefs(std::vector<double>& num, std::vector<double>& den, size_t num_pole, double cutoff_freq) {
    const size_t POLE_DATA_SIZE = 3;
    if (num_pole <= 0) {
      throw std::invalid_argument("num_pole must be greater than zero");
    }

    num.resize(kFilterOrder + 1);
    den.resize(kFilterOrder + 1);

    const size_t pole_buff_size = num_pole + POLE_DATA_SIZE;

    std::vector<double> a(pole_buff_size), b(pole_buff_size), ta(pole_buff_size), tb(pole_buff_size);
    std::array<double, POLE_DATA_SIZE> ap{}, bp{};
    double sa{}, sb{}, gain{};

    a[POLE_DATA_SIZE - 1] = 1.0;
    b[POLE_DATA_SIZE - 1] = 1.0;

    for (auto p = 1; p <= num_pole / 2; ++p) {
      OnePoleCoefs(p, static_cast<double>(num_pole), cutoff_freq, ap.data(), bp.data());

      std::copy(a.begin(), a.end(), ta.begin());
      std::copy(b.begin(), b.end(), tb.begin());

      for (auto i = POLE_DATA_SIZE - 1; i <= num_pole + 2; ++i) {
        a[i] = ap[0] * ta[i] + ap[1] * ta[i - 1] + ap[2] * ta[i - 2];
        b[i] = tb[i] - bp[1] * tb[i - 1] - bp[2] * tb[i - 2];
      }
    }

    b[POLE_DATA_SIZE - 1] = 0.0;

    for (auto i = 0; i <= num_pole; ++i) {
      a[i] = a[i + 2];
      b[i] = -b[i + 2];
    }

    for (auto i = 0; i <= num_pole; ++i) {
      sa += a[i];
      sb += b[i];
    }

    gain = sa / (1.0 - sb);

    for (auto i = 0; i <= num_pole; ++i)
      a[i] /= gain;

    for (auto i = 0; i <= num_pole; ++i) {
      num[i] = a[i];
      den[i] = -b[i];
    }

    den[0] = 1.0;
  }


  std::vector<double> coefs_a_;
  std::vector<double> coefs_b_;

 public:
  ButterworthLowpass(double cutoff_freq, double sampling_rate) {
    auto normalized_cutoff = cutoff_freq / sampling_rate;
    CalculateCoefs(coefs_b_, coefs_a_, kFilterOrder, normalized_cutoff);
  }

  const std::vector<double>& GetCoefs_A() {
    return coefs_a_;
  }

  const std::vector<double>& GetCoefs_B() {
    return coefs_b_;
  }

  std::vector<float> Process(const std::vector<float>& input) {
    std::vector<float> output(input.size(), 0.0);

    // Initialize delay elements
    double x_n_1 = 0.0, x_n_2 = 0.0, x_n_3 = 0.0, x_n_4 = 0.0;
    double y_n_1 = 0.0, y_n_2 = 0.0, y_n_3 = 0.0, y_n_4 = 0.0;

    for (size_t i = 0; i < input.size(); ++i) {
      double x = input[i];

      // Compute the output
      double y = coefs_b_[0] * x + coefs_b_[1] * x_n_1 + coefs_b_[2] * x_n_2 + coefs_b_[3] * x_n_3 + coefs_b_[4] * x_n_4
        - coefs_a_[1] * y_n_1 - coefs_a_[2] * y_n_2 - coefs_a_[3] * y_n_3 - coefs_a_[4] * y_n_4;

      // Shuffle old input and output values
      x_n_4 = x_n_3;
      x_n_3 = x_n_2;
      x_n_2 = x_n_1;
      x_n_1 = x;

      y_n_4 = y_n_3;
      y_n_3 = y_n_2;
      y_n_2 = y_n_1;
      y_n_1 = y;

      output[i] = static_cast<float>(y);
    }

    return output;
  }
};

// https://ccrma.stanford.edu/~jos/sasp/Kaiser_Window.html
class KaiserWindowInterpolation {
 private:
  // Kaiser window parameters, empirically
  static constexpr double kBeta = 6.0;  // Beta controls the width of the transition band

 public:
  static void Process(const std::vector<float>& input, std::vector<float>& output, float inputSampleRate, float outputSampleRate) {
    // Downsampling factor
    float factor = outputSampleRate / inputSampleRate;

    // Calculate the number of output samples
    int outputSize = static_cast<int>(std::ceil(static_cast<float>(input.size()) * factor));
    output.resize(outputSize);

    for (int i = 0; i < outputSize; i++) {
      float index = i / factor;  // Fractional index for interpolation

      // Calculate the integer and fractional parts of the index
      int integerPart = static_cast<int>(index);
      float fractionalPart = index - integerPart;

      // Calculate the range of input samples for interpolation
      int range = static_cast<int>(std::ceil(kBeta / (2.0 * factor)));
      int startSample = std::max(0, integerPart - range);
      int endSample = std::min(static_cast<int>(input.size()) - 1, integerPart + range);

      // Calculate the Kaiser window weights for the input samples
      std::vector<double> weights = KaiserWin(static_cast<size_t>(endSample - startSample + 1));
      for (int j = startSample; j <= endSample; j++) {
        double distance = std::abs(j - index);
        double sincValue = (distance < 1e-6f) ? 1.0f : std::sin(M_PI * distance) / (M_PI * distance);
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
    size_t n = 0;
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
