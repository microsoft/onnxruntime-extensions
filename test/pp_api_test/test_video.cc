// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_processor.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

using ort_extensions::OrtxObjectPtr;

namespace {
OrtxVideoLoadOptions Count(size_t count, double start = 0, double end = 3) {
  return {sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingCount, 0, count, nullptr, 0, start, end, 1};
}

#ifdef OCOS_ENABLE_VIDEO_DECODER
OrtxVideoLimits Limits() {
  return {sizeof(OrtxVideoLimits),
          64,
          256ULL * 1024 * 1024,
          512ULL * 1024 * 1024,
          1024ULL * 1024 * 1024,
          100000,
          4096,
          4096,
          4096};
}
#endif
}  // namespace

#ifdef OCOS_ENABLE_VIDEO_DECODER
TEST(VideoLoadingTest, CountUsesLeftBinsAndReportsUnderfillWithoutDuplicatingFrames) {
  auto options = Count(8, 0, 4);
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK)
      << OrtxGetLastErrorMessage();
  OrtxVideoMetadata metadata{sizeof(OrtxVideoMetadata)};
  ASSERT_EQ(OrtxGetVideoMetadata(frames.get(), &metadata), kOrtxOK);
  EXPECT_EQ(metadata.requested_samples, 8);
  EXPECT_EQ(metadata.selected_frames, 6);
  for (size_t i = 0; i < metadata.requested_samples; ++i) {
    OrtxVideoSample sample{};
    ASSERT_EQ(OrtxGetVideoSample(frames.get(), i, &sample), kOrtxOK);
    EXPECT_DOUBLE_EQ(sample.target_seconds, static_cast<double>(i) / 2);
    EXPECT_EQ(sample.frame_index, i < 6 ? i : (std::numeric_limits<size_t>::max)());
    if (i < 6) {
      OrtxFrame frame{};
      ASSERT_EQ(OrtxGetRawFrame(frames.get(), i, &frame), kOrtxOK);
      EXPECT_DOUBLE_EQ(frame.timestamp_seconds, sample.target_seconds);
      EXPECT_EQ(frame.width, 96);
      EXPECT_EQ(frame.height, 64);
      EXPECT_EQ(sample.source_frame_index, i);
      EXPECT_DOUBLE_EQ(static_cast<double>(sample.source_pts - metadata.origin_pts) *
                           static_cast<double>(metadata.time_base_num) / static_cast<double>(metadata.time_base_den),
                       frame.timestamp_seconds);
    }
  }
  options = Count(1);
  OrtxObjectPtr<OrtxRawFrames> single;
  ASSERT_EQ(OrtxLoadVideo(single.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK);
  size_t count{};
  ASSERT_EQ(OrtxGetRawFrameCount(single.get(), &count), kOrtxOK);
  EXPECT_EQ(count, 1);
}

TEST(VideoLoadingTest, FpsDeduplicatesFirstFollowingFramesAndKeepsCorrespondence) {
  const OrtxVideoLoadOptions options{sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingFps, 4, 0, nullptr, 0, 0, 3, 1};
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK);
  OrtxVideoMetadata metadata{sizeof(OrtxVideoMetadata)};
  ASSERT_EQ(OrtxGetVideoMetadata(frames.get(), &metadata), kOrtxOK);
  EXPECT_EQ(metadata.requested_samples, 12);
  EXPECT_EQ(metadata.selected_frames, 6);
  const size_t expected[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, (std::numeric_limits<size_t>::max)()};
  for (size_t i = 0; i < 12; ++i) {
    OrtxVideoSample sample{};
    ASSERT_EQ(OrtxGetVideoSample(frames.get(), i, &sample), kOrtxOK);
    EXPECT_EQ(sample.frame_index, expected[i]);
  }
}

TEST(VideoLoadingTest, BFramesAreSelectedInPresentationOrder) {
  const auto options = Count(6);
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/bframes.mp4", &options, nullptr), kOrtxOK)
      << OrtxGetLastErrorMessage();
  const uint8_t colors[][3] = {{240, 20, 20},  {20, 240, 20},  {20, 20, 240},
                               {240, 240, 20}, {20, 240, 240}, {240, 20, 240}};
  const size_t decode_indices[] = {0, 2, 3, 1, 5, 4};
  for (size_t i = 0; i < 6; ++i) {
    OrtxFrame frame{};
    OrtxVideoSample sample{};
    ASSERT_EQ(OrtxGetRawFrame(frames.get(), i, &frame), kOrtxOK);
    ASSERT_EQ(OrtxGetVideoSample(frames.get(), i, &sample), kOrtxOK);
    EXPECT_DOUBLE_EQ(frame.timestamp_seconds, static_cast<double>(i) / 2);
    EXPECT_EQ(sample.source_frame_index, decode_indices[i]);
    const auto* rgb = static_cast<const uint8_t*>(frame.data) + (32 * 96 + 48) * 3;
    for (size_t channel = 0; channel < 3; ++channel) EXPECT_NEAR(rgb[channel], colors[i][channel], 5);
  }
}

TEST(VideoLoadingTest, FullFileFpsStopsAtEofAndReportsTheUncoveredTarget) {
  const OrtxVideoLoadOptions options{sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingFps, 2, 0, nullptr, 0, 0, 0, 0};
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/bframes.mp4", &options, nullptr), kOrtxOK);
  OrtxVideoMetadata metadata{sizeof(OrtxVideoMetadata)};
  ASSERT_EQ(OrtxGetVideoMetadata(frames.get(), &metadata), kOrtxOK);
  EXPECT_EQ(metadata.selected_frames, 6);
  EXPECT_EQ(metadata.requested_samples, 7);
  OrtxVideoSample last{};
  ASSERT_EQ(OrtxGetVideoSample(frames.get(), 6, &last), kOrtxOK);
  EXPECT_DOUBLE_EQ(last.target_seconds, 3);
  EXPECT_EQ(last.frame_index, (std::numeric_limits<size_t>::max)());
}

TEST(VideoLoadingTest, IgnoresAudioWithoutChangingSelectedVideoPixelsOrTimes) {
  const auto options = Count(6);
  OrtxObjectPtr<OrtxRawFrames> video, audio;
  ASSERT_EQ(OrtxLoadVideo(video.ToBeAssigned(), "data/video/bframes.mp4", &options, nullptr), kOrtxOK);
  ASSERT_EQ(OrtxLoadVideo(audio.ToBeAssigned(), "data/video/audio.mp4", &options, nullptr), kOrtxOK)
      << OrtxGetLastErrorMessage();
  for (size_t i = 0; i < 6; ++i) {
    OrtxFrame original{}, result{};
    ASSERT_EQ(OrtxGetRawFrame(video.get(), i, &original), kOrtxOK);
    ASSERT_EQ(OrtxGetRawFrame(audio.get(), i, &result), kOrtxOK);
    EXPECT_DOUBLE_EQ(original.timestamp_seconds, result.timestamp_seconds);
    ASSERT_EQ(original.data_size, result.data_size);
    EXPECT_EQ(std::memcmp(original.data, result.data, original.data_size), 0);
  }
}

TEST(VideoLoadingTest, RejectsUnsupportedHdrAndNonSquarePixelProfiles) {
  const auto options = Count(1);
  for (const auto* path : {"data/video/hdr.mp4", "data/video/sar.mp4"}) {
    OrtxRawFrames* frames = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
    EXPECT_EQ(OrtxLoadVideo(&frames, path, &options, nullptr), kOrtxErrorNotImplemented)
        << path << ": " << OrtxGetLastErrorMessage();
    EXPECT_EQ(frames, nullptr);
  }
}

TEST(VideoLoadingTest, ExplicitTimesAreStrictAndFailedLoadsReturnNoPartialHandle) {
  double times[] = {0.25, 1.25, 2.25};
  OrtxVideoLoadOptions options{sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingTimes, 0, 0, times, 3, 0, 3, 1};
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK);
  for (size_t i = 0; i < 3; ++i) {
    OrtxFrame frame{};
    ASSERT_EQ(OrtxGetRawFrame(frames.get(), i, &frame), kOrtxOK);
    EXPECT_DOUBLE_EQ(frame.timestamp_seconds, 0.5 + static_cast<double>(i));
  }
  for (const auto& invalid : {std::vector<double>{0.1, 0.2}, std::vector<double>{0, 2.75}}) {
    options.target_times = invalid.data();
    options.target_count = invalid.size();
    OrtxRawFrames* rejected = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
    EXPECT_EQ(OrtxLoadVideo(&rejected, "data/video/cfr.mp4", &options, nullptr), kOrtxErrorOutOfRange);
    EXPECT_EQ(rejected, nullptr);
  }
}

TEST(VideoLoadingTest, RejectsContainerOnlyNonSquarePixels) {
  const auto options = Count(1);
  OrtxRawFrames* frames = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
  EXPECT_EQ(OrtxLoadVideo(&frames, "data/video/container_sar.mp4", &options, nullptr), kOrtxErrorNotImplemented)
      << OrtxGetLastErrorMessage();
  EXPECT_EQ(frames, nullptr);
  if (frames != nullptr) ORTX_DISPOSE(frames);
}

TEST(VideoLoadingTest, VfrUsesPresentationTimesNotAverageFrameIndices) {
  double times[] = {0, 0.2, 0.5, 1.2};
  const OrtxVideoLoadOptions options{sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingTimes, 0, 0, times, 4, 0, 2, 1};
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/vfr.mp4", &options, nullptr), kOrtxOK)
      << OrtxGetLastErrorMessage();
  const double expected[] = {0, 0.32, 1.12, 1.24};
  for (size_t i = 0; i < 4; ++i) {
    OrtxFrame frame{};
    ASSERT_EQ(OrtxGetRawFrame(frames.get(), i, &frame), kOrtxOK);
    EXPECT_DOUBLE_EQ(frame.timestamp_seconds, expected[i]);
  }
}

TEST(VideoLoadingTest, SelectedRangePreservesOriginalTimelineAndRgbChannels) {
  const auto options = Count(2, 0.4, 1.4);
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/origin.mp4", &options, nullptr), kOrtxOK);
  OrtxVideoMetadata metadata{sizeof(OrtxVideoMetadata)};
  ASSERT_EQ(OrtxGetVideoMetadata(frames.get(), &metadata), kOrtxOK);
  EXPECT_DOUBLE_EQ(static_cast<double>(metadata.origin_pts) * static_cast<double>(metadata.time_base_num) /
                       static_cast<double>(metadata.time_base_den),
                   5);
  for (size_t i = 0; i < 2; ++i) {
    OrtxFrame frame{};
    ASSERT_EQ(OrtxGetRawFrame(frames.get(), i, &frame), kOrtxOK);
    EXPECT_DOUBLE_EQ(frame.timestamp_seconds, 0.5 + static_cast<double>(i) / 2);
    const auto* rgb = static_cast<const uint8_t*>(frame.data) + (32 * 96 + 48) * 3;
    if (i == 0)
      EXPECT_GT(rgb[1], 200);
    else
      EXPECT_GT(rgb[2], 200);
    EXPECT_LT(rgb[0], 40);
  }
}

TEST(VideoLoadingTest, AppliesOrthogonalOrientationOnce) {
  const auto options = Count(1);
  OrtxObjectPtr<OrtxRawFrames> original, rotated;
  ASSERT_EQ(OrtxLoadVideo(original.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK);
  ASSERT_EQ(OrtxLoadVideo(rotated.ToBeAssigned(), "data/video/rotated.mp4", &options, nullptr), kOrtxOK);
  OrtxFrame source{}, result{};
  ASSERT_EQ(OrtxGetRawFrame(original.get(), 0, &source), kOrtxOK);
  ASSERT_EQ(OrtxGetRawFrame(rotated.get(), 0, &result), kOrtxOK);
  ASSERT_EQ(result.width, source.height);
  ASSERT_EQ(result.height, source.width);
  EXPECT_EQ(result.data_size, source.data_size);
  const auto* from = static_cast<const uint8_t*>(source.data);
  const auto* to = static_cast<const uint8_t*>(result.data);
  for (size_t y = 0; y < 64; ++y) {
    for (size_t x = 0; x < 96; ++x) {
      for (size_t c = 0; c < 3; ++c) {
        EXPECT_EQ(from[(y * 96 + x) * 3 + c], to[((95 - x) * 64 + y) * 3 + c]);
      }
    }
  }
}

TEST(VideoLoadingTest, ValidatesPoliciesAndTargetWorkBeforeOpeningTheFile) {
  auto options = Count(3);
  std::vector<OrtxVideoLoadOptions> invalid;
  options.has_end = 0;
  invalid.push_back(options);
  options = Count(3);
  options.sampling_fps = 2;
  invalid.push_back(options);
  options = Count(0);
  invalid.push_back(options);
  options = Count(3);
  options.end_seconds = (std::numeric_limits<double>::infinity)();
  invalid.push_back(options);
  double times[] = {1, 0};
  options = {sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingTimes, 0, 0, times, 2, 0, 3, 1};
  invalid.push_back(options);
  for (const auto& value : invalid) {
    OrtxRawFrames* frames = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
    EXPECT_EQ(OrtxValidateVideoLoadOptions(&value, nullptr), kOrtxErrorInvalidArgument);
    EXPECT_EQ(OrtxLoadVideo(&frames, "missing.mp4", &value, nullptr), kOrtxErrorInvalidArgument);
    EXPECT_EQ(frames, nullptr);
  }
  options = {sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingFps, 10000, 0, nullptr, 0, 0, 3, 1};
  OrtxRawFrames* frames{};
  EXPECT_EQ(OrtxValidateVideoLoadOptions(&options, nullptr), kOrtxErrorOutOfRange);
  EXPECT_EQ(OrtxLoadVideo(&frames, "missing.mp4", &options, nullptr), kOrtxErrorOutOfRange);
  EXPECT_EQ(frames, nullptr);
  options = Count(3);
  EXPECT_EQ(OrtxValidateVideoLoadOptions(&options, nullptr), kOrtxOK);
  EXPECT_EQ(OrtxValidateVideoLoadOptions(nullptr, nullptr), kOrtxErrorInvalidArgument);
}

TEST(VideoLoadingTest, EnforcesOutputReadPacketFrameAndCodecGeometryQuotas) {
  const auto options = Count(6);
  for (int quota = 0; quota < 6; ++quota) {
    auto limits = Limits();
    switch (quota) {
      case 0:
        limits.max_file_bytes = 1;
        break;
      case 1:
        limits.max_read_bytes = 1;
        break;
      case 2:
        limits.max_rgb_bytes = 96 * 64 * 3;
        break;
      case 3:
        limits.max_packets = 1;
        break;
      case 4:
        limits.max_decoded_frames = 1;
        break;
      case 5:
        limits.max_dimension = 32;
        break;
    }
    OrtxRawFrames* frames = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
    EXPECT_EQ(OrtxLoadVideo(&frames, "data/video/cfr.mp4", &options, &limits), kOrtxErrorOutOfRange) << quota;
    EXPECT_EQ(frames, nullptr) << quota;
  }
  OrtxObjectPtr<OrtxRawFrames> recovered;
  ASSERT_EQ(OrtxLoadVideo(recovered.ToBeAssigned(), "data/video/cfr.mp4", &options, nullptr), kOrtxOK);
}

TEST(VideoLoadingTest, ExactPacketQuotaStillDrainsDelayedFrames) {
  const auto options = Count(6);
  auto limits = Limits();
  limits.max_packets = 6;
  limits.max_decoded_frames = 6;
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxLoadVideo(frames.ToBeAssigned(), "data/video/bframes.mp4", &options, &limits), kOrtxOK)
      << OrtxGetLastErrorMessage();
  OrtxFrame last{};
  ASSERT_EQ(OrtxGetRawFrame(frames.get(), 5, &last), kOrtxOK);
  EXPECT_DOUBLE_EQ(last.timestamp_seconds, 2.5);
  limits.max_packets = 5;
  OrtxRawFrames* rejected = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
  EXPECT_EQ(OrtxLoadVideo(&rejected, "data/video/bframes.mp4", &options, &limits), kOrtxErrorOutOfRange);
  EXPECT_EQ(rejected, nullptr);
}

TEST(VideoLoadingTest, UnicodePathAndTruncatedPacketsRejectUnderfill) {
  std::ifstream input("data/video/cfr.mp4", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  ASSERT_GT(bytes.size(), 128);
  struct Temporary {
    std::filesystem::path path = std::filesystem::temp_directory_path() / u8"ortx-video-\u00e9-";
    Temporary() { path += std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mp4"; }
    ~Temporary() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } temporary;
  const auto options = Count(6);
  const auto utf8 = temporary.path.u8string();
  const std::string path{reinterpret_cast<const char*>(utf8.data()), utf8.size()};
  std::filesystem::copy_file("data/video/cfr.mp4", temporary.path);
  {
    OrtxObjectPtr<OrtxRawFrames> complete;
    ASSERT_EQ(OrtxLoadVideo(complete.ToBeAssigned(), path.c_str(), &options, nullptr), kOrtxOK)
        << OrtxGetLastErrorMessage();
  }
  const OrtxVideoLoadOptions fps{sizeof(OrtxVideoLoadOptions), kOrtxVideoSamplingFps, 2, 0, nullptr, 0, 0, 3, 1};
  for (const size_t removed : {65u, 128u}) {
    {
      std::ofstream output(temporary.path, std::ios::binary);
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - removed));
      ASSERT_TRUE(output);
    }
    for (const auto& sampling : {options, fps}) {
      OrtxRawFrames* frames = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
      EXPECT_EQ(OrtxLoadVideo(&frames, path.c_str(), &sampling, nullptr), kOrtxErrorCorruptData) << removed;
      EXPECT_EQ(frames, nullptr);
      if (frames != nullptr) ORTX_DISPOSE(frames);
    }
  }
}
#else
TEST(VideoLoadingTest, DecoderDisabledRetainsSymbolsAndFrameInputs) {
  const auto options = Count(1);
  EXPECT_EQ(OrtxValidateVideoLoadOptions(&options, nullptr), kOrtxErrorNotImplemented);
  OrtxRawFrames* loaded = reinterpret_cast<OrtxRawFrames*>(uintptr_t{1});
  EXPECT_EQ(OrtxLoadVideo(&loaded, "unused.mp4", &options, nullptr), kOrtxErrorNotImplemented);
  EXPECT_EQ(loaded, nullptr);
  const uint8_t pixels[3] = {1, 2, 3};
  const OrtxFrame descriptor{pixels, sizeof(pixels), 1, 1, 3, 0};
  OrtxObjectPtr<OrtxRawFrames> frames;
  ASSERT_EQ(OrtxCreateRawFrames(frames.ToBeAssigned(), &descriptor, 1, nullptr), kOrtxOK);
  size_t count{};
  ASSERT_EQ(OrtxGetRawFrameCount(frames.get(), &count), kOrtxOK);
  EXPECT_EQ(count, 1);
  OrtxVideoMetadata metadata{sizeof(OrtxVideoMetadata)};
  EXPECT_EQ(OrtxGetVideoMetadata(frames.get(), &metadata), kOrtxErrorNotImplemented);
}
#endif
