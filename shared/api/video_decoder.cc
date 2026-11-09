// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "video_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/intreadwrite.h>
#include <libswscale/swscale.h>
}

static_assert(LIBAVFORMAT_VERSION_MAJOR == 62 && LIBAVCODEC_VERSION_MAJOR == 62 && LIBAVUTIL_VERSION_MAJOR == 60 &&
                  LIBSWSCALE_VERSION_MAJOR == 9,
              "The video decoder requires the qualified FFmpeg 8.x SDK");

namespace ort_extensions {
namespace {

struct FileInput {
  std::ifstream file;
  int64_t size{};
  int64_t position{};
  size_t read_bytes{};
  size_t read_limit{};
  const char* failure{};
  const OrtxVideoLimits* limits{};
  std::chrono::steady_clock::time_point deadline;

  static int Read(void* opaque, uint8_t* buffer, int count) {
    auto& input = *static_cast<FileInput*>(opaque);
    if (count <= 0) return AVERROR(EINVAL);
    const auto bytes = static_cast<size_t>((std::min)(static_cast<int64_t>(count), input.size - input.position));
    if (bytes == 0) return AVERROR_EOF;
    if (bytes > input.read_limit - input.read_bytes) {
      input.failure = "Video read-work quota exceeded";
      return AVERROR(EIO);
    }
    input.file.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(bytes));
    input.read_bytes += bytes;
    if (static_cast<size_t>(input.file.gcount()) != bytes) {
      input.failure = "Video file changed or could not be read completely";
      return AVERROR(EIO);
    }
    input.position += static_cast<int64_t>(bytes);
    return static_cast<int>(bytes);
  }

  static int64_t Seek(void* opaque, int64_t offset, int whence) {
    auto& input = *static_cast<FileInput*>(opaque);
    if (whence == AVSEEK_SIZE) return input.size;
    whence &= ~AVSEEK_FORCE;
    int64_t base{};
    if (whence == SEEK_CUR)
      base = input.position;
    else if (whence == SEEK_END)
      base = input.size;
    else if (whence != SEEK_SET)
      return AVERROR(EINVAL);
    if (offset < -base || offset > input.size - base) return AVERROR(EINVAL);
    input.file.clear();
    input.file.seekg(base + offset);
    if (!input.file) return AVERROR(EIO);
    input.position = base + offset;
    return input.position;
  }

  static int Interrupt(void* opaque) {
    auto& input = *static_cast<FileInput*>(opaque);
    if (std::chrono::steady_clock::now() < input.deadline) return 0;
    input.failure = "Video decode deadline exceeded";
    return 1;
  }

  static int GetBuffer(AVCodecContext* context, AVFrame* frame, int flags) {
    auto& input = *static_cast<FileInput*>(context->opaque);
    if (frame->width <= 0 || frame->height <= 0 || frame->width > input.limits->max_dimension ||
        frame->height > input.limits->max_dimension ||
        static_cast<uint64_t>(frame->width) * static_cast<uint64_t>(frame->height) > 16777216) {
      input.failure = "Video codec allocation geometry quota exceeded";
      return AVERROR(ENOMEM);
    }
    return avcodec_default_get_buffer2(context, frame, flags);
  }

  static int RejectExternal(AVFormatContext* context, AVIOContext** output, const char*, int, AVDictionary**) {
    if (output != nullptr) *output = nullptr;
    static_cast<FileInput*>(context->opaque)->failure = "External video references are unsupported";
    return AVERROR(EPERM);
  }
};

struct Decoder {
  AVIOContext* io{};
  AVFormatContext* format{};
  AVCodecContext* codec{};
  AVPacket* packet{};
  AVFrame* frame{};
  SwsContext* scaler{};

  ~Decoder() {
    sws_freeContext(scaler);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io != nullptr) av_freep(&io->buffer);
    avio_context_free(&io);
  }
};

OrtxStatus Validate(const OrtxVideoLoadOptions& options, const OrtxVideoLimits& limits, std::vector<double>& targets) {
  if (options.struct_size < sizeof(options) || limits.struct_size < sizeof(limits) || limits.max_frames == 0 ||
      limits.max_rgb_bytes == 0 || limits.max_file_bytes == 0 || limits.max_read_bytes == 0 ||
      limits.max_packets == 0 || limits.max_decoded_frames == 0 || limits.max_targets == 0 ||
      limits.max_targets > (std::numeric_limits<size_t>::max)() / sizeof(OrtxVideoSample) ||
      limits.max_dimension <= 0 || limits.max_dimension > 16384 || !std::isfinite(options.start_seconds) ||
      options.start_seconds < 0 || (options.has_end != 0 && options.has_end != 1) ||
      (options.has_end && (!std::isfinite(options.end_seconds) || options.end_seconds <= options.start_seconds))) {
    return {kOrtxErrorInvalidArgument, "Invalid video sampling range or loading limits"};
  }
  if (options.sampling == kOrtxVideoSamplingFps) {
    if (!std::isfinite(options.sampling_fps) || options.sampling_fps <= 0 || options.frame_count != 0 ||
        options.target_count != 0 || options.target_times != nullptr) {
      return {kOrtxErrorInvalidArgument, "FPS sampling requires exactly one finite positive rate"};
    }
    if (options.has_end) {
      for (size_t i = 0;; ++i) {
        const double target = options.start_seconds + static_cast<double>(i) / options.sampling_fps;
        if (!std::isfinite(target) || (i != 0 && target <= targets.back())) {
          return {kOrtxErrorInvalidArgument, "FPS targets overflow or cannot advance"};
        }
        if (target >= options.end_seconds) break;
        if (i == limits.max_targets) return {kOrtxErrorOutOfRange, "Video target-work quota exceeded"};
        targets.push_back(target);
      }
    }
  } else if (options.sampling == kOrtxVideoSamplingCount) {
    if (!options.has_end || options.frame_count == 0 || options.frame_count > limits.max_targets ||
        options.frame_count > limits.max_frames || options.sampling_fps != 0 || options.target_count != 0 ||
        options.target_times != nullptr) {
      return {kOrtxErrorInvalidArgument, "Count sampling requires a bounded positive count and explicit range"};
    }
    for (size_t i = 0; i < options.frame_count; ++i) {
      const double target =
          options.start_seconds + (options.end_seconds - options.start_seconds) *
                                      (static_cast<double>(i) / static_cast<double>(options.frame_count));
      if (!std::isfinite(target) || target >= options.end_seconds || (i != 0 && target <= targets.back())) {
        return {kOrtxErrorInvalidArgument, "Count targets overflow or cannot advance"};
      }
      targets.push_back(target);
    }
  } else if (options.sampling == kOrtxVideoSamplingTimes) {
    if (options.target_times == nullptr || options.target_count == 0 || options.target_count > limits.max_targets ||
        options.target_count > limits.max_frames || options.sampling_fps != 0 || options.frame_count != 0) {
      return {kOrtxErrorInvalidArgument, "Explicit sampling requires one nonempty bounded target-time array"};
    }
    for (size_t i = 0; i < options.target_count; ++i) {
      const double target = options.target_times[i];
      if (!std::isfinite(target) || target < options.start_seconds ||
          (options.has_end && target >= options.end_seconds) || (i != 0 && target <= options.target_times[i - 1])) {
        return {kOrtxErrorInvalidArgument, "Explicit video times must be finite, increasing and inside the range"};
      }
    }
    targets.assign(options.target_times, options.target_times + options.target_count);
  } else {
    return {kOrtxErrorInvalidArgument, "Unknown video sampling policy"};
  }
  return {};
}

OrtxStatus CopyRgb(Decoder& decoder, const OrtxVideoLimits& limits, const AVCodecParameters& parameters,
                   std::vector<uint8_t>& pixels, int& width, int& height) {
  const auto& frame = *decoder.frame;
  if ((frame.flags & AV_FRAME_FLAG_INTERLACED) != 0 ||
      (frame.sample_aspect_ratio.num > 0 && frame.sample_aspect_ratio.den > 0 &&
       frame.sample_aspect_ratio.num != frame.sample_aspect_ratio.den) ||
      frame.color_trc == AVCOL_TRC_SMPTE2084 || frame.color_trc == AVCOL_TRC_ARIB_STD_B67) {
    return {kOrtxErrorNotImplemented, "Video loading requires progressive square-pixel SDR frames"};
  }
  if (frame.width <= 0 || frame.height <= 0 || frame.width > limits.max_dimension ||
      frame.height > limits.max_dimension ||
      static_cast<uint64_t>(frame.width) * static_cast<uint64_t>(frame.height) > 16777216) {
    return {kOrtxErrorOutOfRange, "Decoded video geometry exceeds the codec allocation limit"};
  }
  int a = 1, b = 0, c = 0, d = 1;
  const auto* side =
      av_packet_side_data_get(parameters.coded_side_data, parameters.nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
  const auto* frame_side = av_frame_get_side_data(decoder.frame, AV_FRAME_DATA_DISPLAYMATRIX);
  const uint8_t* matrix = frame_side != nullptr ? frame_side->data : side != nullptr ? side->data : nullptr;
  const size_t matrix_size = frame_side != nullptr ? frame_side->size : side != nullptr ? side->size : 0;
  if (matrix != nullptr) {
    if (matrix_size < 9 * sizeof(int32_t)) return {kOrtxErrorCorruptData, "Invalid video display matrix"};
    int32_t values[9];
    std::memcpy(values, matrix, sizeof(values));
    if (values[2] != 0 || values[5] != 0 || values[8] != (1 << 30)) {
      return {kOrtxErrorNotImplemented, "Projective video orientation is unsupported"};
    }
    for (const auto index : {0, 1, 3, 4}) {
      if (values[index] != 0 && values[index] != 65536 && values[index] != -65536) {
        return {kOrtxErrorNotImplemented, "Only orthogonal video rotation/reflection is supported"};
      }
    }
    a = values[0] / 65536;
    b = values[1] / 65536;
    c = values[3] / 65536;
    d = values[4] / 65536;
    if (a * a + c * c != 1 || b * b + d * d != 1 || a * b + c * d != 0) {
      return {kOrtxErrorCorruptData, "Invalid orthogonal video orientation"};
    }
  }
  width = a != 0 ? frame.width : frame.height;
  height = d != 0 ? frame.height : frame.width;
  const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  if (bytes > limits.max_rgb_bytes) return {kOrtxErrorOutOfRange, "Selected video RGB quota exceeded"};
  decoder.scaler =
      sws_getCachedContext(decoder.scaler, frame.width, frame.height, static_cast<AVPixelFormat>(frame.format),
                           frame.width, frame.height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (decoder.scaler == nullptr) return {kOrtxErrorOutOfMemory, "Video color-conversion allocation failed"};
  int colorspace = SWS_CS_DEFAULT;
  switch (frame.colorspace) {
    case AVCOL_SPC_UNSPECIFIED:
    case AVCOL_SPC_RGB:
      break;
    case AVCOL_SPC_BT709:
      colorspace = SWS_CS_ITU709;
      break;
    case AVCOL_SPC_FCC:
      colorspace = SWS_CS_FCC;
      break;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      colorspace = SWS_CS_ITU601;
      break;
    case AVCOL_SPC_SMPTE240M:
      colorspace = SWS_CS_SMPTE240M;
      break;
    case AVCOL_SPC_BT2020_NCL:
      colorspace = SWS_CS_BT2020;
      break;
    default:
      return {kOrtxErrorNotImplemented, "Unsupported video color matrix"};
  }
  const auto* coefficients = sws_getCoefficients(colorspace);
  if (sws_setColorspaceDetails(decoder.scaler, coefficients, frame.color_range == AVCOL_RANGE_JPEG, coefficients, 1, 0,
                               1 << 16, 1 << 16) < 0) {
    return {kOrtxErrorNotImplemented, "Unsupported video color conversion"};
  }
  std::vector<uint8_t> rgb(bytes);
  uint8_t* output[4] = {rgb.data()};
  const int strides[4] = {frame.width * 3};
  if (sws_scale(decoder.scaler, frame.data, frame.linesize, 0, frame.height, output, strides) != frame.height) {
    return {kOrtxErrorCorruptData, "Incomplete video color conversion"};
  }
  if (a == 1 && d == 1) {
    pixels = std::move(rgb);
    return {};
  }
  pixels.resize(bytes);
  const int min_x = (std::min)(0, a * (frame.width - 1)) + (std::min)(0, c * (frame.height - 1));
  const int min_y = (std::min)(0, b * (frame.width - 1)) + (std::min)(0, d * (frame.height - 1));
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) {
      const size_t destination = (static_cast<size_t>(b * x + d * y - min_y) * static_cast<size_t>(width) +
                                  static_cast<size_t>(a * x + c * y - min_x)) *
                                 3;
      const size_t source = (static_cast<size_t>(y) * static_cast<size_t>(frame.width) + static_cast<size_t>(x)) * 3;
      std::memcpy(pixels.data() + destination, rgb.data() + source, 3);
    }
  }
  return {};
}
}  // namespace

OrtxStatus ValidateVideoOptions(const OrtxVideoLoadOptions& options, const OrtxVideoLimits& limits) {
  std::vector<double> targets;
  return Validate(options, limits, targets);
}

OrtxStatus DecodeVideo(const char* path, const OrtxVideoLoadOptions& options, const OrtxVideoLimits& limits,
                       std::vector<std::vector<uint8_t>>& pixels, std::vector<OrtxFrame>& frames,
                       std::vector<OrtxVideoSample>& samples, OrtxVideoMetadata& metadata) {
  std::vector<double> targets;
  ORTX_RETURN_IF_ERROR(Validate(options, limits, targets));
#ifdef __cpp_char8_t
  std::u8string utf8_path(std::strlen(path), u8'\0');
  std::memcpy(utf8_path.data(), path, utf8_path.size());
  const std::filesystem::path file_path{utf8_path};
#else
  const auto file_path = std::filesystem::u8path(path);
#endif
  if (!std::filesystem::is_regular_file(file_path)) return {kOrtxErrorInvalidFile, "Video input must be a local file"};
  FileInput input;
  input.file.open(file_path, std::ios::binary | std::ios::ate);
  if (!input.file) return {kOrtxErrorInvalidFile, "Could not open video file"};
  const auto file_size = input.file.tellg();
  if (file_size <= 0 || static_cast<uint64_t>(file_size) > limits.max_file_bytes) {
    return {kOrtxErrorOutOfRange, "Video file is empty or exceeds the encoded-input quota"};
  }
  input.size = static_cast<int64_t>(file_size);
  input.file.seekg(0);
  input.read_limit = limits.max_read_bytes;
  input.limits = &limits;
  input.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  Decoder decoder;
  auto* buffer = static_cast<uint8_t*>(av_malloc(32768));
  if (buffer == nullptr) return {kOrtxErrorOutOfMemory, "Video I/O allocation failed"};
  decoder.io = avio_alloc_context(buffer, 32768, 0, &input, FileInput::Read, nullptr, FileInput::Seek);
  if (decoder.io == nullptr) {
    av_free(buffer);
    return {kOrtxErrorOutOfMemory, "Video I/O context allocation failed"};
  }
  decoder.format = avformat_alloc_context();
  if (decoder.format == nullptr) return {kOrtxErrorOutOfMemory, "Video format allocation failed"};
  decoder.format->pb = decoder.io;
  decoder.format->opaque = &input;
  decoder.format->io_open = FileInput::RejectExternal;
  decoder.format->flags |= AVFMT_FLAG_CUSTOM_IO;
  decoder.format->max_streams = 8;
  decoder.format->probesize = 1024 * 1024;
  decoder.format->max_index_size = 65536;
  decoder.format->interrupt_callback = {FileInput::Interrupt, &input};
  if (avformat_open_input(&decoder.format, nullptr, av_find_input_format("mov"), nullptr) < 0) {
    return {input.failure != nullptr ? kOrtxErrorOutOfRange : kOrtxErrorInvalidFile,
            input.failure != nullptr ? input.failure : "Invalid MP4 video input"};
  }
  int stream_index = -1;
  for (unsigned i = 0; i < decoder.format->nb_streams; ++i) {
    if (decoder.format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      stream_index = static_cast<int>(i);
      break;
    }
  }
  if (stream_index < 0) return {kOrtxErrorInvalidFile, "MP4 contains no video stream"};
  const auto& stream = *decoder.format->streams[stream_index];
  const auto& parameters = *stream.codecpar;
  if (parameters.codec_id != AV_CODEC_ID_H264 || stream.time_base.num <= 0 || stream.time_base.den <= 0) {
    return {kOrtxErrorNotImplemented, "Video loading requires H.264 with a valid presentation time base"};
  }
  if (parameters.width <= 0 || parameters.height <= 0 || parameters.width > limits.max_dimension ||
      parameters.height > limits.max_dimension ||
      static_cast<uint64_t>(parameters.width) * static_cast<uint64_t>(parameters.height) > 16777216) {
    return {kOrtxErrorOutOfRange, "Video header geometry exceeds the codec allocation quota"};
  }
  const auto* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (codec == nullptr) return {kOrtxErrorNotImplemented, "FFmpeg H.264 decoder is unavailable"};
  decoder.codec = avcodec_alloc_context3(codec);
  decoder.packet = av_packet_alloc();
  decoder.frame = av_frame_alloc();
  if (decoder.codec == nullptr || decoder.packet == nullptr || decoder.frame == nullptr) {
    return {kOrtxErrorOutOfMemory, "Video decoder allocation failed"};
  }
  if (avcodec_parameters_to_context(decoder.codec, &parameters) < 0) {
    return {kOrtxErrorCorruptData, "Invalid H.264 decoder parameters"};
  }
  decoder.codec->thread_count = 1;
  decoder.codec->opaque = &input;
  decoder.codec->get_buffer2 = FileInput::GetBuffer;
  decoder.codec->max_pixels = (std::min)(limits.max_dimension * limits.max_dimension, int64_t{16777216});
  decoder.codec->err_recognition = AV_EF_EXPLODE | AV_EF_CAREFUL;
  if (avcodec_open2(decoder.codec, codec, nullptr) < 0)
    return {kOrtxErrorCorruptData, "Could not initialize H.264 decoder"};

  metadata = {sizeof(OrtxVideoMetadata), stream.time_base.num, stream.time_base.den, stream.start_time, 0, 0};
  size_t target_index = 0, decoded_frames = 0, packet_count = 0, rgb_bytes = 0;
  int64_t last_pts = AV_NOPTS_VALUE;
  bool done = false, draining = false;
  auto target_at = [&](size_t index) -> double {
    if (!targets.empty()) return index < targets.size() ? targets[index] : (std::numeric_limits<double>::infinity)();
    return options.start_seconds + static_cast<double>(index) / options.sampling_fps;
  };
  while (!done) {
    if (FileInput::Interrupt(&input)) return {kOrtxErrorOutOfRange, input.failure};
    int result = avcodec_receive_frame(decoder.codec, decoder.frame);
    if (input.failure != nullptr) return {kOrtxErrorOutOfRange, input.failure};
    if (result == AVERROR(EAGAIN)) {
      if (draining) return {kOrtxErrorCorruptData, "H.264 decoder did not drain"};
      if (packet_count == limits.max_packets) return {kOrtxErrorOutOfRange, "Video packet-work quota exceeded"};
      result = av_read_frame(decoder.format, decoder.packet);
      if (input.failure != nullptr) return {kOrtxErrorOutOfRange, input.failure};
      if (result == AVERROR_EOF) {
        draining = true;
        if (avcodec_send_packet(decoder.codec, nullptr) < 0)
          return {kOrtxErrorCorruptData, "Could not drain H.264 decoder"};
        continue;
      }
      if (result < 0 || (decoder.packet->flags & AV_PKT_FLAG_CORRUPT) != 0) {
        return {kOrtxErrorCorruptData, "Corrupt MP4 packet"};
      }
      ++packet_count;
      if (decoder.packet->stream_index == stream_index && avcodec_send_packet(decoder.codec, decoder.packet) < 0) {
        return {kOrtxErrorCorruptData, "Corrupt H.264 packet"};
      }
      av_packet_unref(decoder.packet);
      continue;
    }
    if (result == AVERROR_EOF) break;
    if (result < 0 || decoder.frame->decode_error_flags != 0) return {kOrtxErrorCorruptData, "Corrupt H.264 frame"};
    if (decoded_frames == limits.max_decoded_frames) return {kOrtxErrorOutOfRange, "Video frame-work quota exceeded"};
    const size_t source_index = decoded_frames++;
    const int64_t pts = decoder.frame->pts;
    if (pts == AV_NOPTS_VALUE || (last_pts != AV_NOPTS_VALUE && pts < last_pts)) {
      return {kOrtxErrorCorruptData, "Video has missing or decreasing presentation timestamps"};
    }
    last_pts = pts;
    if (metadata.origin_pts == AV_NOPTS_VALUE) metadata.origin_pts = pts;
    const uint64_t relative = static_cast<uint64_t>(pts) - static_cast<uint64_t>(metadata.origin_pts);
    if (pts < metadata.origin_pts || relative > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
      return {kOrtxErrorCorruptData, "Video presentation timeline overflows"};
    }
    const double seconds =
        static_cast<double>(static_cast<long double>(relative) * stream.time_base.num / stream.time_base.den);
    if (!std::isfinite(seconds)) return {kOrtxErrorCorruptData, "Invalid video presentation timeline"};
    if (options.has_end && seconds >= options.end_seconds) {
      done = true;
    } else {
      size_t selected_index = (std::numeric_limits<size_t>::max)();
      while (target_at(target_index) <= seconds) {
        if (target_index == limits.max_targets) return {kOrtxErrorOutOfRange, "Video target-work quota exceeded"};
        const double target = target_at(target_index);
        if (!std::isfinite(target) || (target_index != 0 && target <= samples.back().target_seconds)) {
          return {kOrtxErrorInvalidArgument, "Video targets overflow or cannot advance"};
        }
        if (selected_index == (std::numeric_limits<size_t>::max)()) {
          if (frames.size() == limits.max_frames) return {kOrtxErrorOutOfRange, "Selected video frame quota exceeded"};
          if (decoder.frame->width <= 0 || decoder.frame->height <= 0 || decoder.frame->width > limits.max_dimension ||
              decoder.frame->height > limits.max_dimension) {
            return {kOrtxErrorOutOfRange, "Selected video geometry exceeds the dimension quota"};
          }
          const auto bytes = static_cast<size_t>(decoder.frame->width) * static_cast<size_t>(decoder.frame->height) * 3;
          if (bytes > limits.max_rgb_bytes - rgb_bytes)
            return {kOrtxErrorOutOfRange, "Selected video RGB quota exceeded"};
          std::vector<uint8_t> rgb;
          int width{}, height{};
          ORTX_RETURN_IF_ERROR(CopyRgb(decoder, limits, parameters, rgb, width, height));
          rgb_bytes += rgb.size();
          selected_index = frames.size();
          pixels.push_back(std::move(rgb));
          frames.push_back(
              {pixels.back().data(), pixels.back().size(), width, height, static_cast<size_t>(width) * 3, seconds});
        } else if (options.sampling == kOrtxVideoSamplingTimes) {
          return {kOrtxErrorOutOfRange, "Explicit video targets select the same source frame"};
        }
        samples.push_back({target, selected_index, pts, source_index});
        ++target_index;
      }
      done = !targets.empty() && target_index == targets.size();
    }
    av_frame_unref(decoder.frame);
  }
  if (options.sampling == kOrtxVideoSamplingTimes && target_index != targets.size()) {
    return {kOrtxErrorOutOfRange, "Explicit video target has no following in-range frame"};
  }
  do {
    const double target = target_at(target_index);
    if (!std::isfinite(target) || (options.has_end && target >= options.end_seconds)) break;
    if (target_index == limits.max_targets) return {kOrtxErrorOutOfRange, "Video target-work quota exceeded"};
    samples.push_back(
        {target, (std::numeric_limits<size_t>::max)(), AV_NOPTS_VALUE, (std::numeric_limits<size_t>::max)()});
    ++target_index;
    if (!options.has_end && targets.empty()) break;
  } while (true);
  if (frames.empty()) return {kOrtxErrorOutOfRange, "Video sampling selected no frames"};
  metadata.requested_samples = samples.size();
  metadata.selected_frames = frames.size();
  return {};
}
}  // namespace ort_extensions
