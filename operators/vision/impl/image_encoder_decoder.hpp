#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace ort_extensions {
class BaseImageDecoder {
 public:
  virtual ~BaseImageDecoder() {}

  // HWC
  const std::vector<int64_t>& Shape() const { return shape_; }
  int64_t NumDecodedBytes() const { return decoded_bytes_; }

  bool Decode(uint8_t* output, uint64_t out_bytes) {
    // temporary hack to validate size
    assert(shape_.size() == 3 && out_bytes == shape_[0] * shape_[1] * shape_[2]);
    return DecodeImpl(output, out_bytes);
  }

 protected:
  BaseImageDecoder(const uint8_t* bytes, uint64_t num_bytes)
      : bytes_{bytes}, num_bytes_{num_bytes} {
  }

  const uint8_t* Bytes() const { return bytes_; }
  uint64_t NumBytes() const { return num_bytes_; }

  void SetShape(int height, int width, int channels) {
    assert(height > 0 && width > 0 && channels == 3);
    shape_ = {height, width, channels};
    decoded_bytes_ = height * width * channels;
  }

 private:
  virtual bool DecodeImpl(uint8_t* output, uint64_t out_bytes) = 0;

  const uint8_t* bytes_;
  const uint64_t num_bytes_;
  std::vector<int64_t> shape_;
  int64_t decoded_bytes_{0};
};

// class BaseImageEncoder {
//  public:
//   BaseImageEncoder();
//   virtual ~BaseImageEncoder() {}
//   // virtual bool isFormatSupported(int depth) const;
//
//   // virtual bool setDestination(const String& filename);
//   virtual bool Write(uint64_t height, uint64_t width);  // asssuming 3 channels for now
//
//   // virtual bool SetDestination(std::vector<uint8_t>& buf);
//   // virtual bool write(const Mat& img, const std::vector<int>& params) = 0;
//   // virtual bool writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params);
//
//   // virtual String getDescription() const;
//   // virtual ImageEncoder newEncoder() const;
//
//   // virtual void throwOnEror() const;
//
//  private:
//   // String m_description;
//
//   // String m_filename;
//   // std::vector<uchar>* m_buf;
//   // bool m_buf_supported;
//
//   // String m_last_error;
// };
}  // namespace ort_extensions
