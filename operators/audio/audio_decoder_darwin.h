// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <map>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <list>
#include <array>

#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>
#include <AudioToolbox/AudioFile.h>

#include "audio_decoder.h"

struct AudioToolboxError : public std::runtime_error {
  AudioToolboxError(const std::string& what_arg, OSStatus status) : std::runtime_error(what_arg), status(status) {}
  OSStatus status;
};

struct AudioFile {
  AudioFile(const uint8_t* ptr, size_t size) {
    data_ptr = ptr;
    data_size = size;
  }
  size_t GetProperty(AudioFilePropertyID inPropertyID, size_t inDataSize, void* outPropertyData) {
    UInt32 dataSize = (UInt32)inDataSize;
    const OSStatus err = AudioFileGetProperty(mAudioFileID, inPropertyID, &dataSize, outPropertyData);
    if (err != noErr) {
      std::ostringstream buf;
      buf << "unable to get the property " << inPropertyID << " from the audio converter";
      throw AudioToolboxError(buf.str(), err);
    }
    return (size_t)dataSize;
  }

  std::optional<size_t> GetPropertySize(AudioFilePropertyID inPropertyID) {
    UInt32 size;
    UInt32 isWritable;
    const OSStatus err = AudioFileGetPropertyInfo(mAudioFileID, inPropertyID, &size, &isWritable);
    if (err != noErr) {
      if (err == kAudioFileUnsupportedPropertyError) {
        return std::nullopt;
      } else {
        std::ostringstream buf;
        buf << "unable to get the property " << inPropertyID << " info from the audio converter";
        throw AudioToolboxError(buf.str(), err);
      }
    }
    return (size_t)size;
  }

  void ConverterSetProperty(AudioConverterRef converter, AudioConverterPropertyID inPropertyID, size_t inDataSize,
                            const void* inPropertyData) {
    const OSStatus err = AudioConverterSetProperty(converter, inPropertyID, (UInt32)inDataSize, inPropertyData);
    if (err != noErr) {
      std::ostringstream buf;
      buf << "unable to set the property " << inPropertyID << " on the audio converter";
      throw AudioToolboxError(buf.str(), err);
    }
  }

  void ReadPackets(UInt32& ioNumBytes, AudioStreamPacketDescription* outPacketDescriptions, UInt32& ioNumPackets,
                   void* outBuffer) {
    const OSStatus err = AudioFileReadPacketData(mAudioFileID, false, &ioNumBytes, outPacketDescriptions, mNextPacket,
                                                 &ioNumPackets, outBuffer);
    if (err == noErr) {
      mNextPacket += ioNumPackets;
    } else {
    }
  }

  SInt64 NextPacket() const { return mNextPacket; }

  AudioFileID mAudioFileID;
  SInt64 mNextPacket;

  const uint8_t* data_ptr;
  size_t data_size;
};

class InputContext {
 public:
  InputContext(AudioFile inputFile, AudioStreamBasicDescription inputDescription, UInt32 maxInputPacketSize,
               bool inputUsesPacketDescriptions)
      : mInputFile(std::move(inputFile)),
        mInputDescription(std::move(inputDescription)),
        mMaxInputPacketSize(maxInputPacketSize),
        mInputUsesPacketDescriptions(inputUsesPacketDescriptions) {}

  SInt64 NumPacketsRead() const { return mInputFile.NextPacket(); }

  static OSStatus InputDataProc(AudioConverterRef inAudioConverter, UInt32* ioNumberDataPackets,
                                AudioBufferList* ioData, AudioStreamPacketDescription** outDataPacketDescription,
                                void* inUserData) {
    InputContext& self = *(InputContext*)inUserData;

    // Set up the input buffer.
    self.mInputBuffer.resize((size_t)(*ioNumberDataPackets * self.mMaxInputPacketSize));

    // If the input uses packet descriptions, set up the buffer for that,
    // and provide access to it for the caller by storing a pointer to it
    // in the `outDataPacketDescription` provided.
    if (self.mInputUsesPacketDescriptions) {
      self.mPacketDescriptions.resize((size_t)*ioNumberDataPackets);
      *outDataPacketDescription = self.mPacketDescriptions.data();
    }

    // Fill in the AudioBufferList to refer to the packet buffer.
    ioData->mNumberBuffers = 1;
    ioData->mBuffers[0].mNumberChannels = self.mInputDescription.mChannelsPerFrame;
    ioData->mBuffers[0].mDataByteSize = (UInt32)self.mInputBuffer.size();
    ioData->mBuffers[0].mData = self.mInputBuffer.data();

    // Read packets from the file into the buffer, along with packet descriptions, if any exist.
    try {
      self.mInputFile.ReadPackets(ioData->mBuffers[0].mDataByteSize,
                                  self.mInputUsesPacketDescriptions ? self.mPacketDescriptions.data() : NULL,
                                  *ioNumberDataPackets, ioData->mBuffers[0].mData);
    } catch (AudioToolboxError err) {
      std::cerr << "Encountered an error while reading packets: " << err.what() << std::endl;
      return err.status;
    }
    return noErr;
  }

 private:
  AudioFile mInputFile;
  AudioStreamBasicDescription mInputDescription;
  std::vector<uint8_t> mInputBuffer;
  std::vector<AudioStreamPacketDescription> mPacketDescriptions;
  const UInt32 mMaxInputPacketSize;
  const bool mInputUsesPacketDescriptions;
};

OSStatus AudioFileReadProc(void* inClientData, SInt64 inPosition, UInt32 requestCount, void* outBuffer,
                           UInt32* actualCount) {
  AudioFile* f = (AudioFile*)inClientData;
  auto copyCount = requestCount;
  if (inPosition + requestCount > f->data_size) {
    copyCount = f->data_size - inPosition;
  }
  std::copy((uint8_t*)f->data_ptr + inPosition, (uint8_t*)f->data_ptr + inPosition + copyCount, (uint8_t*)outBuffer);
  *actualCount = copyCount;
  return 0;
}

SInt64 GetSizeProc(void* inClientData) {
  AudioFile* f = (AudioFile*)inClientData;
  return f->data_size;
}

OrtxStatus AudioDecoder::Compute(const ortc::Tensor<uint8_t>& input, const std::optional<std::string> format,
                                 ortc::Tensor<float>& output0) const {
  const uint8_t* p_data = input.Data();
  auto input_dim = input.Shape();
  OrtxStatus status;
  if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
    return {kOrtxErrorInvalidArgument, "[AudioDecoder]: Expect input dimension [n] or [1,n]."};
  }

  AudioFile inputFile(p_data, input.NumberOfElement());
  AudioFileOpenWithCallbacks((void*)&inputFile, AudioFileReadProc, nullptr, GetSizeProc, nullptr, 0,
                             &inputFile.mAudioFileID);

  AudioStreamBasicDescription inputDescription;
  inputFile.GetProperty(kAudioFilePropertyDataFormat, sizeof(inputDescription), &inputDescription);

  const Float64 orig_sample_rate = inputDescription.mSampleRate;
  const uint64_t orig_channels = inputDescription.mChannelsPerFrame;

  AudioStreamBasicDescription outputDescription{.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                                .mFormatID = kAudioFormatLinearPCM,
                                                .mSampleRate = inputDescription.mSampleRate,
                                                .mChannelsPerFrame = inputDescription.mChannelsPerFrame,
                                                .mBytesPerPacket = 4 * inputDescription.mChannelsPerFrame,
                                                .mFramesPerPacket = 1,
                                                .mBytesPerFrame = 4 * inputDescription.mChannelsPerFrame,
                                                .mBitsPerChannel = 32};

  AudioConverterRef audioConverter;
  const OSStatus err = AudioConverterNew(&inputDescription, &outputDescription, &audioConverter);
  UInt32 maxInputPacketSize, maxOutputPacketSize;
  inputFile.GetProperty(kAudioFilePropertyMaximumPacketSize, sizeof(maxInputPacketSize), &maxInputPacketSize);
  maxOutputPacketSize = outputDescription.mBytesPerPacket;

  // Determine whether the input uses packet descriptions.
  const bool inputUsesPacketDescriptions =
      (inputDescription.mBytesPerPacket == 0 || inputDescription.mFramesPerPacket == 0);

  InputContext inputContext(std::move(inputFile), inputDescription, maxInputPacketSize, inputUsesPacketDescriptions);

  // Determine the number of output packets to attempt to produce
  // per loop, based on whether the sample is encoding.
  const UInt32 packetsPerLoop = 10000;

  // Convert audio until the sample runs out of input.
  std::vector<uint8_t> packetBuffer((size_t)(packetsPerLoop * maxOutputPacketSize));
  std::vector<float> buf;

  size_t total_buf_size = 0;
  int64_t offset = 0;

  for (;;) {
    // Try to handle more packets depending on whether the sample is encoding or decoding.
    UInt32 numPackets = packetsPerLoop;
    AudioBufferList abl{1,
                        {
                            outputDescription.mChannelsPerFrame,  // mNumberChannels
                            (UInt32)packetBuffer.size(),          // mDataByteSize
                            packetBuffer.data()                   // mData
                        }};
    AudioConverterFillComplexBuffer(audioConverter, InputContext::InputDataProc, &inputContext, &numPackets, &abl,
                                    NULL);

    // If there are output packets, write them to the output file.
    if (numPackets > 0) {
      total_buf_size += abl.mBuffers[0].mDataByteSize / 4;
      buf.resize(total_buf_size);

      std::copy((float*)abl.mBuffers[0].mData, (float*)abl.mBuffers[0].mData + abl.mBuffers[0].mDataByteSize / 4,
                buf.begin() + offset);
      offset += abl.mBuffers[0].mDataByteSize / 4;
    }

    // Stop if the sample decodes fewer packets than it requests.
    // This happens when the sample runs out of input.
    if (numPackets < packetsPerLoop) {
      break;
    }
  }

  MixAndDownsampleIfNeeded(buf, orig_channels, orig_sample_rate);

  std::vector<int64_t> dim_out = {1, ort_extensions::narrow<int64_t>(buf.size())};
  float* p_output = output0.Allocate(dim_out);
  std::copy(buf.begin(), buf.end(), p_output);
  return status;

  return {};
}