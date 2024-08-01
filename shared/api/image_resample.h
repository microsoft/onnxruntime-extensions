// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define IMAGING_MODE_LENGTH 6 + 1 /* Band names ("1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "BGR;xy") */

/* standard filters */
#define IMAGING_TRANSFORM_NEAREST 0
#define IMAGING_TRANSFORM_BOX 4
#define IMAGING_TRANSFORM_BILINEAR 2
#define IMAGING_TRANSFORM_HAMMING 5
#define IMAGING_TRANSFORM_BICUBIC 3
#define IMAGING_TRANSFORM_LANCZOS 1

typedef struct ImagingMemoryInstance* Imaging;
typedef struct {
  char* ptr;
  int size;
} ImagingMemoryBlock;

struct ImagingMemoryInstance {
  /* Format */
  char mode[IMAGING_MODE_LENGTH]; /* Band names ("1", "L", "P", "RGB", "RGBA", "CMYK",
                                     "YCbCr", "BGR;xy") */
  int type;                       /* Data type (IMAGING_TYPE_*) */
  int bands;                      /* Number of bands (1, 2, 3, or 4) */
  int xsize;                      /* Image dimension. */
  int ysize;

  /* Data pointers */
  uint8_t** image8;  /* Set for 8-bit images (pixelsize=1). */
  int32_t** image32; /* Set for 32-bit images (pixelsize=4). */

  /* Internals */
  char** image;               /* Actual raster data. */
  char* block;                /* Set if data is allocated in a single block. */
  ImagingMemoryBlock* blocks; /* Memory blocks for pixel storage */

  int pixelsize; /* Size of a pixel, in bytes (1, 2 or 4) */
  int linesize;  /* Size of a line, in bytes (xsize * pixelsize) */

  /* Virtual methods */
  void (*destroy)(Imaging im);
};

#ifdef __cplusplus
extern "C" {
#endif

Imaging ImagingNew(const char* mode, int xsize, int ysize);
Imaging ImagingResample(Imaging imIn, int xsize, int ysize, int filter, float box[4]);
void ImagingDelete(Imaging im);

#ifdef __cplusplus
}
#endif
