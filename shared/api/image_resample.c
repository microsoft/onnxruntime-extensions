// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The major implementation is based on the Python Imaging Library (PIL) by Fredrik Lundh and Contributors.
// https://github.com/python-pillow/Pillow/blob/73dfe67736565805f4c769cdfc92447f2b098187/src/libImaging/Resample.c
/*
 * The Python Imaging Library
 * $Id$
 *
 * declarations for the imaging core library
 *
 * Copyright (c) 1997-2005 by Secret Labs AB
 * Copyright (c) 1995-2005 by Fredrik Lundh
 *
 * See the README file for information on usage and redistribution.
 * https://github.com/python-pillow/Pillow/blob/73dfe67736565805f4c769cdfc92447f2b098187/README.md
 */

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "image_resample.h"


#if defined(_WIN32) || defined(__CYGWIN__) /* WIN */

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAKE_UINT32(u0, u1, u2, u3) \
    ((uint32_t)(u0) | ((uint32_t)(u1) << 8) | ((uint32_t)(u2) << 16) | ((uint32_t)(u3) << 24))


#define IMAGING_MODE_LENGTH \
    6 + 1 /* Band names ("1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "BGR;xy") */

#define IMAGING_TYPE_UINT8 0
#define IMAGING_TYPE_INT32 1
#define IMAGING_TYPE_FLOAT32 2
#define IMAGING_TYPE_SPECIAL 3 /* check mode for details */


/* standard transforms */
#define IMAGING_TRANSFORM_AFFINE 0
#define IMAGING_TRANSFORM_PERSPECTIVE 2
#define IMAGING_TRANSFORM_QUAD 3

#define IMAGING_PIXEL_1(im, x, y) ((im)->image8[(y)][(x)])
#define IMAGING_PIXEL_L(im, x, y) ((im)->image8[(y)][(x)])
#define IMAGING_PIXEL_LA(im, x, y) ((im)->image[(y)][(x) * 4])
#define IMAGING_PIXEL_I(im, x, y) ((im)->image32[(y)][(x)])
#define IMAGING_PIXEL_F(im, x, y) (((float *)(im)->image32[y])[x])
#define IMAGING_PIXEL_RGB(im, x, y) ((im)->image[(y)][(x) * 4])
#define IMAGING_PIXEL_RGBA(im, x, y) ((im)->image[(y)][(x) * 4])
#define IMAGING_PIXEL_CMYK(im, x, y) ((im)->image[(y)][(x) * 4])
#define IMAGING_PIXEL_YCbCr(im, x, y) ((im)->image[(y)][(x) * 4])

#define IMAGING_PIXEL_UINT8(im, x, y) ((im)->image8[(y)][(x)])
#define IMAGING_PIXEL_INT32(im, x, y) ((im)->image32[(y)][(x)])
#define IMAGING_PIXEL_FLOAT32(im, x, y) (((float *)(im)->image32[y])[x])

typedef struct ImagingMemoryArenaInstance {
    int alignment;     /* Alignment in memory of each line of an image */
    int block_size;    /* Preferred block size, bytes */
    int blocks_max;    /* Maximum number of cached blocks */
    int blocks_cached; /* Current number of blocks not associated with images */
    ImagingMemoryBlock *blocks_pool;
    int stats_new_count;        /* Number of new allocated images */
    int stats_allocated_blocks; /* Number of allocated blocks */
    int stats_reused_blocks;    /* Number of blocks which were retrieved from a pool */
    int stats_reallocated_blocks; /* Number of blocks which were actually reallocated
                                     after retrieving */
    int stats_freed_blocks;       /* Number of freed blocks */
} *ImagingMemoryArena;

/* Array Storage Type */
/* ------------------ */
/* Allocate image as an array of line buffers. */

#define IMAGING_PAGE_SIZE (4096)

struct ImagingMemoryArenaInstance ImagingDefaultArena = {
    1,                 // alignment
    16 * 1024 * 1024,  // block_size
    0,                 // blocks_max
    0,                 // blocks_cached
    NULL,              // blocks_pool
    0,
    0,
    0,
    0,
    0  // Stats
};


extern void *
ImagingError_MemoryError(void)  {
    fprintf(stderr, "*** exception: out of memory\n");
    return NULL;
}

extern void *
ImagingError_ValueError(const char *message){
    if (!message) {
        message = "exception: bad argument to function";
    }
    fprintf(stderr, "*** %s\n", message);
    return NULL;
}

extern void *
ImagingError_ModeError(void){
    return ImagingError_ValueError("bad image mode");
}

Imaging
ImagingNewPrologueSubtype(const char *mode, int xsize, int ysize, int size) {
    Imaging im;

    /* linesize overflow check, roughly the current largest space req'd */
    if (xsize > (INT_MAX / 4) - 1) {
        return (Imaging)ImagingError_MemoryError();
    }

    im = (Imaging)calloc(1, size);
    if (!im) {
        return (Imaging)ImagingError_MemoryError();
    }

    /* Setup image descriptor */
    im->xsize = xsize;
    im->ysize = ysize;

    im->type = IMAGING_TYPE_UINT8;

    if (strcmp(mode, "1") == 0) {
        /* 1-bit images */
        im->bands = im->pixelsize = 1;
        im->linesize = xsize;
    } else if (strcmp(mode, "L") == 0) {
        /* 8-bit grayscale (luminance) images */
        im->bands = im->pixelsize = 1;
        im->linesize = xsize;
    } else if (strcmp(mode, "LA") == 0) {
        /* 8-bit grayscale (luminance) with alpha */
        im->bands = 2;
        im->pixelsize = 4; /* store in image32 memory */
        im->linesize = xsize * 4;
    } else if (strcmp(mode, "La") == 0) {
        /* 8-bit grayscale (luminance) with premultiplied alpha */
        im->bands = 2;
        im->pixelsize = 4; /* store in image32 memory */
        im->linesize = xsize * 4;
    } else if (strcmp(mode, "F") == 0) {
        /* 32-bit floating point images */
        im->bands = 1;
        im->pixelsize = 4;
        im->linesize = xsize * 4;
        im->type = IMAGING_TYPE_FLOAT32;
    } else if (strcmp(mode, "I") == 0) {
        /* 32-bit integer images */
        im->bands = 1;
        im->pixelsize = 4;
        im->linesize = xsize * 4;
        im->type = IMAGING_TYPE_INT32;
    } else if (
        strcmp(mode, "I;16") == 0 || strcmp(mode, "I;16L") == 0 ||
        strcmp(mode, "I;16B") == 0 || strcmp(mode, "I;16N") == 0) {
        /* EXPERIMENTAL */
        /* 16-bit raw integer images */
        im->bands = 1;
        im->pixelsize = 2;
        im->linesize = xsize * 2;
        im->type = IMAGING_TYPE_SPECIAL;

    } else if (strcmp(mode, "RGB") == 0) {
        /* 24-bit true colour images */
        im->bands = 3;
        im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "BGR;15") == 0) {
        /* EXPERIMENTAL */
        /* 15-bit reversed true colour */
        im->bands = 3;
        im->pixelsize = 2;
        im->linesize = (xsize * 2 + 3) & -4;
        im->type = IMAGING_TYPE_SPECIAL;

    } else if (strcmp(mode, "BGR;16") == 0) {
        /* EXPERIMENTAL */
        /* 16-bit reversed true colour */
        im->bands = 3;
        im->pixelsize = 2;
        im->linesize = (xsize * 2 + 3) & -4;
        im->type = IMAGING_TYPE_SPECIAL;

    } else if (strcmp(mode, "BGR;24") == 0) {
        /* EXPERIMENTAL */
        /* 24-bit reversed true colour */
        im->bands = 3;
        im->pixelsize = 3;
        im->linesize = (xsize * 3 + 3) & -4;
        im->type = IMAGING_TYPE_SPECIAL;

    } else if (strcmp(mode, "RGBX") == 0) {
        /* 32-bit true colour images with padding */
        im->bands = im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "RGBA") == 0) {
        /* 32-bit true colour images with alpha */
        im->bands = im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "RGBa") == 0) {
        /* 32-bit true colour images with premultiplied alpha */
        im->bands = im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "CMYK") == 0) {
        /* 32-bit colour separation */
        im->bands = im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "YCbCr") == 0) {
        /* 24-bit video format */
        im->bands = 3;
        im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "LAB") == 0) {
        /* 24-bit color, luminance, + 2 color channels */
        /* L is uint8, a,b are int8 */
        im->bands = 3;
        im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else if (strcmp(mode, "HSV") == 0) {
        /* 24-bit color, luminance, + 2 color channels */
        /* L is uint8, a,b are int8 */
        im->bands = 3;
        im->pixelsize = 4;
        im->linesize = xsize * 4;

    } else {
        free(im);
        return (Imaging)ImagingError_ValueError("unrecognized image mode");
    }

    /* Setup image descriptor */
    #if defined(_MSC_VER)
    strncpy_s(im->mode, IMAGING_MODE_LENGTH - 1, mode, IMAGING_MODE_LENGTH - 1);
    #else
    strncpy(im->mode, mode, IMAGING_MODE_LENGTH - 1);
    #endif // MSVC

    /* Pointer array (allocate at least one line, to avoid MemoryError
       exceptions on platforms where calloc(0, x) returns NULL) */
    im->image = (char **)calloc((ysize > 0) ? ysize : 1, sizeof(void *));

    if (!im->image) {
        free(im);
        return (Imaging)ImagingError_MemoryError();
    }

    /* Initialize alias pointers to pixel data. */
    switch (im->pixelsize) {
        case 1:
        case 2:
        case 3:
            im->image8 = (uint8_t **)im->image;
            break;
        case 4:
            im->image32 = (int32_t **)im->image;
            break;
    }

    ImagingDefaultArena.stats_new_count += 1;

    return im;
}

Imaging
ImagingNewPrologue(const char *mode, int xsize, int ysize) {
    return ImagingNewPrologueSubtype(
        mode, xsize, ysize, sizeof(struct ImagingMemoryInstance));
}

ImagingMemoryBlock
memory_get_block(ImagingMemoryArena arena, int requested_size, int dirty) {
    ImagingMemoryBlock block = {NULL, 0};

    if (arena->blocks_cached > 0) {
        // Get block from cache
        arena->blocks_cached -= 1;
        block = arena->blocks_pool[arena->blocks_cached];
        // Reallocate if needed
        if (block.size != requested_size) {
            block.ptr = realloc(block.ptr, requested_size);
        }
        if (!block.ptr) {
            // Can't allocate, free previous pointer (it is still valid)
            free(arena->blocks_pool[arena->blocks_cached].ptr);
            arena->stats_freed_blocks += 1;
            return block;
        }
        if (!dirty) {
            memset(block.ptr, 0, requested_size);
        }
        arena->stats_reused_blocks += 1;
        if (block.ptr != arena->blocks_pool[arena->blocks_cached].ptr) {
            arena->stats_reallocated_blocks += 1;
        }
    } else {
        if (dirty) {
            block.ptr = malloc(requested_size);
        } else {
            block.ptr = calloc(1, requested_size);
        }
        arena->stats_allocated_blocks += 1;
    }
    block.size = requested_size;
    return block;
}

void
memory_return_block(ImagingMemoryArena arena, ImagingMemoryBlock block) {
    if (arena->blocks_cached < arena->blocks_max) {
        // Reduce block size
        if (block.size > arena->block_size) {
            block.size = arena->block_size;
            block.ptr = realloc(block.ptr, arena->block_size);
        }
        arena->blocks_pool[arena->blocks_cached] = block;
        arena->blocks_cached += 1;
    } else {
        free(block.ptr);
        arena->stats_freed_blocks += 1;
    }
}

static void
ImagingDestroyArray(Imaging im) {
    int y = 0;

    if (im->blocks) {
        while (im->blocks[y].ptr) {
            memory_return_block(&ImagingDefaultArena, im->blocks[y]);
            y += 1;
        }
        free(im->blocks);
    }
}

Imaging
ImagingAllocateArray(Imaging im, int dirty, int block_size) {
    int y, line_in_block, current_block;
    ImagingMemoryArena arena = &ImagingDefaultArena;
    ImagingMemoryBlock block = {NULL, 0};
    int aligned_linesize, lines_per_block, blocks_count;
    char *aligned_ptr = NULL;

    /* 0-width or 0-height image. No need to do anything */
    if (!im->linesize || !im->ysize) {
        return im;
    }

    aligned_linesize = (im->linesize + arena->alignment - 1) & -arena->alignment;
    lines_per_block = (block_size - (arena->alignment - 1)) / aligned_linesize;
    if (lines_per_block == 0) {
        lines_per_block = 1;
    }
    blocks_count = (im->ysize + lines_per_block - 1) / lines_per_block;
    // printf("NEW size: %dx%d, ls: %d, lpb: %d, blocks: %d\n",
    //        im->xsize, im->ysize, aligned_linesize, lines_per_block, blocks_count);

    /* One extra pointer is always NULL */
    im->blocks = calloc(sizeof(*im->blocks), blocks_count + 1);
    if (!im->blocks) {
        return (Imaging)ImagingError_MemoryError();
    }

    /* Allocate image as an array of lines */
    line_in_block = 0;
    current_block = 0;
    for (y = 0; y < im->ysize; y++) {
        if (line_in_block == 0) {
            int required;
            int lines_remaining = lines_per_block;
            if (lines_remaining > im->ysize - y) {
                lines_remaining = im->ysize - y;
            }
            required = lines_remaining * aligned_linesize + arena->alignment - 1;
            block = memory_get_block(arena, required, dirty);
            if (!block.ptr) {
                ImagingDestroyArray(im);
                return (Imaging)ImagingError_MemoryError();
            }
            im->blocks[current_block] = block;
            /* Bulletproof code from libc _int_memalign */
            aligned_ptr = (char *)(((size_t)(block.ptr + arena->alignment - 1)) &
                                   -((ptrdiff_t)arena->alignment));
        }

        im->image[y] = aligned_ptr + aligned_linesize * line_in_block;

        line_in_block += 1;
        if (line_in_block >= lines_per_block) {
            /* Reset counter and start new block */
            line_in_block = 0;
            current_block += 1;
        }
    }

    im->destroy = ImagingDestroyArray;

    return im;
}

void
ImagingDelete(Imaging im) {
    if (!im) {
        return;
    }

    if (im->destroy) {
        im->destroy(im);
    }

    if (im->image) {
        free(im->image);
    }

    free(im);
}

Imaging
ImagingNewInternal(const char *mode, int xsize, int ysize, int dirty) {
    Imaging im;

    if (xsize < 0 || ysize < 0) {
        return (Imaging)ImagingError_ValueError("bad image size");
    }

    im = ImagingNewPrologue(mode, xsize, ysize);
    if (!im) {
        return NULL;
    }

    if (ImagingAllocateArray(im, dirty, ImagingDefaultArena.block_size)) {
        return im;
    }

    // Try to allocate the image once more with smallest possible block size
    if (ImagingAllocateArray(im, dirty, IMAGING_PAGE_SIZE)) {
        return im;
    }

    ImagingDelete(im);
    return NULL;
}

Imaging
ImagingNew(const char *mode, int xsize, int ysize) {
    return ImagingNewInternal(mode, xsize, ysize, 0);
}

Imaging
ImagingNewDirty(const char *mode, int xsize, int ysize) {
    return ImagingNewInternal(mode, xsize, ysize, 1);
}

extern Imaging
ImagingCopy(Imaging imIn){
    int y;

    Imaging imOut = ImagingNewDirty(imIn->mode, imIn->xsize, imIn->ysize);


    if (!imIn) {
        return (Imaging)ImagingError_ValueError(NULL);
    }


    if (imIn->block != NULL && imOut->block != NULL) {
        memcpy(imOut->block, imIn->block, imIn->ysize * imIn->linesize);
    } else {
        for (y = 0; y < imIn->ysize; y++) {
            memcpy(imOut->image[y], imIn->image[y], imIn->linesize);
        }
    }

    return imOut;  
}


#define ROUND_UP(f) ((int)((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))

struct filter {
    double (*filter)(double x);
    double support;
};

static inline double
box_filter(double x) {
    if (x > -0.5 && x <= 0.5) {
        return 1.0;
    }
    return 0.0;
}

static inline double
bilinear_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

static inline double
hamming_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x >= 1.0) {
        return 0.0;
    }
    x = x * M_PI;
    return sin(x) / x * (0.54f + 0.46f * cos(x));
}

static inline double
bicubic_filter(double x) {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
#define a -0.5
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
#undef a
}

static inline double
sinc_filter(double x) {
    if (x == 0.0) {
        return 1.0;
    }
    x = x * M_PI;
    return sin(x) / x;
}

static inline double
lanczos_filter(double x) {
    /* truncated sinc */
    if (-3.0 <= x && x < 3.0) {
        return sinc_filter(x) * sinc_filter(x / 3);
    }
    return 0.0;
}

static struct filter BOX = {box_filter, 0.5};
static struct filter BILINEAR = {bilinear_filter, 1.0};
static struct filter HAMMING = {hamming_filter, 1.0};
static struct filter BICUBIC = {bicubic_filter, 2.0};
static struct filter LANCZOS = {lanczos_filter, 3.0};

/* 8 bits for result. Filter can have negative areas.
   In one cases the sum of the coefficients will be negative,
   in the other it will be more than 1.0. That is why we need
   two extra bits for overflow and int type. */
#define PRECISION_BITS (32 - 8 - 2)

/* Handles values form -640 to 639. */
uint8_t _clip8_lookups[1280] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,   4,   5,
    6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
    57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
    91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
};

uint8_t *clip8_lookups = &_clip8_lookups[640];

static inline uint8_t
clip8(int in) {
    return clip8_lookups[in >> PRECISION_BITS];
}

int
precompute_coeffs(
    int inSize,
    float in0,
    float in1,
    int outSize,
    struct filter *filterp,
    int **boundsp,
    double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    // check for overflow
    if (outSize > INT_MAX / (ksize * (int)sizeof(double))) {
        ImagingError_MemoryError();
        return 0;
    }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = malloc(outSize * ksize * sizeof(double));
    if (!kk) {
        ImagingError_MemoryError();
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = malloc(outSize * 2 * sizeof(int));
    if (!bounds) {
        free(kk);
        ImagingError_MemoryError();
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}

void
normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
    int x;
    int32_t *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t *)prekk;

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = (int)(-0.5 + prekk[x] * (1 << PRECISION_BITS));
        } else {
            kk[x] = (int)(0.5 + prekk[x] * (1 << PRECISION_BITS));
        }
    }
}

void
ImagingResampleHorizontal_8bpc(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *prekk) {
    int ss0, ss1, ss2, ss3;
    int xx, yy, x, xmin, xmax;
    int32_t *k, *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t *)prekk;
    normalize_coeffs_8bpc(imOut->xsize, ksize, prekk);

    if (imIn->image8) {
        for (yy = 0; yy < imOut->ysize; yy++) {
            for (xx = 0; xx < imOut->xsize; xx++) {
                xmin = bounds[xx * 2 + 0];
                xmax = bounds[xx * 2 + 1];
                k = &kk[xx * ksize];
                ss0 = 1 << (PRECISION_BITS - 1);
                for (x = 0; x < xmax; x++) {
                    ss0 += ((uint8_t)imIn->image8[yy + offset][x + xmin]) * k[x];
                }
                imOut->image8[yy][xx] = clip8(ss0);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->bands == 2) {
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss0 = ss3 = 1 << (PRECISION_BITS - 1);
                    for (x = 0; x < xmax; x++) {
                        ss0 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 0]) *
                               k[x];
                        ss3 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 3]) *
                               k[x];
                    }
                    v = MAKE_UINT32(clip8(ss0), 0, 0, clip8(ss3));
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        } else if (imIn->bands == 3) {
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss0 = ss1 = ss2 = 1 << (PRECISION_BITS - 1);
                    for (x = 0; x < xmax; x++) {
                        ss0 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 0]) *
                               k[x];
                        ss1 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 1]) *
                               k[x];
                        ss2 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 2]) *
                               k[x];
                    }
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        } else {
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss0 = ss1 = ss2 = ss3 = 1 << (PRECISION_BITS - 1);
                    for (x = 0; x < xmax; x++) {
                        ss0 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 0]) *
                               k[x];
                        ss1 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 1]) *
                               k[x];
                        ss2 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 2]) *
                               k[x];
                        ss3 += ((uint8_t)imIn->image[yy + offset][(x + xmin) * 4 + 3]) *
                               k[x];
                    }
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), clip8(ss3));
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        }
    }
}

void
ImagingResampleVertical_8bpc(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *prekk) {
    int ss0, ss1, ss2, ss3;
    int xx, yy, y, ymin, ymax;
    int32_t *k, *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t *)prekk;
    normalize_coeffs_8bpc(imOut->ysize, ksize, prekk);

    if (imIn->image8) {
        for (yy = 0; yy < imOut->ysize; yy++) {
            k = &kk[yy * ksize];
            ymin = bounds[yy * 2 + 0];
            ymax = bounds[yy * 2 + 1];
            for (xx = 0; xx < imOut->xsize; xx++) {
                ss0 = 1 << (PRECISION_BITS - 1);
                for (y = 0; y < ymax; y++) {
                    ss0 += ((uint8_t)imIn->image8[y + ymin][xx]) * k[y];
                }
                imOut->image8[yy][xx] = clip8(ss0);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->bands == 2) {
            for (yy = 0; yy < imOut->ysize; yy++) {
                k = &kk[yy * ksize];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    ss0 = ss3 = 1 << (PRECISION_BITS - 1);
                    for (y = 0; y < ymax; y++) {
                        ss0 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 0]) * k[y];
                        ss3 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 3]) * k[y];
                    }
                    v = MAKE_UINT32(clip8(ss0), 0, 0, clip8(ss3));
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        } else if (imIn->bands == 3) {
            for (yy = 0; yy < imOut->ysize; yy++) {
                k = &kk[yy * ksize];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    ss0 = ss1 = ss2 = 1 << (PRECISION_BITS - 1);
                    for (y = 0; y < ymax; y++) {
                        ss0 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 0]) * k[y];
                        ss1 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 1]) * k[y];
                        ss2 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 2]) * k[y];
                    }
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        } else {
            for (yy = 0; yy < imOut->ysize; yy++) {
                k = &kk[yy * ksize];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    uint32_t v;
                    ss0 = ss1 = ss2 = ss3 = 1 << (PRECISION_BITS - 1);
                    for (y = 0; y < ymax; y++) {
                        ss0 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 0]) * k[y];
                        ss1 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 1]) * k[y];
                        ss2 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 2]) * k[y];
                        ss3 += ((uint8_t)imIn->image[y + ymin][xx * 4 + 3]) * k[y];
                    }
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), clip8(ss3));
                    memcpy(imOut->image[yy] + xx * sizeof(v), &v, sizeof(v));
                }
            }
        }
    }
}

void
ImagingResampleHorizontal_32bpc(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *kk) {
    double ss;
    int xx, yy, x, xmin, xmax;
    double *k;

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < imOut->xsize; xx++) {
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++) {
                        ss += IMAGING_PIXEL_I(imIn, x + xmin, yy + offset) * k[x];
                    }
                    IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < imOut->xsize; xx++) {
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++) {
                        ss += IMAGING_PIXEL_F(imIn, x + xmin, yy + offset) * k[x];
                    }
                    IMAGING_PIXEL_F(imOut, xx, yy) = (float)ss;
                }
            }
            break;
    }
}

void
ImagingResampleVertical_32bpc(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *kk) {
    double ss;
    int xx, yy, y, ymin, ymax;
    double *k;

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                k = &kk[yy * ksize];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++) {
                        ss += IMAGING_PIXEL_I(imIn, xx, y + ymin) * k[y];
                    }
                    IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                k = &kk[yy * ksize];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++) {
                        ss += IMAGING_PIXEL_F(imIn, xx, y + ymin) * k[y];
                    }
                    IMAGING_PIXEL_F(imOut, xx, yy) = (float)ss;
                }
            }
            break;
    }
}

typedef void (*ResampleFunction)(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *kk);

Imaging
ImagingResampleInner(
    Imaging imIn,
    int xsize,
    int ysize,
    struct filter *filterp,
    float box[4],
    ResampleFunction ResampleHorizontal,
    ResampleFunction ResampleVertical);

Imaging
ImagingResample(Imaging imIn, int xsize, int ysize, int filter, float box[4]) {
    struct filter *filterp;
    ResampleFunction ResampleHorizontal;
    ResampleFunction ResampleVertical;

    if (strcmp(imIn->mode, "P") == 0 || strcmp(imIn->mode, "1") == 0) {
        return (Imaging)ImagingError_ModeError();
    }

    if (imIn->type == IMAGING_TYPE_SPECIAL) {
        return (Imaging)ImagingError_ModeError();
    } else if (imIn->image8) {
        ResampleHorizontal = ImagingResampleHorizontal_8bpc;
        ResampleVertical = ImagingResampleVertical_8bpc;
    } else {
        switch (imIn->type) {
            case IMAGING_TYPE_UINT8:
                ResampleHorizontal = ImagingResampleHorizontal_8bpc;
                ResampleVertical = ImagingResampleVertical_8bpc;
                break;
            case IMAGING_TYPE_INT32:
            case IMAGING_TYPE_FLOAT32:
                ResampleHorizontal = ImagingResampleHorizontal_32bpc;
                ResampleVertical = ImagingResampleVertical_32bpc;
                break;
            default:
                return (Imaging)ImagingError_ModeError();
        }
    }

    /* check filter */
    switch (filter) {
        case IMAGING_TRANSFORM_BOX:
            filterp = &BOX;
            break;
        case IMAGING_TRANSFORM_BILINEAR:
            filterp = &BILINEAR;
            break;
        case IMAGING_TRANSFORM_HAMMING:
            filterp = &HAMMING;
            break;
        case IMAGING_TRANSFORM_BICUBIC:
            filterp = &BICUBIC;
            break;
        case IMAGING_TRANSFORM_LANCZOS:
            filterp = &LANCZOS;
            break;
        default:
            return (Imaging)ImagingError_ValueError("unsupported resampling filter");
    }

    return ImagingResampleInner(
        imIn, xsize, ysize, filterp, box, ResampleHorizontal, ResampleVertical);
}

Imaging
ImagingResampleInner(
    Imaging imIn,
    int xsize,
    int ysize,
    struct filter *filterp,
    float box[4],
    ResampleFunction ResampleHorizontal,
    ResampleFunction ResampleVertical) {
    Imaging imTemp = NULL;
    Imaging imOut = NULL;

    int i, need_horizontal, need_vertical;
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    need_horizontal = xsize != imIn->xsize || box[0] || box[2] != xsize;
    need_vertical = ysize != imIn->ysize || box[1] || box[3] != ysize;

    ksize_horiz = precompute_coeffs(
        imIn->xsize, box[0], box[2], xsize, filterp, &bounds_horiz, &kk_horiz);
    if (!ksize_horiz) {
        return NULL;
    }

    ksize_vert = precompute_coeffs(
        imIn->ysize, box[1], box[3], ysize, filterp, &bounds_vert, &kk_vert);
    if (!ksize_vert) {
        free(bounds_horiz);
        free(kk_horiz);
        return NULL;
    }

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1];

    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        // Shift bounds for vertical pass
        for (i = 0; i < ysize; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }

        imTemp = ImagingNewDirty(imIn->mode, xsize, ybox_last - ybox_first);
        if (imTemp) {
            ResampleHorizontal(
                imTemp, imIn, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
        }
        free(bounds_horiz);
        free(kk_horiz);
        if (!imTemp) {
            free(bounds_vert);
            free(kk_vert);
            return NULL;
        }
        imOut = imIn = imTemp;
    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }

    /* vertical pass */
    if (need_vertical) {
        imOut = ImagingNewDirty(imIn->mode, imIn->xsize, ysize);
        if (imOut) {
            /* imIn can be the original image or horizontally resampled one */
            ResampleVertical(imOut, imIn, 0, ksize_vert, bounds_vert, kk_vert);
        }
        /* it's safe to call ImagingDelete with empty value
           if previous step was not performed. */
        ImagingDelete(imTemp);
        free(bounds_vert);
        free(kk_vert);
        if (!imOut) {
            return NULL;
        }
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    /* none of the previous steps are performed, copying */
    if (!imOut) {
        imOut = ImagingCopy(imIn);
    }

    return imOut;
}
