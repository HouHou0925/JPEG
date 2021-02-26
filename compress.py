import logging
import math
import os
import time
from scipy.fftpack import dct
from bitarray import bitarray, bits2bytes
import numpy as np
from encode import *


R, G, B = 'r', 'g', 'b'
Y, CB, CR = 'y', 'cb', 'cr'


EOB = (0, 0)
ZRL = (15, 0)
DC = 'DC'
AC = 'AC'
LUMINANCE = frozenset({Y})
CHROMINANCE = frozenset({CB, CR})


LUMINANCE_QUANTIZATION_TABLE = np.array((
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 36, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99)
))

CHROMINANCE_QUANTIZATION_TABLE = np.array((
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99)
))

def rgb2ycbcr(r, g, b):
  
    return collections.OrderedDict((
        (Y, + 0.299 * r + 0.587 * g + 0.114 * b),
        (CB, - 0.168736 * r - 0.331264 * g + 0.5 * b),
        (CR, + 0.5 * r - 0.418688 * g - 0.081312 * b)
    ))


def downsample(arr, mode):
  
    if mode not in {1, 2, 4}:
        raise ValueError('Mode ({}) must be 1, 2 or 4.'.format(mode))

    if mode == 4:
        return arr
    return arr[::3 - mode, ::2]


def block_slice(arr, nrows, ncols):
    
    h, _ = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))



def dct2d(arr):
    return dct(dct(arr, norm='ortho', axis=0), norm='ortho', axis=1)



def quantize(block, block_type, quality=50, inverse=False):
    if block_type == Y:
        quantization_table = LUMINANCE_QUANTIZATION_TABLE
    else:  # Cb or Cr (LUMINANCE)
        quantization_table = CHROMINANCE_QUANTIZATION_TABLE
    factor = 5000 / quality if quality < 50 else 200 - 2 * quality
    if inverse:
        return block * (quantization_table * factor / 100)
    return block / (quantization_table * factor / 100)


def comp(f, size, quality=50, grey_level=False, subsampling_mode=1):
    start_time = time.perf_counter()
    
    print('before file size: ' ,os.fstat(f.fileno()).st_size , ' Bytes' )
    
    # logging.getLogger(__name__).info("Original file size:{} Bytes".format(os.fstat(f.fileno()).st_size))

    if quality <= 0 or quality > 95:
        raise ValueError('Quality should within (0, 95].')

    img_arr = np.fromfile(f, dtype=np.uint8).reshape(
        size if grey_level else (*size, 3)
    )

    if grey_level:
        data = {Y: img_arr.astype(float)}

    else:  # RGB
        # Color Space Conversion (w/o Level Offset)
        data = rgb2ycbcr(*(img_arr[:, :, idx] for idx in range(3)))

        # Subsampling
        data[CB] = downsample(data[CB], subsampling_mode)
        data[CR] = downsample(data[CR], subsampling_mode)

    # Level Offset
    data[Y] = data[Y] - 128

    for key, layer in data.items():
        nrows, ncols = layer.shape

        # Pad Layers to 8N * 8N
        data[key] = np.pad(
            layer,
            (
                (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
                (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
            ),
            mode='constant'
        )

        # Block Slicing
        data[key] = block_slice(data[key], 8, 8)

        for idx, block in enumerate(data[key]):
            # 2D DCT
            data[key][idx] = dct2d(block)

            # Quantization
            data[key][idx] = quantize(data[key][idx], key, quality=quality)

        # Rounding
        data[key] = np.rint(data[key]).astype(int)

    if grey_level:
        # Entropy Encoder
        encoded = Encoder(data[Y], LUMINANCE).encode()

        # Combine grey level data as binary in the order:
        #   DC, AC
        order = (encoded[DC], encoded[AC])

    else:  # RGB
        # Entropy Encoder
        encoded = {
            LUMINANCE: Encoder(data[Y], LUMINANCE).encode(),
            CHROMINANCE: Encoder(
                np.vstack((data[CB], data[CR])),
                CHROMINANCE
            ).encode()
        }

        # Combine RGB data as binary in the order:
        #   LUMINANCE.DC, LUMINANCE.AC, CHROMINANCE.DC, CHROMINANCE.AC
        order = (encoded[LUMINANCE][DC], encoded[LUMINANCE][AC],
                 encoded[CHROMINANCE][DC], encoded[CHROMINANCE][AC])

    bits = bitarray(''.join(order))


    

    # logging.getLogger(__name__).info(
        # 'Time elapsed: %.4f seconds' % (time.perf_counter() - start_time)
    # )
    return {
        'data': bits,
        'header': {
            'size': size,
            'grey_level': grey_level,
            'quality': quality,
            'subsampling_mode': subsampling_mode,
            # Remaining bits length is the fake filled bits for 8 bits as a
            # byte.
            'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
            'data_slice_lengths': tuple(len(d) for d in order)
        }
    }