import logging
import math
import os
import time
from scipy.fftpack import idct
from bitarray import bitarray, bits2bytes
import numpy as np
from decode import *



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


def ycbcr2rgb(y, cb, cr):
  

    return collections.OrderedDict((
        (R, y + 1.402 * cr),
        (G, y - 0.344136 * cb - 0.714136 * cr),
        (B, y + 1.772 * cb)
    ))



def upsample(arr, mode):
   
    if mode not in {1, 2, 4}:
        raise ValueError('Mode ({}) must be 1, 2 or 4.'.format(mode))

    if mode == 4:
        return arr
    return arr.repeat(3 - mode, axis=0).repeat(2, axis=1)



def quantize(block, block_type, quality=50, inverse=False):
    if block_type == Y:
        quantization_table = LUMINANCE_QUANTIZATION_TABLE
    else:  # Cb or Cr (LUMINANCE)
        quantization_table = CHROMINANCE_QUANTIZATION_TABLE
    factor = 5000 / quality if quality < 50 else 200 - 2 * quality
    if inverse:
        return block * (quantization_table * factor / 100)
    return block / (quantization_table * factor / 100)


def idct2d(arr):
    return idct(idct(arr, norm='ortho', axis=0), norm='ortho', axis=1)

def block_combine(arr, nrows, ncols):
   
    if arr.size != nrows * ncols:
        raise ValueError('The size of arr ({}) should be equal to nrows * ncols ({} * {})').format(arr.size,nrows,ncols)

    _, block_nrows, block_ncols = arr.shape

    return (arr.reshape(nrows // block_nrows, -1, block_nrows, block_ncols)
            .swapaxes(1, 2)
            .reshape(nrows, ncols))



def extract(f, header):
    def school_round(val):
        if float(val) % 1 >= 0.5:
            return math.ceil(val)
        return round(val)

    start_time = time.perf_counter()
    
    print('after file size:  ',os.fstat(f.fileno()).st_size, ' Bytes')
    
    # logging.getLogger(__name__).info('Compressed file size:{} Bytes'.format(os.fstat(f.fileno()).st_size))

    bits = bitarray()
    bits.fromfile(f)
    bits = bits.to01()

    # Read Header
    size = header['size']
    grey_level = header['grey_level']
    quality = header['quality']
    subsampling_mode = header['subsampling_mode']
    remaining_bits_length = header['remaining_bits_length']
    dsls = header['data_slice_lengths']  # data_slice_lengths

    # Calculate the size after subsampling.
    if subsampling_mode == 4:
        subsampled_size = size
    else:
        subsampled_size = (
            size[0] if subsampling_mode == 2 else school_round(size[0] / 2),
            school_round(size[1] / 2)
        )

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure for `Decoder`.

    if remaining_bits_length:
        bits = bits[:-remaining_bits_length]

    if grey_level:
        # The order of dsls (grey level) is:
        #   DC, AC
        sliced = {
            DC: bits[:dsls[0]],
            AC: bits[dsls[0]:]
        }
    else:  # RGB
        # The order of dsls (RGB) is:
        #   LUMINANCE.DC, LUMINANCE.AC, CHROMINANCE.DC, CHROMINANCE.AC
        sliced = {
            LUMINANCE: {
                DC: bits[:dsls[0]],
                AC: bits[dsls[0]:dsls[0] + dsls[1]]
            },
            CHROMINANCE: {
                DC: bits[dsls[0] + dsls[1]:dsls[0] + dsls[1] + dsls[2]],
                AC: bits[dsls[0] + dsls[1] + dsls[2]:]
            }
        }

    # Huffman Decoding
    if grey_level:
        data = {Y: Decoder(sliced, LUMINANCE).decode()}
    else:
        cb, cr = np.split(Decoder(
            sliced[CHROMINANCE],
            CHROMINANCE
        ).decode(), 2)
        data = {
            Y: Decoder(sliced[LUMINANCE], LUMINANCE).decode(),
            CB: cb,
            CR: cr
        }

    for key, layer in data.items():
        for idx, block in enumerate(layer):
            # Inverse Quantization
            layer[idx] = quantize(
                block,
                key,
                quality=quality,
                inverse=True
            )

            # 2D IDCT.
            layer[idx] = idct2d(layer[idx])

        # Calculate the size after subsampling and padding.
        if key == Y:
            padded_size = ((s // 8 + 1) * 8 if s % 8 else s for s in size)
        else:
            padded_size = ((s // 8 + 1) * 8 if s % 8 else s
                           for s in subsampled_size)
        # Combine the blocks into original image
        data[key] = block_combine(layer, *padded_size)

    # Inverse Level Offset
    data[Y] = data[Y] + 128

    # Clip Padded Image
    data[Y] = data[Y][:size[0], :size[1]]
    if not grey_level:
        data[CB] = data[CB][:subsampled_size[0], :subsampled_size[1]]
        data[CR] = data[CR][:subsampled_size[0], :subsampled_size[1]]

        # Upsampling and Clipping
        data[CB] = upsample(data[CB], subsampling_mode)[:size[0], :size[1]]
        data[CR] = upsample(data[CR], subsampling_mode)[:size[0], :size[1]]

        # Color Space Conversion
        data = ycbcr2rgb(**data)

    # Rounding, Clipping and Flatten
    for k, v in data.items():
        data[k] = np.rint(np.clip(v, 0, 255)).flatten()

    # logging.getLogger(__name__).info(
        # 'Time elapsed: %.4f seconds' % (time.perf_counter() - start_time)
    # )
    # Combine layers into signle raw data.
    return (np.dstack((data.values()))
            .flatten()
            .astype(np.uint8))
