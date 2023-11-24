# -*- coding: utf-8 -*-
import numpy
from math import ceil

def bitarray_to_bytearray(xbit, xsize):
    """Converts a bits array to a bytes array.

    Parameters:
        xbit: int
            Input bitarray to be converted to a bytearray.
        xsize: list of int
            Size of the original bytes array dimensions.

    Returns:
        numpy array
            Bytes array obtained by converting the input bits array.

    Notes:
        The function performs a bitwise operation to convert the input bits array to a bytes array. The resulting bytes array
        is reshaped based on the size of the input original bytes array dimensions.
    """


def bitarray_to_bytearray(xbit, xsize):
    ns = len(xsize)
    nx = xsize[-1]
    nx8 = numpy.ceil(nx / 8).astype(numpy.int64) * 8
    xbyte = numpy.reshape(
                numpy.transpose(
                        numpy.reshape(
                                     [(xbit & 0b10000000) / 128, (xbit & 0b01000000) / 64,
                                     (xbit & 0b00100000) / 32, (xbit & 0b00010000) / 16,
                                     (xbit & 0b00001000) / 8, (xbit & 0b00000100) / 4,
                                     (xbit & 0b00000010) / 2, (xbit & 0b00000001)],
                                     (nx8 // 8, 8), order='F'
                                     )
                                ),
                            nx8)
    ndim = xsize[0]
    if ndim == 1:
        xbyte = numpy.reshape(xbyte[:nx], xsize[1], order='F')
    elif ndim == 2:
        xbyte = numpy.reshape(xbyte[:nx], (xsize[2], xsize[1]), order='F')
    elif ndim == 3:
        xbyte = numpy.reshape(xbyte, (xsize[1], xsize[2], xsize[3]), order='F')
    elif ndim == 4:
        xbyte = numpy.reshape(xbyte[:nx], (xsize[4], xsize[3], xsize[2], xsize[1]), order='F')

    return xbyte
