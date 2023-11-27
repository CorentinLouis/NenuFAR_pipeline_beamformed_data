# -*- coding: utf-8 -*-
from astropy.io import fits
import numpy
from bitarray_to_bytearray import bitarray_to_bytearray

class ReadFits_rfimask:
    """Reads and opens L1 or L1a mask FITS files from ES/LT02 NenuFAR program

    Attributes:
        ndatasize: numpy array
            Contains the information about the size of the RFI flags array
        rfimask_level0: numpy array
            RFI mask level 0 obtained from the corresponding data FITS file.
            This contains the flagging of the ratio of bad to good pixels, as calculating during the pre-processing step (L0 to L1 data)
        rfimask_level1: numpy array
            RFI mask level 1. This is based on RFI level 0 and with applying the PATROL algorithm
        rfimask_level2: numpy array
            RFI mask level 2. This is based on RFI level 1 and applying the SUM_THRESHOLD algorithm at 7.5 sigma (medium severity)
        rfimask_level3: numpy array
            RFI mask level 3. This is based on RFI level 1 and applying the SUM_THRESHOLD algorithm at 6 sigma (medium severity)


    Methods:
        __init__(self, filename: str, rfilevel0)
            Instantiate a ReadFits_rfimask object from a FITS file.

            Parameters:
                filename (str):
                    Path of the RFI FITS file to open.
                    Example: "AD_LEO_TRACKING_20230731_084037_0.rfimask_a.fits"
                rfilevel0:
                    RFI mask level 0 obtained from the corresponding data FITS file.
        """

    def __init__(self, filename: str, rfilevel0):
        """
        Instantiate a ReadFits_rfimask object from a .fits file.

        Parameters:
            filename (str):
                Path of the observation to open.
            rfilevel0:
                RFI mask level 0 obtained from the corresponding data FITS file.
        """
        with fits.open(filename) as hdus:
            self.ndatasize = hdus[2].data
            ndatabit = hdus[3].data
    
            # Convert bits array to bytes array
            ndatabyte = bitarray_to_bytearray(ndatabit, self.ndatasize)


            # Perform the final operation
            self.rfimask_level0 = rfilevel0
            self.rfimask_level1 = ndatabyte[:,:, 0]
            self.rfimask_level2 = ndatabyte[:,:, 1]
            self.rfimask_level3 = ndatabyte[:,:, 2]
