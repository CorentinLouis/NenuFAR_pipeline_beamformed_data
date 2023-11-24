# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy

class ReadFits_data:
    """Reads and opens L1 or L1a data FITS files from ES/LT02 NenuFAR program

    Attributes:
        command: numpy array
            Primary header of the FITS file
        param: numpy array
            Parameters header of the FITS file
        variable: numpy array
            Variable header of the FITS file
        data: numpy array
            Array of float with dimensions [nfreq, ntime, nStokes] representing Stokes I, Q, U, V.
        ndata: numpy array
            Array of float with dimensions [nfreq, ntime] representing RFI flagging level 0 (0: all bad pixels, 1: all correct pixels, 0.5: half correct pixels).
        time: astropy Time object
            Time table obtained from the 'timestamp' and 'TimeDelta' information in the FITS file in unix format.
        frequency: astropy Quantity object
            Frequency table with units in MHz.

    Methods:
        __init__(self, filename: str)
            Instantiate a ReadFits_data object from a FITS file.

            Parameters:
                filename (str):
                    Path of the observation to open.
                    Example: "AD_LEO_TRACKING_20230731_084037_0.spectra_a.fits"

    Notes:
        The class reads FITS files with specific extensions and extracts information about the observation, including time, frequency, data, and RFI flagging level 0.
    """

    def __init__(self, filename: str):
        """
        Instantiate a ReadFits_data object from a .fits file.

        Parameters:
            filename (str):
                Path of the observation to open.
        """
        with fits.open(filename) as hdus:
            self.command = hdus[0].data
            self.param = hdus[1].data
            self.variable = hdus[2].data
            nt = self.variable.NT[0]
            nf = self.variable.NF[0]
            ns = self.variable.NS[0]
            
            # Testing if the “data” for each Stokes are not gathered, as it is the case in the L1a fits file...
            datasize=4*nt*nf*ns
            datasizemax=2**31-1
            # Case where Stokes data are gathered
            if datasize <= datasizemax:
                self.data = hdus[3].data
                k=0
            # Case where Stokes data are *not* gathered
            else:
                data = numpy.zeros((ns,nf,nt))
                for k in range(4):
                    data[k,:,:] = hdus[3+k].data
                self.data = data

            self.ndata = hdus[4+k].data
            self.time = Time(hdus[2].data['timestamp'][0], format='unix') + TimeDelta(hdus[5+k].data, format='sec')
            self.frequency = hdus[6+k].data * u.MHz
