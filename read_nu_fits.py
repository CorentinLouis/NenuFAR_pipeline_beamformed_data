import numpy as np
from astropy.io import fits

def read_nu_fits(file, nstokes=0, nodata=False, quiet=False, help=False):
    """
    Read parameters and data from a FITS file.

    Parameters:
    - file (str): The path to the FITS file.
    - nstokes (int): Stokes parameter selection (0 for all, 1 for I, 2 for IV, -2 for XY, 3 for IVL, 4 for IQUV).
    - nodata (bool): If True, only read parameters, not data.
    - quiet (bool): If True, suppress print statements.
    - help (bool): If True, print the function signature and return.

    Returns:
    - Various data arrays and parameters based on the file contents.
    """

    # Print function signature and return if help is True
    if help:
        print(f"\nread_nu_fits, file, nstokes=nstokes, nodata=nodata, quiet=quiet, help=help\n")
        return

    # Set silent flag based on quiet parameter
    silent = 1 if quiet else 0

    # Set default values for parameters
    if nstokes is None:
        nstokes = 0

    # Read header and data from FITS file
    command = str(fits.getheader(file, ext=0, ignore_missing_end=True, memmap=True))
    if not quiet:
        print(command)

    param = fits.getheader(file, ext=1, ignore_missing_end=True, memmap=True)
    if not quiet:
        print(f"\nversion,versiondate: {param['version']} {param['versiondate']}")
        print(f"tmin,tmax,julian: {param['tmin']} {param['tmax']} {param['julian']}")
        print(f"fmin,fmax,exactfreq: {param['fmin']} {param['fmax']} {param['exactfreq']}")
        print(f"beams: {param['beams']}")
        print(f"nchannels,ntimes,nobandpass: {param['nchannels']} {param['ntimes']} {param['nobandpass']}")
        print(f"ex_chan: {param['ex_chan']}")
        print(f"ex_beamlets: {param['ex_beamlets']}")
        print(f"fclean: {param['fclean']}")
        print(f"bclean: {param['bclean']}")
        print(f"tclean: {param['tclean']}")
        print(f"fflat: {param['fflat']}")
        print(f"tflat: {param['tflat']}")
        print(f"dm: {param['dm']}")
        if 'rms=' in command:
            print(f"rms: {param['rms']}")
        print(f"fcompress,tcompress,fill,round_times,round_freq: {param['fcompress']} {param['tcompress']} {param['fill']} {param['round_times']} {param['round_freq']}\n")

    variab = fits.getheader(file, ext=2, ignore_missing_end=True, memmap=True)
    nt, nf, ns = variab['nt'], variab['nf'], variab['ns']
    dt, df = variab['dt'], variab['df']

    h0 = {
        'fes': np.uint64(0), 'timestamp': np.uint64(0), 'blseqnum': np.uint64(0),
        'fftlen': 0, 'nfft2int': 0, 'fftovl': 0, 'apod': 0, 'nffte': 0, 'nbeamlets': 0
    }

    h0['fes'] = variab['fes'] + np.int64(2) ** 63
    h0['timestamp'] = variab['timestamp'] + np.int64(2) ** 63
    h0['blseqnum'] = variab['blseqnum'] + np.int64(2) ** 63
    h0['fftlen'], h0['nfft2int'], h0['fftovl'] = variab['fftlen'], variab['nfft2int'], variab['fftovl']
    h0['apod'], h0['nffte'], h0['nbeamlets'] = variab['apod'], variab['nffte'], variab['nbeamlets']

    jd0, fref = variab['jd0'], variab['fref']

    if not quiet:
        print(f"\nfes,timestamp,blseqnum: {h0['fes']} {h0['timestamp']} {h0['blseqnum']}")
        print(f"fftlen,nfft2int,fftovl,apod,nffte: {h0['fftlen']} {h0['nfft2int']} {h0['fftovl']} {h0['apod']} {h0['nffte']}")
        print(f"nbeamlets,filesize,beamletsize,blocksize,nblocks: {h0['nbeamlets']} {variab['filesize']} {variab['beamletsize']} {variab['blocksize']} {variab['nblocks']}")
        print(f"nt,nf,ns: {nt} {nf} {ns}")
        print(f"dt,df: {dt} {variab['dtunit']} {df} {variab['dfunit']}")
        print(f"jd0,fref: {jd0} {fref} {variab['frefunit']}\n")

    if not nodata:
        datasize = np.int64(4) * nt * nf * ns
        datasizemax = np.int64(2) ** 31 - 1

        if datasize <= datasizemax:
            data = fits.getdata(file, ext=3, ignore_missing_end=True, memmap=True)
            k = 4
        else:
            data = np.zeros((nt, nf, ns), dtype=np.float32)
            for k in range(ns):
                x = fits.getdata(file, ext=3 + k, ignore_missing_end=True, memmap=True)
                data[:, :, k] = x
            k = k + 3

        # nstokes = 1 (I), 2 (IV), -2 (XY), 3 (IVL), 4 (IQUV)
        if nstokes != 0:
            if nstokes == 1:
                data = np.squeeze(data[:, :, 0])
            elif nstokes == 2 and (ns == 2 or ns == 3):
                data = data[:, :, 0:2]
            elif nstokes == 2 and ns == 4:
                data = data[:, :, [0, 3]]
            elif nstokes == 3 and ns == 4:
                x = np.sqrt(data[:, :, 1] ** 2 + data[:, :, 2] ** 2)
                data = np.concatenate((data[:, :, [0, 3, 1]], x[:, :, None]), axis=-1)
                x = 0

            if nstokes > ns:
                print(f"data selection impossible: ns = {int(ns)}")

        ndata = fits.getdata(file, ext=k, ignore_missing_end=True, memmap=True)
        time = fits.getdata(file, ext=k + 1, ignore_missing_end=True, memmap=True)
        freq = fits.getdata(file, ext=k + 2, ignore_missing_end=True, memmap=True)
        beam = fits.getdata(file, ext=k + 3, ignore_missing_end=True, memmap=True)
        corrt = fits.getdata(file, ext=k + 4, ignore_missing_end=True, memmap=True)
        corrf = fits.getdata(file, ext=k + 5, ignore_missing_end=True, memmap=True)

        if 'rms=' in command:
            rmdata = fits.getdata(file, ext=k + 6, ignore_missing_end=True, memmap=True)
            rm = fits.getdata(file, ext=k + 7, ignore_missing_end=True, memmap=True)

        if not quiet and len(rm) == 0:
            help(file, command, param, variab, nt, dt, nf, df, ns, jd0, h0, fref, data, time, freq, beam, ndata, corrf, corrt)

        if not quiet and len(rm) != 0:
            help(file, command, param, variab, nt, dt, nf, df, ns, jd0, h0, fref, data, time, freq, beam, ndata, corrf, corrt, rmdata, rm)

    else:
        data = np.zeros(1)
        time = np.zeros(1)
        freq = np.zeros(1)
        beam = np.zeros(1)
        ndata = np.ones(1)
        corrt = np.ones(1)
        corrf = np.ones(1)
        rmdata = np.zeros(1)
        rm = np.zeros(1)

        if not quiet:
            help(file, command, param, variab, nt, dt, nf, df, ns, jd0, h0, fref, rmdata, rm)

    return command, param, variab, nt, dt, nf, df, ns, jd0, h0, fref, data, time, freq, beam, ndata, corrf, corrt
