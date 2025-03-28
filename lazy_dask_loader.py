import numpy
import dask
from dask import delayed
import dask.array as da
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
import astropy.units as u
import math

from scipy.signal import lombscargle as LombScargle_scipy
from astropy.timeseries import LombScargle as LombScargle_astropy
#from astroML.time_series import lomb_scargle as LombScargle_astroML

from scipy.interpolate import interp1d

from bitarray_to_bytearray import bitarray_to_bytearray

from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize, ProgressBar

import sys

import glob
from tqdm import tqdm

from typing import Tuple

import csv

def read_csv_to_dict(file_path):
    data_dict = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            data_dict.append(row)
    return data_dict


class LazyFITSLoader:
    """
    LazyFITSLoader class for lazy loading and processing of FITS data using Dask.
    """

    def __init__(
        self, data_fits_file_paths,
        rfi_fits_file_paths,
        stokes,
        exoplanet_name,
        key_project,
        log
        ):
        """
        Initialize the LazyFITSLoader.

        Parameters:
        - data_fits_file_paths (list): List of paths to data FITS files.
        - rfi_fits_file_paths (List): List of paths to RFI FITS file
        - exoplanet_name (str): name of the exoplanet targeted
        """
        self.data_fits_file_paths = data_fits_file_paths
        self.rfi_fits_file_paths = rfi_fits_file_paths
        self.exoplanet_name = exoplanet_name
        self.stokes = stokes
        self.key_project = key_project
        self.client = None
        self.log = log


    def find_rotation_period_exoplanet(self):
        """
        Assumed rotation period (in days) for an exoplanet or a star
        """
        
        if self.log != None:
            self.log.info(f"Starting searching for exoplanet and star period")

        exoplanet_infos = [{
                            'name': 'GJ_687',
                            'star_period': 61.8,
                            'exoplanet_period': 38.142},
                            {
                            'name': 'AD_LEO',
                            'star_period': 2.23,
                            'exoplanet_period': 2.23},
                            {
                            'name': 'CR_DRA',
                            'star_period': 1.984,
                            'exoplanet_period': numpy.nan},
                            {
                            'name': 'COROT-7',
                            'star_period': 11.730891133982574,
                            'exoplanet_period': [0.5825434179024767, 0.13319386531196653, 0.04794132111425209]
                            },
                            {
                            'name': 'TAU_BOO',
                            'star_period': 16.787098628096576,
                            'exoplanet_period': 0.14700659746948372
                            },
                            {'name': "HD_189733",
                            'star_period': numpy.nan,
                            'exoplanet_period': 2.2185733
                            },
                            {'name': "GJ_1151",
                            'star_period': 132,
                            'exoplanet_period': 389.7
                            },
                            {'name':'JUPITER',
                            'star_period':0.41351,
                            'exoplanet_period': 1.7708333333333333} # period of Io, that's what we're looking for (and not droids...)
                            ]

        def get_exoplanet_info(exoplanet_name, exoplanet_infos):
            for info in exoplanet_infos:
                if info['name'].upper() == exoplanet_name:
                    return info['star_period'], info['exoplanet_period']
            return None, None  # Return None if exoplanet_name is not found

        if self.log != None:
            self.log.info(f"Searching in pre-defined informations")
        star_period, exoplanet_period = get_exoplanet_info(self.exoplanet_name.upper(), exoplanet_infos)
        if star_period is not None and exoplanet_period is not None:
            self.exoplanet_period = exoplanet_period
            self.star_period = star_period
            if self.log != None:
                self.log.info(f"Got it in pre-defined informations")
        else:        
            if self.log != None:
                self.log.info(f"Searching in csv file from Palantir")
            self.exoplanet_period = exoplanet_infos[self.exoplanet_name.upper()]
            #self.star_period = star_period[self.exoplanet_name.upper()]
            csv_file = 'exoplanet_informations.csv'
            exoplanet_info_dict =  read_csv_to_dict(csv_file)

            def get_planet_info_by_name(planet_name, csv_data):
                for row in csv_data:
                    if row['name'].upper() == planet_name.upper():
                        return {
                                'planet_orbital_period': row['planet_orbital_period'],
                                'star_rotperiod': row['star_rotperiod']
                            }
                    return None  # Return None if planet_name is not found
            
            planet_info = get_planet_info_by_name(self.exoplanet_name, exoplanet_info_dict)
            if planet_info != None:
                if self.log != None:
                    self.log.info(f"Got it in csv file from Palantir")
                self.exoplanet_period = planet_info['planet_orbital_period']
                self.star_period = planet_info['star_rotperiod']
            else:
                raise ValueError("Exoplanet not found")

        if self.log != None:
            self.log.info(f"Ending searching for exoplanet and star period")


    def _load_data_from_fits(self, log_infos = False):
        """
        Load FITS data and rfi mask in a lazy manner.

        Returns:
        Tuple of Dask arrays representing time, frequency, data, and rfi_mask.
        """
        time = []
        time_interp = []
        frequency = []
        data = []
        rfi_mask0 = []


        progress_bar = tqdm(total=len(self.data_fits_file_paths), desc = "Progress loading data from FITS file")

        if log_infos:
                self.log.info(f"{len(self.data_fits_file_paths)} data FITS file will be read")
        # loading data fits file per fits file
        for count, fits_file_path in enumerate(self.data_fits_file_paths):
            if log_infos:
                self.log.info(f"Start loading data, file {count+1} / {len(self.data_fits_file_paths)}")
            with fits.open(fits_file_path, memmap=True) as hdus:
                len_fits_file_ = len(hdus)
                #command = hdus[0].data
                #param = hdus[1].data
                if len_fits_file_ > 6:
                    variable = hdus[2].data
                    nt_ = variable.NT[0]
                    nf_ = variable.NF[0]
                    ns_ = variable.NS[0]

                    chunk_size_time = nt_
                    chunk_size_frequency = nf_
                    chunk_size_stokes = ns_
                    
                    if ns_ == 4:
                        datasize = 4 * nt_ * nf_ * ns_
                    if ns_ == 3:
                        datasize = 3 * nt_ * nf_ * ns_
                    datasizemax = 2**31 - 1
                    
                    
                    if self.stokes.lower() != 'rm':
                        if datasize <= datasizemax:
                            data_ = da.from_array(hdus[3].data.T, chunks=(chunk_size_time, chunk_size_frequency, chunk_size_stokes))
                            k = 0
                        else:
                            data_ = da.zeros((chunk_size_stokes, chunk_size_frequency, chunk_size_time))
                            for k in range(ns_):
                                data_[k, :, :] = da.from_array(hdus[3 + k].data, chunks=(chunk_size_frequency, chunk_size_time))
                            data_ = data_.T
                            
                    else:
                            if datasize <= datasizemax:
                                k = 0
                            else:
                                k = 3
                            data_ = da.from_array(hdus[-2].data.T, chunks = (chunk_size_frequency, chunk_size_time))
                            
                    rfilevel0_ = da.from_array(hdus[4 + k].data.T, chunks=(chunk_size_time, chunk_size_frequency))
                    #time_ = da.from_array((Time(hdus[2].data['timestamp'][0], format='unix') + TimeDelta(hdus[5 + k].data, format='sec')).value, chunks = chunk_size_time)
                    time_ = da.from_array(Time(hdus[2].data["timestamp"][0] + hdus[5+k].data, format="unix").unix, chunks = chunk_size_time)

                    #frequency_ = hdus[6 + k].data * u.MHz
                    if self.stokes.lower() != 'rm':
                        frequency_ = da.from_array((hdus[6 + k].data * u.MHz).value, chunks=chunk_size_frequency)
                    else:
                        frequency_ = da.from_array(hdus[-1].data, chunks=chunk_size_frequency)

                    if self.interpolation_in_time:
                        #new_interval = (time_[1]-time_[0])*self.interpolation_in_time_factor
                        time_interp_ = da.arange(time_[0], time_[-1], self.interpolation_in_time_value)
                    else:
                        time_interp_ = []

            if len_fits_file_ > 6:
                # Appending 
                time.append(time_)
                time_interp.append(time_interp_)
                frequency.append(frequency_)
                data.append(data_)
                if (self.apply_rfi_mask == True) and (self.rfi_mask_level == 0):
                        rfi_mask0.append(rfilevel0_)
            else:
                if log_infos:
                    self.log.info(f"File {count+1} / {len(self.data_fits_file_paths)}: {fits_file_path} seems to have a length issue. Skipping...")
            
            if log_infos:
                self.log.info(f"End loading data, file {count+1} / {len(self.data_fits_file_paths)}")

            progress_bar.update()
        progress_bar.close()
    
        return time, time_interp, frequency, data, rfi_mask0


    def _load_RFI_data_from_fits(self, log_infos = False):
        """
        Load FITS rfi mask in a lazy manner.

        Returns:
        Tuple of Dask array representing rfi_mask.
        """
        rfi_mask = []

        if log_infos:
            self.log.info("Start reading mask level > 0")
            self.log.info(f"{len(self.rfi_fits_file_paths)} RFI FITS file will be read")
            
        progress_bar = tqdm(total=len(self.rfi_fits_file_paths), desc = "Progress loading RFI data from FITS file")
        
        for count, fits_file_path in enumerate(self.rfi_fits_file_paths):
            if log_infos:
                self.log.info(f"Start reading RFI data (mask > 0), file {count+1} / {len(self.rfi_fits_file_paths)}")
            
            with fits.open(fits_file_path, memmap=True) as hdus:
                ndatasize = hdus[2].data
                ndatabit = da.from_array(hdus[3].data, chunks=(ndatasize[1]))

                # Convert bits array to bytes array
                ndatabyte = bitarray_to_bytearray(ndatabit, ndatasize)
                
                rfimask_level1_ = ndatabyte[:,:, 0]
                rfimask_level2_ = ndatabyte[:,:, 1]
                rfimask_level3_ = ndatabyte[:,:, 2]


                if self.rfi_mask_level == 1:
                    #rfi_mask.append(da.from_array(rfimask_level1_, chunks=(nt_, 1)))
                    rfi_mask.append(rfimask_level1_)
                elif self.rfi_mask_level == 2:
                    rfi_mask.append(rfimask_level2_)
                elif self.rfi_mask_level == 3:
                    rfi_mask.append(rfimask_level3_)

            if log_infos:
                self.log.info(f"End reading RFI data (mask > 0), file {count+1} / {len(self.rfi_fits_file_paths)}")

            progress_bar.update()
        
        progress_bar.close()
        if log_infos:
            self.log.info("End reading mask > 0")

        return rfi_mask


    def _multiply_data(self, data1, data2):
        """
        Multiply data1 by data2 element-wise.

        Parameters:
        - data1 (Dask array): First array to multiply.
        - data2 (Dask array): Second array to multiply.

        Returns:
        Dask array resulting from element-wise multiplication.
        """
        
        return data1 * data2


    def lazy_interp(self, x, xp, fp, axis=0, dtype = float):
        """
        Rebin the data to a new array (axis = 0 == Time, axis = 1 == Frequency)

        Parameters:
            fp: data to be rebinned
            x (dask.array): array on which the the rebin will be done
            xp: original array corresponding to fp in the axis direction
            axis : axis over which interpolation needs to be done (axis = 0 == Time, axis = 1 == Frequency)

        Returns:
            rebinned data array
        """ 
        interp_func = interp1d(xp, fp, axis=axis, kind='linear', bounds_error = None)
        return interp_func(x)

    def lazy_rebin(self,
                    new_axis_array,
                    dx_new_axis_array,
                    original_axis_array,
                    data,
                    axis = 0
                    ):
            """
            Rebins the data along a specified axis by averaging over bins.

            Parameters:
                data: Dask array representing the data to be rebinned.
                original_axis_array: NumPy array representing the original axis values.
                new_axis_array: NumPy array representing the new axis values after rebinning.
                axis: Integer indicating the axis along which rebinning needs to be done (0 or 1).

            Returns:
                The rebinned data array (Dask array).
            """

            if axis == 0:
                data_rebined = numpy.zeros((len(new_axis_array), data.shape[1]))
                for index_axis, value_axis  in enumerate(new_axis_array):
                    data_rebined[index_axis, :] = numpy.nanmean(data[(original_axis_array >= value_axis) & (original_axis_array < value_axis+dx_new_axis_array),:], axis=0)
            if axis == 1:
                data_rebined = numpy.zeros((data.shape[0], len(new_axis_array)))
                for index_axis, value_axis  in enumerate(new_axis_array):
                    data_rebined[:, index_axis] = numpy.nanmean(data[:, (original_axis_array >= value_axis) & (original_axis_array < value_axis+dx_new_axis_array)], axis=1)

            return data_rebined

    def lazy_rebin_old(self,
                    new_axis_array,
                    original_axis_array,
                    data,
                    axis = 0
                    ):
            """
            Rebins the data along a specified axis by averaging over bins.

            Parameters:
                data: Dask array representing the data to be rebinned.
                original_axis_array: NumPy array representing the original axis values.
                new_axis_array: NumPy array representing the new axis values after rebinning.
                axis: Integer indicating the axis along which rebinning needs to be done (0 or 1).

            Returns:
                The rebinned data array (Dask array).
            """

            if axis not in (0, 1):
                raise ValueError("Axis value should be 0 or 1.")

            initial_size = original_axis_array.size
            final_size = new_axis_array.size

            # Calculate the spacing values dx and new_dx
            dx = numpy.mean(numpy.diff(original_axis_array))
            new_dx = numpy.mean(numpy.diff(new_axis_array))
            # Determine the bin edges for the new axis
            bin_edges = numpy.concatenate(([new_axis_array[0] - new_dx / 2], new_axis_array + new_dx / 2))

            # Calculate the bin indices for each point along the original axis
            bin_indices = numpy.digitize(original_axis_array, bin_edges) - 1

            # Perform rebinning along the specified axis
            if axis == 0:
                rebinned_data = da.stack([da.nanmean(data[bin_indices == i], axis=axis) for i in range(len(new_axis_array))])
            else:
                rebinned_data = da.stack([da.nanmean(data[:, bin_indices == i], axis=1) for i in range(len(new_axis_array))], axis=1)

            return rebinned_data


        #return da.from_array(rebinned_data, chunks=new_time_chunks)

         
    def lazy_interpolate_with_rfi_mask(self, time_interp, time, data1, data2, axis = 0, dtype = float):
        # Perform linear interpolation on data1_block
        # Still need to be tested
        interp_func1 = interp1d(time, (data1*data2), axis = axis, kind='linear')
        interpolated_values1 = interp_func1(time_interp)
        
        interp_func2 = interp1d(time, data2, axis = axis, kind = 'linear')
        interpolated_values2 = interp_func2(time_interp)  

        numerator = da.from_array(interpolated_values1, chunks=(len(time_interp))) 
        denominator = da.from_array(interpolated_values2, chunks=(len(time_interp)))

        result = self.safe_divide(numerator, denominator)
        
        return result


    def lazy_rebin_with_rfi_mask(self, time_interp, dtime_interp, time, data1, data2, axis = 0, dtype = float):
    
        data_tmp = data1*data2
        
        numerator = da.empty_like(time_interp)
        denominator = da.empty_like(time_interp)

        end_times = time_interp + dtime_interp

        for index_axis, value_axis  in enumerate(time_interp):
            tmp_mask = (time >= value_axis) & (time < end_times[index_axis])
            numerator[index_axis] = numpy.nanmean(data_tmp[tmp_mask])
            denominator[index_axis] = numpy.nanmean(data2[tmp_mask])

        #numerator = da.from_array(numerator, chunks=(len(time_interp)))
        #denominator = da.from_array(denominator, chunks=(len(time_interp)))

        
        
        return self.safe_divide(numerator, denominator)


    def safe_divide(self, numerator, denominator):
        def divide_chunk(numerator_chunk, denominator_chunk):
            result_chunk = numpy.empty_like(numerator_chunk, dtype=float)
            zero_mask = denominator_chunk == 0
            #result_chunk[zero_mask] = numpy.nan
            result_chunk[~zero_mask] = numerator_chunk[~zero_mask] / denominator_chunk[~zero_mask]
            return result_chunk

        return da.map_blocks(divide_chunk, numerator, denominator, dtype=float)


    def get_dask_array(self, frequency_interval = [4,5], stokes='I',
                        apply_rfi_mask = False, rfi_mask_level = 0, rfi_mask_level0_percentage = 10,
                        interpolation_in_time = False, interpolation_in_time_value = 1,
                        interpolation_in_frequency = False, interpolation_in_frequency_value = 0.100,
                        verbose = False,
                        log_infos = False,
                        output_directory = './'):
        """
        Get Dask array for a specific frequency and Stokes parameter.

        Parameters:
        - frequency_interval (list(int)): list of min and max frequency interval over which operations would be done.
        - stokes (str): Stokes parameter ('I', 'Q', 'U', 'V', or 'L').
        - apply_rfi_mask (Bool): Boolean set to optionnaly apply a RFI mask on the data
        - rfi_mask_level (str): Level of the RFI mask to apply on the data
        - rfi_mask_level0_percentage (float): percentage at which thresholed the rfi mask level 0 (active if apply_rfi_mask = True & rfi_mask_level = 0)
        - interpolation_in_time (Bool): Boolean set to optionnaly interpol the data over the time direction
        - interpolation_in_time_value (float): Value in second to be used for the interpolation (default is 1 second)
        - interpolation_in_frequency (Bool): Boolean set to optionnaly interpol the data over the frequency direction
        - interpolation_in_frequency_value (float): Value in MegaHertz (MHz) to be used for the interpolation (default is 0.100 MHz)

        Returns:
        Tuple of Dask arrays representing time, frequency, data, ndata, and the resulting masked Stokes parameter.
        """
        self.apply_rfi_mask = apply_rfi_mask
        self.rfi_mask_level = rfi_mask_level
        self.rfi_mask_level0_percentage = rfi_mask_level0_percentage
        self.interpolation_in_time = interpolation_in_time
        self.interpolation_in_time_value = interpolation_in_time_value
        self.interpolation_in_frequency = interpolation_in_frequency
        self.interpolation_in_frequency_value = interpolation_in_frequency_value
        #self.stokes = stokes


        stokes = self.stokes
        

        #if len(i_frequency) == 1:
        #    i_frequency = [i_frequency[0],i_frequency[0]+1]
        #if i_frequency[1] == i_frequency[0]:
        #    i_frequency[1] += 1
        
        
        
        lazy_object_data = self._load_data_from_fits(log_infos=log_infos)
        time_, time_interp_, frequency_, data_, rfi_mask_ = lazy_object_data

        frequencies =  [(frequency_[i_obs].compute()) for i_obs in range(len(frequency_))]
        
        #return time_, frequencies, data_
        #if i_frequency[1] == -1:
        #    i_frequency[1] = len(frequency_)-1
    

        if (self.apply_rfi_mask == True) and (self.rfi_mask_level > 0):
            rfi_mask_ = self._load_RFI_data_from_fits(log_infos=log_infos)


        #time_interp_tmp = []
        #for i_obs in range(len(time_)):
        #    if self.interpolation_in_time:
        #        time_interp_tmp = da.arange(time_[i_obs][0], time_[i_obs][-1], self.interpolation_in_time_value, chunks = (time_[i_obs][-1]-time_[i_obs][0])/interpolation_in_time_value)
        #    else:
        #        time_interp_tmp = []
        #    time_interp_.append(time_interp_tmp)

        data_final_ = []
        frequency_final_ = []
        iobs_wrong = []
        time_final_ = []
        #rfi_mask_tmp_ = []
        if self.apply_rfi_mask:
            progress_bar = tqdm(total=len(time_), desc = "Progress interpolating data and applying mask")
            if log_infos:
                self.log.info("Start applying mask and interpolating data")
        else: 
            progress_bar = tqdm(total=len(time_), desc = "Progress interpolating data")
            if log_infos:
                self.log.info("Start interpolating data")

        for i_obs in range(len(time_)):

            if data_[i_obs].shape[2] == 4:
                stokes_index = {
                                'I': 0,
                                'Q': 1,
                                'U': 2,
                                'V': 3,
                                'V+':3,
                                'V-':3
                                }
            elif data_[i_obs].shape[2] == 3:
                stokes_index = {
                                'I': 0,
                                'V': 1,
                                'V+':1,
                                'V-':1,
                                'L': 2
                                }
            else:
                stokes_index = {
                                'RM': 0,
                                }
            # Interpolation in time and mask applying is done obs. per obs.    
            if log_infos:
                self.log.info(f"Starting {i_obs+1} / {len(time_)} observation")
            
            w_frequency = numpy.where((frequencies[i_obs] >= frequency_interval[0]) & (frequencies[i_obs] <= frequency_interval[1]))[0]
            if log_infos:
                if len(w_frequency) == 0:
                    self.log.info(f"No observations in the frequency range asked by users")

            if len(w_frequency) != 0:
                if self.apply_rfi_mask == True:
                    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                        rfi_mask_to_apply = rfi_mask_[i_obs][:, w_frequency]

                    if self.rfi_mask_level == 0:
                        rfi_mask_to_apply[rfi_mask_to_apply >= self.rfi_mask_level0_percentage/100] = 1
                        rfi_mask_to_apply[rfi_mask_to_apply < self.rfi_mask_level0_percentage/100] = 0

                    if stokes == 'VI':
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))
                            for count, index_frequency in enumerate(w_frequency):

                                data_tmp_[:, count] = da.map_blocks(
                                                        self.lazy_interpolate_with_rfi_mask,
                                                        time_interp_[i_obs],
                                                        time_[i_obs],
                                                        self.safe_divide(data_[i_obs][:, index_frequency, stokes_index['V']].rechunk((time_[i_obs].chunks[0])), 
                                                                        data_[i_obs][:, index_frequency, stokes_index['I']].rechunk((time_[i_obs].chunks[0]))),
                                                        rfi_mask_to_apply[:,count],
                                                        axis = 0,
                                                        dtype=float   
                                                        )
                        else:
                            data_tmp_ = self._multiply_data(self.safe_divide(data_[i_obs][:, w_frequency,  stokes_index['V']],
                                                                            data_[i_obs][:, w_frequency,  stokes_index['I']]),
                                                            rfi_mask_to_apply)       
                    elif stokes == 'L':
                        if data_[i_obs].shape[2] == 4:
                            data_stokes_L = numpy.sqrt(data_[i_obs][:, w_frequency, stokes_index['Q']]**2 + data_[i_obs][:, w_frequency, stokes_index['U']]**2)
                        else:
                            data_stokes_L = data_[i_obs][:, index_frequency, stokes_index[stokes]]
                        if self.interpolation_in_time:
                            for count, index_frequency in enumerate(w_frequency):
                                chunk_size_time_interp = len(time_interp_[i_obs])
                                data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))
                                data_tmp_[:, count] = da.map_blocks(
                                                        self.lazy_interpolate_with_rfi_mask,
                                                        time_interp_[i_obs],
                                                        time_[i_obs],
                                                        data_stokes_L[:, count].rechunk((time_[i_obs].chunks[0])),
                                                        rfi_mask_to_apply[:,count],
                                                        axis = 0,
                                                        dtype=float   
                                                        )
                        else:
                            data_tmp_ = self._multiply_data(data_stokes_L, rfi_mask_to_apply)

                    else: # elif stokes != 'L' & != VI:
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))
                            for count, index_frequency in enumerate(w_frequency):
                                data_tmp_[:, count] = da.map_blocks(
                                                        self.lazy_interpolate_with_rfi_mask,
                                                        time_interp_[i_obs],
                                                        time_[i_obs],
                                                        data_[i_obs][:, index_frequency, stokes_index[stokes]].rechunk((time_[i_obs].chunks[0])),
                                                        rfi_mask_to_apply[:,count],
                                                        axis = 0,
                                                        dtype=float   
                                                        )
                        else:
                            data_tmp_ = self._multiply_data(data_[i_obs][:, w_frequency,  stokes_index[stokes]], rfi_mask_to_apply)                
                    
                    
                                                
                else:
                    if stokes == 'RM':
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))
                            for count, index_frequency in enumerate(w_frequency):
                                    data_tmp_[:,count] = da.map_blocks(
                                                            self.lazy_interp,
                                                            time_interp_[i_obs],
                                                            time_[i_obs],
                                                            data_[i_obs][:, index_frequency].rechunk((time_[i_obs].chunks[0])),
                                                            axis = 0,
                                                            dtype=float                 
                                                            )
                                    
                        else:
                            data_tmp_ = data_[i_obs][:, w_frequency]

                    elif stokes == 'L':
                        print(data_[i_obs].shape[2])
                        if data_[i_obs].shape[2] == 4:
                            data_stokes_L = numpy.sqrt(data_[i_obs][:, w_frequency, stokes_index['Q']]**2 + data_[i_obs][:, w_frequency, stokes_index['U']]**2)
                        else:
                            data_stokes_L = data_[i_obs][:, index_frequency, stokes_index[stokes]]
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))
                            for count, index_frequency in enumerate(w_frequency):
                                data_tmp_[:,count] = da.map_blocks(
                                                            self.lazy_interp,
                                                            time_interp_[i_obs],
                                                            time_[i_obs],
                                                            data_stokes_L[:, count].rechunk((time_[i_obs].chunks[0])),
                                                            axis = 0,
                                                            dtype=float                 
                                                            )
                        else:
                            data_tmp_ = data_stokes_L

                    elif stokes == 'VI':
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))

                            for count, index_frequency in enumerate(w_frequency):
                                data_tmp_[:,count] = da.map_blocks(
                                                        self.lazy_interp,
                                                        time_interp_[i_obs],
                                                        time_[i_obs],
                                                        self.safe_divide(data_[i_obs][:, index_frequency, stokes_index['V']].rechunk((time_[i_obs].chunks[0])), 
                                                                        data_[i_obs][:, index_frequency, stokes_index['I']].rechunk((time_[i_obs].chunks[0]))),
                                                        axis = 0,
                                                        dtype=float                 
                                                        )
                                    
                        else:
                            data_tmp_ = self.safe_divide(data_[i_obs][:, w_frequency, stokes_index['V']], 
                                                         data_[i_obs][:, w_frequency, stokes_index['I']])

                                                                    
                    else: # elif stokes != 'L' & != 'RM' & != 'VI':
                    # This part works!
                        if self.interpolation_in_time:
                            chunk_size_time_interp = len(time_interp_[i_obs])
                            data_tmp_ = da.zeros((chunk_size_time_interp, len(frequencies[i_obs][w_frequency])))

                        # This part is commented, because the rebinning (instaed of interpolating) needs to be tested and double checked
                            #data_tmp_ = da.map_blocks(
                            #                        lazy_rebin,
                            #                        data_[i_obs][:, i_frequency,0],
                            #                        new_time=time_interp_[i_obs],
                            #                        new_time_chunks=(time_interp_[i_obs].size),
                            #                        dtype=float
                            #                        )
                            
                            for count, index_frequency in enumerate(w_frequency):
                                data_tmp_[:,count] = da.map_blocks(
                                                        self.lazy_interp,
                                                        time_interp_[i_obs],
                                                        time_[i_obs],
                                                        data_[i_obs][:, index_frequency, stokes_index[stokes]].rechunk((time_[i_obs].chunks[0])),
                                                        axis = 0,
                                                        dtype=float                 
                                                        )
                                    
                        else:
                            data_tmp_ = data_[i_obs][:, w_frequency,  stokes_index[stokes]]
                        
                    


                # Interpolating in frequency        
                if self.interpolation_in_frequency:
                    #if (frequencies[i_obs][w_frequency][-1] - frequencies[i_obs][w_frequency][0]) >= self.interpolation_in_frequency_value:
                    frequency_interp = da.arange(frequency_interval[0], frequency_interval[-1], self.interpolation_in_frequency_value)
                    data_tmp_ = da.map_blocks(
                                            self.lazy_rebin,
                                            frequency_interp,
                                            self.interpolation_in_frequency_value,
                                            frequencies[i_obs][w_frequency],
                                            data_tmp_,
                                            axis = 1,
                                            dtype=float                 
                                            )
                    frequency = frequency_interp
                    
                    #else:
                    #    if log_infos:
                    #        self.log.info("Interpolation in frequency can't be done, because selected frequency range is smaller than the interpolation value")
                            #raise Warning("Interpolation in frequency can't be done, because selected frequency range is smaller than the interpolation value")
                    #    frequency = da.array(frequencies[i_obs][w_frequency])
                else:
                    frequency = da.array(frequencies[i_obs][w_frequency])
                
                #if log_infos:
                #    self.log.info(f"Time_interp_ length: {len(time_interp_[i_obs])} / {len(data_tmp_[:,0])}: data_tmp_ length")

                if self.interpolation_in_time:
                    if len(time_interp_[i_obs]) != len(data_tmp_[:,0]):
                        iobs_wrong.append(i_obs)
                else:
                    if len(time_[i_obs]) != len(data_tmp_):
                        iobs_wrong.append(i_obs)
                

                data_final_.append(data_tmp_)
                #if self.apply_rfi_mask == True:
                #    rfi_mask_tmp_.append(rfi_mask_to_apply)
                frequency_final_.append(frequency)
                if self.interpolation_in_time:
                    time_final_.append(time_interp_[i_obs])
                else:
                    time_final_.append(time_[i_obs])

            

            if log_infos:
                self.log.info(f"Ending {i_obs+1} / {len(time_)} observation")
        
            progress_bar.update()
        
        progress_bar.close()


        if self.apply_rfi_mask:
            if log_infos:
                self.log.info("End applying mask and interpolating data")
        else: 
            
            if log_infos:
                self.log.info("End interpolating data")
            
            
        # Concatenating of arrays over observation

        if log_infos:
            self.log.info(f"{len(iobs_wrong)} / {len(time_)} observations are wrong")
            for index_iobswrong in iobs_wrong:
                self.log.info(f"Observation {index_iobswrong} is wrong")

        #if log_infos:
            #self.log.info(f"Number of obs kept: time_final_ length: {len(time_final_)} / {len(data_final_)}: data_final_ length")
            #for iobs_included in range(len(data_final_)):
            #    self.log.info(f"Obs {iobs_included}: Time_final_ length: {len(time_final_[iobs_included])} / {len(data_final_[iobs_included])}: data_final_ length")



        if len(iobs_wrong) !=0:
            time_filtered = [time_final_[i] for i in range(len(time_final_)) if i not in iobs_wrong]
            time_final_ = time_filtered
            filtered_data = [data_final_[i] for i in range(len(data_final_)) if i not in iobs_wrong]
            data_final_ = filtered_data

        extra_name = ''
        if self.apply_rfi_mask != False:
            if self.rfi_mask_level == 0:
                extra_name = '_masklevel'+str(int(self.rfi_mask_level))+'_'+str(int(self.rfi_mask_level0_percentage))+'percents'
            else:
                extra_name = '_masklevel'+str(int(self.rfi_mask_level))
        else:
            extra_name = '_nomaskapplied'
        extra_name = extra_name+'_'+f'{int(frequency_interval[0])}-{int(frequency_interval[1])}MHz'


        time_final = da.concatenate(time_final_, axis = 0)
        
        if numpy.max(frequency_final_[-1]) - numpy.max(frequency_final_[0]) > 1e-8:
            raise ValueError("Frequency observation are not the same. Something needs to be modified in the function. Exiting.")
        else:
            frequency = frequency_final_[0]
        #frequency_final = da.concatenate(frequency_final_, axis = 0)
        # Then check if all frequency_final[:,0] == same
        # if not, needs of re-aligning everything (how? That's the question)
        
        data_final = da.concatenate(data_final_, axis=0)



        #if log_infos:
        #    self.log.info("Starting saving data as dask arrays")
        #da.to_hdf5(output_directory+'preliminary_dask_array_data-'+self.stokes+'_LT'+self.key_project+'_'+self.exoplanet_name+extra_name+'.hdf5', {'time': time_final, 'frequency': frequency, 'data': data_final})  
        #if log_infos:
        #    self.log.info("Ending saving data as dask arrays")


        if log_infos:
                self.log.info(f"Number of time step after interpolation / compared to time step in the data set (safety check) => time length: {len(time_final)} / {len(data_final)}: data_final length")
        
        
        #if self.apply_rfi_mask == True:
        #    rfi_mask = da.concatenate(rfi_mask_tmp_, axis=0)
        if log_infos:
            self.log.info("Starting computing data")
        if verbose:
            with Profiler() as prof, ResourceProfiler(dt=0.0025) as rprof, CacheProfiler() as cprof:
                with ProgressBar():
                    time = time_final.compute()
                with ProgressBar():
                    frequency = frequency.compute()
                with ProgressBar():
                    data_final = data_final.compute()
            visualize([prof, rprof, cprof,])
        else:
            time = time_final.compute()

            frequency = frequency.compute()
            data_final = data_final.compute()
        if log_infos:
            self.log.info("Ending computing data")
    
        

        # Because if observations were done with 2 different lanes, it is possible that there is duplicate in time array
        # So checking it and correcting it if necessary

        unique_times = numpy.unique(time)
        merged_data = numpy.full((len(unique_times), data_final.shape[1]), numpy.nan)

        # Iterate through unique times and merge corresponding rows
        for i_, t_ in enumerate(unique_times):
            mask = time == t_  # Find indices of this time in the original array
            # Merge using nanmax --> We can do that as making observations with different lanes implies different frequencies.
            if numpy.any(mask):  # Only apply nanmax if at least one match exists
                merged_data[i_] = numpy.nanmax(data_final[mask], axis=0)
            else:
                merged_data[i_] = numpy.nan  # If no match, keep NaN values
  

        # Replace old arrays with merged ones
        time = unique_times
        data_final = merged_data



        return time, frequency, data_final

    def LS_calculation(self, time, data, threshold, normalized_LS = False, log_infos = False, type_LS = "scipy"):
        """
        Calculate LombScargle periodogram for given time[nt] and data[nt] computed Dask array (so for a specific frequency and Stokes parameter).

        Parameters:
        - time (1D array)
        - data (1D array)
        
        Returns:
        - f_LS: frequency of the LS periodogram
        - power_LS: power of the LS periodogram
        """
        self.find_rotation_period_exoplanet()
        
        nout=100000
        T_exoplanet = numpy.mean(self.exoplanet_period)*24*60*60 # T needs to be in seconds #numpy.mean() is taken, in case T_exoplanet is a list
        T1 = T_exoplanet/10     # Period min
        T2 = T_exoplanet*10     # Period max
        w1 = 2*numpy.pi/T1      # Pulsation max
        w2 = 2*numpy.pi/T2      # Pulsation min
        #f_LS = numpy.logspace(numpy.log10(w2), numpy.log10(w1), nout)  / (2 * numpy.pi) #Frequencies at which to search for periodicity with LombScargle
        f_LS = numpy.linspace(w2, w1, nout)  / (2 * numpy.pi) #Frequencies at which to search for periodicity with LombScargle
        if self.stokes == 'V+':
            data[data < 0] = 0
        if self.stokes == 'V-':
            data[data > 0] = 0
        if self.stokes == 'V':
            data = numpy.abs(data)
        if self.stokes == 'VI':
            data[numpy.abs(data) > 1] = numpy.nan # unphysical points
            if threshold != None:
                data[numpy.abs(data) < threshold] = 0 # remove points <threshold
                data[numpy.abs(data) >= threshold] = 1 # put every point > threshold at 1
            
            data = numpy.abs(data)

        data[numpy.isnan(data)] = 0
        data[numpy.isinf(data)] = 0

        if log_infos:
            self.log.info("Starting Lomb Scargle periodogram computation")
        
        if type_LS.lower() == 'scipy':
            power_LS = LombScargle_scipy(time, data, f_LS, normalize=normalized_LS)
        if type_LS.lower() == 'astroml':
            power_LS = LombScargle_astroML(time, data, f_LS)
        if type_LS.lower() == 'astropy':
            #time in jd Time(data)
            if normalized_LS:
                normalization='standard' # default normalized_power = power/mean(data**2)
                normalization='model' # normalized_data = (data - mean(data))/std(data) &  normalized_power = power/mean(normalized_data**2)
            else:
                normalization='standard' #(default)
            
            method = 'auto'
            #method = 'slow' #   This method uses a slower but more accurate implementation of the Lomb-Scargle algorithm suitable for unevenly sampled data. It's based on the work of Lomb (1976) and Scargle (1982).
                             #   The 'slow' method is suitable for datasets with irregular sampling intervals or when high accuracy is required.
            #method = 'chi2' #   This method computes the Lomb-Scargle periodogram using a chi-square statistic, which provides a robust estimate of the periodogram for unevenly sampled data.
                             #   The 'chi2' method is useful when dealing with datasets with significant measurement uncertainties or when you want a statistically robust estimate of the periodogram.

            #fit_mean:  Astropy's Lomb-Scargle implementation fits a constant mean to the data before computing the periodogram. This involves subtracting the mean value from the data, which effectively centers the data around zero.
            #           Fitting a mean to the data helps remove any systematic offsets or trends that may be present in the data, which can improve the accuracy of the periodogram.
            #           This option is useful when you want to focus on periodic variations in the data while removing any constant offset or trend.
            fit_mean = False

            frequency_LS, power_LS = LombScargle_astropy(time, data, fit_mean = fit_mean).autopower(method = method, minimum_frequency = f_LS[0], maximum_frequency = f_LS[-1], samples_per_peak=100)
            f_LS = frequency_LS
        if log_infos:
            self.log.info("End Lomb Scargle periodogram computation")

        return f_LS, power_LS


    



