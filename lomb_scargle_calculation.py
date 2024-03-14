import numpy
from dask import delayed
import dask.array as da
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
import astropy.units as u
import math

from scipy.signal import lombscargle

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from bitarray_to_bytearray import bitarray_to_bytearray

from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize, ProgressBar
from dask.distributed import Client, LocalCluster, wait

import logging
import sys

import glob

import multiprocessing


from lazy_dask_loader import LazyFITSLoader
from LS_calculator_function import calculate_LS


import argparse


from h5py import File


def save_to_hdf(time,
                frequency_obs,
                data_final,
                frequency_LS,
                power_LS,
                output_directory,
                key_project,
                target,
                ):
    """
    Saves the data to disk as a HDF5 file.
    """
    output_file = File(output_directory+'lomb_scargle_periodogram_LT'+key_project+'_'+target+'.hdf5', 'w')
    output_file.create_dataset('Time', data = time)
    output_file['Time'].attrs.create('format', 'unix')
    output_file['Time'].attrs.create('units', 's')
    output_file.create_dataset('Frequency_Obs', data=frequency_obs)
    output_file['Frequency_Obs'].attrs.create('units', 'MHz')
    output_file.create_dataset('Frequency_LS', data=frequency_LS)
    output_file['Frequency_LS'].attrs.create('units', 's')
    output_file.create_dataset('power_LS', data=power_LS)
    output_file.close()

def plot_LS_periodogram(frequencies,
                        f_LS,
                        power_LS,
                        output_directory):
    dpi = 200
    fig, axs = plt.subplots(nrows=len(frequencies), sharex=True, dpi=dpi)

    T_io = 1.769137786
    T_jupiter = 9.9250/24
    T_synodique = (T_io*T_jupiter)/abs(T_io-T_jupiter)

    for index_freq in range(len(frequencies)):
        

        axs[index_freq].plot(1/f_LS[index_freq]/60/60, power_LS[index_freq])
        #plt.yscale('log')
        axs[index_freq].set_title(f'Frequency: {frequencies[index_freq]} MHz')
        lazy_loader.find_rotation_period_exoplanet()
        T_exoplanet = lazy_loader.exoplanet_period # in days
        axs[index_freq].vlines([T_io*24], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='r')
        axs[index_freq].vlines([T_io*24/2], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='r', linestyles="dashed")
        axs[index_freq].vlines([T_jupiter*24], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='g')
        axs[index_freq].vlines([T_jupiter*24/2], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='g', linestyles="dashed")
        axs[index_freq].vlines([T_synodique*24], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='y')
        axs[index_freq].vlines([T_synodique*24/2], power_LS[index_freq].min(), power_LS[index_freq].max(), colors='y', linestyles="dashed")
        axs[index_freq].xaxis.set_minor_locator(MultipleLocator(1))
        axs[index_freq].xaxis.set_major_locator(MultipleLocator(5))

    axs[index_freq].set_xlim([4,50])
    axs[index_freq].set_xlabel("Periodicity (Hours)")
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_directory+'lomb_scargle_periodogram_LT'+key_project+'_'+target, dpi = dpi, format = png)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Calculation of Lomb Scargle Periodogram for observations of a given radio emitter (planet, exoplanet or star)")
    parser.add_argument('-key_project', dest = 'key_project', required = True, type = str, help = "NenuFAR Key Project number (02, or 07)")
    parser.add_argument('-target', dest = 'target', required = True, type = str, help = "Observation Target Name (planet, exoplanet or star )")
    parser.add_argument('--main_directory_path', dest = 'root', default = './data/', type = str, help = "Main directory path where the observation are stored")
    parser.add_argument('--stokes', dest = 'stokes', default = 'V', type = str, help = "Stokes parameter to be study")
    parser.add_argument('--apply_rfi_mask', dest = 'apply_rfi_mask', default = False, action = 'store_true', help = "Apply RFI mask")
    parser.add_argument('--rfi_mask_level', dest = 'rfi_mask_level', default = 0, type = int, help = "Level of the RFI mask to apply (needed if --apply_rfi_mask True). Option are 0, 1, 2, or 3")
    parser.add_argument('--rfi_mask_level0_percentage', dest = 'rfi_mask_level0_percentage', default = 10, type = float, help = "Percentage (i.e. threshold) of the RFI mask level to apply (needed if --apply_rfi_mask True and rfi_mask_level is 0). Values can be between 0 and 100 %")
    parser.add_argument('--interpolation_in_time', dest = 'interpolation_in_time', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_time_value', dest = 'interpolation_in_time_value', default = 1, type = float, help = "Value in second over which data need to be interpolated")
    parser.add_argument('--interpolation_in_frequency', dest = 'interpolation_in_frequency', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_frequency_value', dest = 'interpolation_in_frequency_value', default = 1, type = float, help = "Value in MegaHertz (MHz) over which data need to be interpolaed")
    parser.add_argument('--frequency_interval', dest = 'frequency_interval', nargs = 2, type = float, default = [10,90], help = "Minimal and maximal frequency values over which the Lomb Scargle analysis has to be done")
    parser.add_argument('--verbose', dest = 'verbose', default = False, action = 'store_true', help = "To print on screen the log infos")
    parser.add_argument('--log_infos', dest = 'log_infos', default = False, action = 'store_true', help = "To print on screen the dask computing info, and control graphics after computation")
    
    parser.add_argument('--save_as_hdf5', dest = 'save_as_hdf5', default = False, action = 'store_true', help = "To save results in an hdf5 file")
    parser.add_argument('--plot_results', dest = 'plot_results', default = False, action = 'store_true', help = "To plot and save results")
    parser.add_argument('--output_directory', dest = 'output_directory', default = './', type = str, help = "Output directory where to save hdf5 and/or plots")
    
    args = parser.parse_args()

    
    # Searching for files
    #key_project = '07'
    #target = 'JUPITER'
    level_of_preprocessed = ''

    if args.key_project == '07':
        sub_path = "*/*/*/"
    else:
        sub_path = "*/*/*/*/"

    data_fits_file_paths = [
                filename
                for filename in glob.iglob(
                    f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*spectra*.fits',
                    recursive=True
                )
            ]

    rfi_fits_file_paths = [
                filename
                for filename in glob.iglob(
                    f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*rfi*.fits',
                    recursive=True
                )
            ] 

    lazy_loader = LazyFITSLoader(data_fits_file_paths, rfi_fits_file_paths, 
                                args.target
                            )
    

    time, frequencies, data_final = lazy_loader.get_dask_array(
        frequency_interval=args.frequency_interval,
        stokes = args.stokes,
        apply_rfi_mask=args.apply_rfi_mask,
        rfi_mask_level=args.rfi_mask_level,
        rfi_mask_level0_percentage = args.rfi_mask_level0_percentage,
        interpolation_in_time = args.interpolation_in_time,
        interpolation_in_time_value = args.interpolation_in_time_value,
        interpolation_in_frequency = args.interpolation_in_frequency,
        interpolation_in_frequency_value = args.interpolation_in_frequency_value,
        verbose = args.verbose,
        log_infos = args.log_infos
    )


    args_list = [(lazy_loader, index_freq, time, data_final, False) for index_freq in range(len(frequencies))]
        
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_LS, args_list)

    f_LS = [result[0] for result in results]
    power_LS = [result[1] for result in results]

    if args.plot_results:
        plot_LS_periodogram(frequencies, f_LS, power_LS, args.output_directory)

    if args.save_as_hdf5:
        save_to_hdf(time,
                frequencies,
                data_final,
                f_LS,
                power_LS,
                args.output_directory,
                args.key_project,
                args.target)