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
import datetime

import logging


    # ============================================================= #
# ------------------- Logging configuration ------------------- #

def configure_logging(args):
    filename = f'{args.output_directory}/lazy_loading_data_LT{args.key_project}_{args.target}_stokes{args.stokes.upper()}.log'

    logging.basicConfig(
        #filename='outputs/lazy_loading_data_LT02.log',
        #filename = filename,
        # filemode='w',
        stream=sys.stdout,
        level=logging.INFO,
        # format='%(asctime)s -- %(levelname)s: %(message)s',
        # format='\033[1m%(asctime)s\033[0m | %(levelname)s: \033[34m%(message)s\033[0m',
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)
    return log



@numpy.vectorize
def datetime_to_timestamp(datetime_table):
    ### Function to return time in floating format (from a datetime object)
    return datetime_table.timestamp()

@numpy.vectorize
def timestamp_to_datetime(timestamp_table):
    ### Function to return time in datetime format (from a timestamp object)
    result = datetime.datetime.fromtimestamp(timestamp_table)
    return (result)


def read_hdf5_file(input_file, dataset=False, LS_dataset = True):
    """
    Reads the LS data from an HDF5 file.
    """
    with File(input_file, 'r') as file_hdf5:
        time_timestamp = numpy.array(file_hdf5['Time'])
        time_datetime = timestamp_to_datetime(time_timestamp)
        frequency_obs = numpy.array(file_hdf5['Frequency_Obs'])
        if LS_dataset:
            frequency_LS = numpy.array(file_hdf5['Frequency_LS'])
            power_LS = numpy.array(file_hdf5['power_LS'])
        
        key_project = file_hdf5['key_project'][()].decode('utf-8')
        target = file_hdf5['Target'][()].decode('utf-8')
        stokes = file_hdf5['Stokes'][()].decode('utf-8')    
        T_exoplanet = file_hdf5['T_exoplanet'][()]

        if dataset == True:
            data = numpy.array(file_hdf5['Dataset'])

    if (dataset == True):
        if (LS_dataset == True):
            return(time_datetime,
                frequency_obs,
                data,
                frequency_LS,
                power_LS,
                stokes,
                key_project,
                target,
                T_exoplanet
                )
        else:
            return(time_datetime,
                frequency_obs,
                data,
                stokes,
                key_project,
                target,
                T_exoplanet
                )
    else:
        if (LS_dataset == True):
            return(time_datetime,
            frequency_obs,
            frequency_LS,
            power_LS,
            stokes,
            key_project,
            target,
            T_exoplanet
            )
        else:
            return(time_datetime,
            frequency_obs,
            stokes,
            key_project,
            target,
            T_exoplanet
            )


def save_preliminary_data_to_hdf5(time,
                                  frequency,
                                  data,
                                  stokes,
                                  output_directory,
                                  key_project,
                                  target,
                                  T_exoplanet,
                                  extra_name = ''):
    
    """
    Saves preliminary data to disk as an HDF5 file.
    """
    with File(output_directory+'preliminary_data_Stokes-'+stokes+'_LT'+key_project+'_'+target+extra_name+'.hdf5', 'w') as output_file:
        output_file.create_dataset('Time', data = time)
        output_file['Time'].attrs.create('format', 'unix')
        output_file['Time'].attrs.create('units', 's')
        output_file.create_dataset('Dataset', data = data_final)
        output_file.create_dataset('Frequency_Obs', data=frequency)
        output_file['Frequency_Obs'].attrs.create('units', 'MHz')
        output_file.create_dataset('key_project', data=key_project)
        output_file.create_dataset('Target', data=target)
        output_file.create_dataset('T_exoplanet', data=T_exoplanet)
        output_file['T_exoplanet'].attrs.create('units', 'h')
        output_file.create_dataset('Stokes', data = stokes)       

def save_to_hdf(time,
                frequency_obs,
                data_final,
                frequency_LS,
                power_LS,
                stokes,
                output_directory,
                key_project,
                target,
                T_exoplanet,
                extra_name = ''):
    """
    Saves the data to disk as an HDF5 file.
    """
    with File(output_directory+'lomb_scargle_periodogram_Stokes-'+stokes+'_LT'+key_project+'_'+target+extra_name+'.hdf5', 'w') as output_file:
        output_file.create_dataset('Time', data = time)
        output_file['Time'].attrs.create('format', 'unix')
        output_file['Time'].attrs.create('units', 's')
        output_file.create_dataset('Dataset', data = data_final)
        output_file.create_dataset('Frequency_Obs', data=frequency_obs)
        output_file['Frequency_Obs'].attrs.create('units', 'MHz')
        output_file.create_dataset('Frequency_LS', data=frequency_LS)
        output_file['Frequency_LS'].attrs.create('units', 's')
        output_file.create_dataset('power_LS', data=power_LS)
        
        output_file.create_dataset('key_project', data=key_project)
        output_file.create_dataset('Target', data=target)
        output_file.create_dataset('T_exoplanet', data=T_exoplanet)
        output_file['T_exoplanet'].attrs.create('units', 'h')
        output_file.create_dataset('Stokes', data = stokes)

def plot_LS_periodogram(frequencies,
                        f_LS,
                        power_LS,
                        stokes,
                        output_directory,
                        background = False,
                        T_exoplanet = 1.769137786,
                        target = 'Jupiter',
                        key_project = '07',
                        figsize = None,
                        extra_name = '',
                        filename = None):
    """
    INPUT:
        - frequencies: Observation frequencies (in MHz)
        - f_LS: LombScargle frequencies (in Hertz)
        - power_LS: LombScargle power (shape : n_{frequencies}, n_{f_LS})
        - T_exoplanet: rotation period (in days) of the exoplanet at which a period should be seen in the periodogram. A vertical line will be plot at this period and half this period
        - target (str): name of the exoplant.
    OUTPUT:
        - png file of LombScargle periodograms (one per value in frequencies), saved in the output_directory directory
    """
    dpi = 500
    if figsize == None:
        figsize = (15,len(frequencies)*3)
    fig, axs = plt.subplots(nrows=len(frequencies), sharex=True, dpi=dpi, figsize = figsize)

    if target == 'Jupiter':
        T_io = 1.769137786
        T_jupiter = 9.9250/24
        T_synodique = (T_io*T_jupiter)/abs(T_io-T_jupiter)

    if len(frequencies) > 1:
        for index_freq in range(len(frequencies)):
            if background:
                bck = numpy.nanmean(f_LS)
                #sig = numpy.std(f_LS)
                f_LS = (f_LS-bck)#/sig
            axs[index_freq].plot(1/(f_LS[index_freq])/60/60, (power_LS[index_freq]))
            #plt.yscale('log')
            axs[index_freq].set_title(f'Frequency: {frequencies[index_freq]} MHz')
            if target == 'Jupiter':
                axs[index_freq].vlines([T_io*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{Io}$")
                axs[index_freq].vlines([T_io*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed", label = r"$\frac{1}{2} x T_{Io}$")
                axs[index_freq].vlines([T_jupiter*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g',label = r"$T_{Jup}$")
                axs[index_freq].vlines([T_jupiter*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g', linestyles="dashed",label = r"$\frac{1}{2} x T_{Io}$")
                axs[index_freq].vlines([T_synodique*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y',label = r"$T_{synodic}$")
                axs[index_freq].vlines([T_synodique*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y', linestyles="dashed",label = r"$\frac{1}{2} x T_{synodic}$")
            else:
                axs[index_freq].vlines([T_exoplanet*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{{target}}$")
                axs[index_freq].vlines([T_exoplanet*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed",label = r"$\frac{1}{2} x T_{{target}}$")
            axs[index_freq].xaxis.set_minor_locator(MultipleLocator(1))
            axs[index_freq].xaxis.set_major_locator(MultipleLocator(5))
            if index_freq == 0:
                axs[index_freq].legend()
        axs[index_freq].set_xlim([(T_exoplanet/10)*24,(T_exoplanet*2)*24])
        axs[index_freq].set_xlabel("Periodicity (Hours)")

    else:
        index_freq = 0
        axs.plot(1/(f_LS[index_freq])/60/60, (power_LS[index_freq]))
        axs.set_title(f'Frequency: {frequencies[index_freq]} MHz')
        if target == 'Jupiter':
            axs.vlines([T_io*24],          (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{Io}$")
            axs.vlines([T_io*24/2],        (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed", label = r"$\frac{1}{2} x T_{Io}$")
            axs.vlines([T_jupiter*24],     (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g',label = r"$T_{Jup}$")
            axs.vlines([T_jupiter*24/2],   (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g', linestyles="dashed",label = r"$\frac{1}{2} x T_{Io}$")
            axs.vlines([T_synodique*24],   (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y',label = r"$T_{synodic}$")
            axs.vlines([T_synodique*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y', linestyles="dashed",label = r"$\frac{1}{2} x T_{synodic}$")
        else:
            axs.vlines([T_exoplanet*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{{target}}$")
            axs.vlines([T_exoplanet*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed",label = r"$\frac{1}{2} x T_{{target}}$")
        axs.xaxis.set_minor_locator(MultipleLocator(1))
        axs.xaxis.set_major_locator(MultipleLocator(5))
        axs.legend()
        axs.set_xlim([(T_exoplanet/10)*24,(T_exoplanet*2)*24])
        axs.set_xlabel("Periodicity (Hours)")



    plt.tight_layout()
    #plt.show()
    if filename == None:
        filename = 'lomb_scargle_periodogram_Stokes-'+stokes+'_LT'+key_project+'_'+target+extra_name
    else:
        filename = filename.split('.')[0]
    plt.savefig(output_directory+filename+'.png', dpi = dpi, format = 'png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Calculation of Lomb Scargle Periodogram for observations of a given radio emitter (planet, exoplanet or star)")
    parser.add_argument('-key_project', dest = 'key_project', required = True, type = str, help = "NenuFAR Key Project number (02, or 07)")
    parser.add_argument('-target', dest = 'target', required = True, type = str, help = "Observation Target Name (planet, exoplanet or star)")
    parser.add_argument('--level_processing', dest = 'level_processing', type = str, default = 'L1', help = "Level of processing to be used")
    parser.add_argument('--main_directory_path', dest = 'root', default = './data/', type = str, help = "Main directory path where the observation are stored")
    parser.add_argument('--stokes', dest = 'stokes', default = 'V', type = str, help = "Stokes parameter to be study. Choices: I, V, V+, V-, Q, U, L.")
    parser.add_argument('--apply_rfi_mask', dest = 'apply_rfi_mask', default = False, action = 'store_true', help = "Apply RFI mask")
    parser.add_argument('--rfi_mask_level', dest = 'rfi_mask_level', default = None, type = int, help = "Level of the RFI mask to apply (needed if --apply_rfi_mask True). Option are 0, 1, 2, or 3")
    parser.add_argument('--rfi_mask_level0_percentage', dest = 'rfi_mask_level0_percentage', default = 10, type = float, help = "Percentage (i.e. threshold) of the RFI mask level to apply (needed if --apply_rfi_mask True and rfi_mask_level is 0). Values can be between 0 and 100 %")
    parser.add_argument('--interpolation_in_time', dest = 'interpolation_in_time', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_time_value', dest = 'interpolation_in_time_value', default = 1, type = float, help = "Value in second over which data need to be interpolated")
    parser.add_argument('--interpolation_in_frequency', dest = 'interpolation_in_frequency', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_frequency_value', dest = 'interpolation_in_frequency_value', default = 1, type = float, help = "Value in MegaHertz (MHz) over which data need to be interpolaed")
    parser.add_argument('--frequency_interval', dest = 'frequency_interval', nargs = 2, type = float, default = [10,90], help = "Half-open Minimal and Maximal (i.e., [Minimal to Maximal)) frequency range over which the Lomb Scargle analysis has to be done")
    parser.add_argument('--verbose', dest = 'verbose', default = False, action = 'store_true', help = "To print on screen the log infos")
    parser.add_argument('--log_infos', dest = 'log_infos', default = False, action = 'store_true', help = "To print on screen the dask computing info, and control graphics after computation")
    
    parser.add_argument('--lombscargle_function', dest = 'lombscargle_function', type = str, default = 'scipy', help = "LombScargle package to be used. Options are 'scipy' or 'astropy'")
    parser.add_argument('--normalize_LS', dest = 'normalize_LS', default = False, action = 'store_true', help = "Normalization of the Lomb-Scargle periodogram")
    parser.add_argument('--remove_background_to_LS', dest = 'background', default = False, action='store_true', help="Set True to remove a background to the Lomb Scargle plots (per LS frequency)")
    parser.add_argument('--save_as_hdf5', dest = 'save_as_hdf5', default = False, action = 'store_true', help = "To save results in an hdf5 file")
    parser.add_argument('--plot_results', dest = 'plot_results', default = False, action = 'store_true', help = "To plot and save results")
    parser.add_argument("--figsize", dest = 'figsize', nargs = 2, type = int, default = None, help = "Figure size")

    parser.add_argument('--plot_only', dest = 'plot_only', default = False, action = 'store_true', help = "Set this as True if you only want to plot the results from pre-calculated data stored in an hdf5 file")
    parser.add_argument('--reprocess_LS_periodogram', dest = 'reprocess_LS_periodogram', default = False, action = 'store_true', help = "Set this as True if you want to read from an hdf5 file data already rebinned, and re-process the Lomb Scargle calculation")
    parser.add_argument('--input_hdf5_file', dest = 'input_hdf5_file', default = None, type = str, help = "HDF5 file path containing pre-calculated data. Required if --plot_only or --reprocess_LS_periodogram is set as True")
    
    parser.add_argument('--output_directory', dest = 'output_directory', default = './', type = str, help = "Output directory where to save hdf5 and/or plots")
    parser.add_argument('--only_data_during_night', dest = 'only_data_during_night', default = False, action = 'store_true', help = "To select only data during night time")
    args = parser.parse_args()

    if (args.plot_only == False):
        if args.log_infos:
            log = configure_logging(args)
        else:
            log = None
        if args.reprocess_LS_periodogram == False:
            level_of_preprocessed = ''

            if args.key_project == '07':
                sub_path = "*/*/*/"
            else:
                sub_path = f"*/*/*/{args.level_processing}/"

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
                                        args.stokes,
                                        args.target,
                                        args.key_project,
                                        log
                                    )
            time, frequencies, data_final = lazy_loader.get_dask_array(
                frequency_interval = args.frequency_interval,
                stokes = args.stokes,
                apply_rfi_mask = args.apply_rfi_mask,
                rfi_mask_level = args.rfi_mask_level,
                rfi_mask_level0_percentage = args.rfi_mask_level0_percentage,
                interpolation_in_time = args.interpolation_in_time,
                interpolation_in_time_value = args.interpolation_in_time_value,
                interpolation_in_frequency = args.interpolation_in_frequency,
                interpolation_in_frequency_value = args.interpolation_in_frequency_value,
                verbose = args.verbose,
                log_infos = args.log_infos,
                output_directory = args.output_directory
            )

            lazy_loader.find_rotation_period_exoplanet()
            T_exoplanet = lazy_loader.exoplanet_period # in days

            extra_name = ''
            if args.apply_rfi_mask != None:
                if args.rfi_mask_level == 0:
                    extra_name = '_masklevel'+str(int(args.rfi_mask_level))+'_'+str(int(args.rfi_mask_level0_percentage))+'percents'
                else:
                    extra_name = '_masklevel'+str(int(args.rfi_mask_level))
            else:
                extra_name = '_nomaskapplied'
            extra_name = extra_name+'_'+f'{int(args.frequency_interval[0])}-{int(args.frequency_interval[1])}MHz_{args.lombscargle_function}LS_{args.normalize_LS}'


            if args.save_as_hdf5:
                save_preliminary_data_to_hdf5(time,
                                    frequencies,
                                    data_final,
                                    args.stokes,
                                    args.output_directory,
                                    args.key_project,
                                    args.target,
                                    T_exoplanet,
                                    extra_name = extra_name)
        
        elif args.reprocess_LS_periodogram == True:
        
            if args.input_hdf5_file == None:
                raise RuntimeError("An hdf5 file containing pre-calculated data needs to be given with the --input_hdf5_file argument if --plot_only is set as True")
        
            (time_datetime,
                frequencies,
                data_final,
                stokes,
                key_project,
                target,
                T_exoplanet
                ) = read_hdf5_file(args.input_hdf5_file, dataset=True, LS_dataset = False)
            
            time = datetime_to_timestamp(time_datetime)

            lazy_loader = LazyFITSLoader(None, None, 
                                        stokes,
                                        target,
                                        key_project,
                                        log
                                    )

            lazy_loader.find_rotation_period_exoplanet()

            extra_name = ''
            if args.apply_rfi_mask != None:
                if args.rfi_mask_level == 0:
                    extra_name = '_masklevel'+str(int(args.rfi_mask_level))+'_'+str(int(args.rfi_mask_level0_percentage))+'percents'
                else:
                    extra_name = '_masklevel'+str(int(args.rfi_mask_level))
            else:
                extra_name = '_nomaskapplied'
            extra_name = extra_name+'_'+f'{int(args.frequency_interval[0])}-{int(args.frequency_interval[1])}MHz_{args.lombscargle_function}LS_{args.normalize_LS}'

        if args.only_data_during_night:
            mask = []
            for index_time,itime in enumerate(time): 
                mask.append(((itime/(24*60*60)-int(itime/(24*60*60)))*24 < 6) or ((itime/(24*60*60)-int(itime/(24*60*60)))*24 > 18))
                time[mask] = 0
                data_final[mask,:] = 0

        args_list = [(
                    lazy_loader,
                    time,
                    20 * numpy.log10(data_final[:, index_freq]) if args.stokes.upper() in ('I', 'RM') else data_final[:, index_freq],
                    args.normalize_LS,
                    args.lombscargle_function,
                    args.log_infos)
                    for index_freq in range(len(frequencies))
                    ]

        #args_list = [(lazy_loader, index_freq, time, data_final, normalize_LS) for index_freq in range(len(frequencies))]

        #time = []
        #frequencies_ = []
        #data_final_ = []
        #f_LS_ = []
        #power_LS_ = []
        with multiprocessing.Pool() as pool:
            results = pool.map(calculate_LS, args_list)

        #if args.verbose:
        #    with Profiler() as prof, ResourceProfiler(dt=0.0025) as rprof, CacheProfiler() as cprof:
        #        with ProgressBar():
        #            time = time.compute()
        #        with ProgressBar():
        #            frequencies = frequencies.compute()
        #        with ProgressBar():
        #            data_final = data_final.compute()
        #        with ProgressBar():
        #            f_LS = [result[0].compute() for result in results]
        #        with ProgressBar():
        #            power_LS = [result[1].compute() for result in results]
        #    visualize([prof, rprof, cprof,])

        #else:
        #    time = time.compute()
        #    frequencies = frequencies.compute()
        #    data_final = data_final.compute()
        #    f_LS = [result[0].compute() for result in results]
        #    power_LS = [result[1].compute() for result in results]

        f_LS = [result[0] for result in results]
        power_LS = [result[1] for result in results]
            
       
         # Concatenating of arrays over observation
        #time = numpy.concatenate(time_, axis=0)
        #if numpy.max(frequencies_[-1]) - numpy.max(frequencies_[0]) > 1e-8:
        #    raise ValueError("Frequency observation are not the same. Something needs to be modified in the function. Exiting.")
        #else:
        #    frequencies = frequencies_[0]
        
        #data_final = numpy.concatenate(data_final_, axis=0)
        #f_LS = numpy.concatenate(f_LS_, axis=0)
        #power_LS = numpy.concatenate(power_LS_, axis=0)




        #print(f'time: {time.shape}')
        #print(f'frequencies: {frequencies.shape}')
        #print(f'data_final: {data_final.shape}')
        #print(f'f_LS: {f_LS.shape}')
        #print(f'power_LS: {power_LS.shape}')

        
        if args.save_as_hdf5:
            save_to_hdf(time,
                    frequencies,
                    data_final,
                    f_LS,
                    power_LS,
                    args.stokes,
                    args.output_directory,
                    args.key_project,
                    args.target,
                    T_exoplanet,
                    extra_name = extra_name)

        if args.plot_results:
            plot_LS_periodogram(frequencies,
                                f_LS,
                                power_LS,
                                args.stokes,
                                args.output_directory,
                                background = args.background,
                                T_exoplanet = T_exoplanet,
                                target = args.target,
                                key_project = args.key_project,
                                figsize = args.figsize,
                                extra_name = extra_name)

       
    if args.plot_only:
        if args.input_hdf5_file == None:
            raise RuntimeError("An hdf5 file containing pre-calculated data needs to be given with the --input_hdf5_file argument if --plot_only is set as True")
        
        (time_datetime, frequency_obs, frequency_LS, power_LS, stokes, key_project, target, T_exoplanet) = read_hdf5_file(args.input_hdf5_file, dataset = False, LS_dataset = True)
        plot_LS_periodogram(frequency_obs,
                            frequency_LS,
                            power_LS,
                            stokes,
                            args.output_directory,
                            background = args.background,
                            T_exoplanet = T_exoplanet,
                            target = target,
                            key_project = key_project,
                            figsize = args.figsize,
                            filename = args.input_hdf5_file.split('.')[0].split('/')[-1])
