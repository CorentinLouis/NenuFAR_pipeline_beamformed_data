import numpy
import numpy.ma as ma
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

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
    filename = f'{args.output_directory}/lazy_loading_data_LT{args.key_project}_{args.target}_stokes{args.stokes.upper()}'
    if args.apply_rfi_mask:
        filename = filename+f'_rfimasklevel{args.rfi_mask_level}'
    filename = filename+'.log'

    logging.basicConfig(
        #filename='outputs/lazy_loading_data_LT02.log',
        filename = filename,
        filemode='w',
        #stream=sys.stdout,
        level=logging.INFO,
        # format='%(asctime)s -- %(levelname)s: %(message)s',
        # format='\033[1m%(asctime)s\033[0m | %(levelname)s: \033[34m%(message)s\033[0m',
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)
    return log

def get_planet_target_type(planet_name, list_type_target):
    for row in list_type_target:
        if row['name'].upper() == planet_name.upper():
            return row['target_type']

import csv
def read_csv_to_dict(file_path):
    data_dict = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            data_dict.append(row)
    return data_dict




@numpy.vectorize
def datetime_to_timestamp(datetime_table):
    ### Function to return time in floating format (from a datetime object)
    result = Time(datetime_table, format="datetime").unix
#    result = datetime_table.timestamp()
    return (result)

@numpy.vectorize
def timestamp_to_datetime(timestamp_table):
    ### Function to return time in datetime format (from a timestamp object)
    result = Time(timestamp_table, format="unix").datetime
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
        T_star = file_hdf5['T_star'][()]

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
                T_exoplanet,
                T_star
                )
        else:
            return(time_datetime,
                frequency_obs,
                data,
                stokes,
                key_project,
                target,
                T_exoplanet,
                T_star
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
            T_exoplanet,
            T_star
            )
        else:
            return(time_datetime,
            frequency_obs,
            stokes,
            key_project,
            target,
            T_exoplanet,
            T_star
            )


def save_preliminary_data_to_hdf5(time,
                                  frequency,
                                  data,
                                  stokes,
                                  output_directory,
                                  key_project,
                                  target,
                                  T_exoplanet,
                                  T_star,
                                  extra_name = ''):
    
    """
    Saves preliminary data to disk as an HDF5 file.
    """
    with File(output_directory+'preliminary_data_Stokes-'+stokes+'_LT'+key_project+'_'+target+extra_name+'.hdf5', 'w') as output_file:
        output_file.create_dataset('Time', data = time)
        output_file['Time'].attrs.create('format', 'unix')
        output_file['Time'].attrs.create('units', 's')
        output_file.create_dataset('Dataset', data = data)
        output_file.create_dataset('Frequency_Obs', data=frequency)
        output_file['Frequency_Obs'].attrs.create('units', 'MHz')
        output_file.create_dataset('key_project', data=key_project)
        output_file.create_dataset('Target', data=target)
        output_file.create_dataset('T_exoplanet', data=T_exoplanet)
        output_file.create_dataset('T_star', data=T_star)
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
                T_star,
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
        output_file.create_dataset('T_star', data=T_star)
        output_file['T_exoplanet'].attrs.create('units', 'h')
        output_file.create_dataset('Stokes', data = stokes)

def plot_LS_periodogram(frequencies,
                        f_LS,
                        power_LS,
                        stokes,
                        output_directory,
                        background = False,
                        T_exoplanet = 1.769137786,
                        T_star = 0.995,
                        target = 'Jupiter',
                        key_project = '07',
                        figsize = None,
                        x_limits = None,
                        extra_name = '',
                        filename = None,
                        log = None):
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
        figsize = (25,8)
    
    #plt.show()
    if filename == None:
        filename = 'lomb_scargle_periodogram_Stokes-'+stokes+'_LT'+key_project+'_'+target+extra_name
    else:
        filename = filename.split('.')[0]
        
    pdf_file = PdfPages(output_directory+filename+'.pdf')

    if target == 'Jupiter':
        T_io = 1.769137786
        T_jupiter = 9.9250/24
        T_synodique = (T_io*T_jupiter)/abs(T_io-T_jupiter)

    target_ = target.split('_')
    target = ''
    for index, itarget in enumerate(target_): 
        if index == 0: 
            target = f'{itarget}' 
        else: 
            target = f'{target} {itarget}' 

    for index_freq in range(len(frequencies)):
        if log != None:
            log.info(f'Plotting frequency {index_freq+1} / {len(frequencies)}')
        #index_freq = 0
        if background:
            bck = numpy.nanmean(f_LS)
            #sig = numpy.std(f_LS)
            f_LS = (f_LS-bck)#/sig
        fig, axs = plt.subplots(dpi=dpi, figsize = figsize)
        axs.plot(1/(f_LS[index_freq])/60/60, (power_LS[index_freq]))
        axs.set_title(f'Frequency: {frequencies[index_freq]} MHz')


        # Enable major ticks automatically (default behavior)
        axs.tick_params(which='major', length=10)

        # Enable minor ticks with AutoMinorLocator
        axs.minorticks_on()
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.yaxis.set_minor_locator(AutoMinorLocator())

        # Customize minor ticks
        axs.tick_params(which='minor', length=5)


        if target == 'Jupiter':
            axs.vlines([T_io*24],          (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='r', label = r"$T_{Io}$")
            axs.vlines([T_io*24/2],        (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='r', linestyles="dashed", label = r"$\frac{1}{2} x T_{Io}$")
            axs.vlines([T_jupiter*24],     (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='g',label = r"$T_{Jup}$")
            axs.vlines([T_jupiter*24/2],   (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='g', linestyles="dashed",label = r"$\frac{1}{2} x T_{Jup}$")
            axs.vlines([T_synodique*24],   (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='y',label = r"$T_{synodic}$")
            axs.vlines([T_synodique*24/2], (power_LS[index_freq]).max(), (power_LS[index_freq]).max()*2, colors='y', linestyles="dashed",label = r"$\frac{1}{2} x T_{synodic}$")
        else:
            axs.vlines([T_star*24],     (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g',label = r"$T_{\mathrm{"+f'{target}'+"}}$")
            axs.vlines([T_star*24/2],   (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='g', linestyles="dashed",label = r"$\frac{1}{2} x T_{\mathrm{"+f'{target}'+"}}$")
            if log:
                log.info(f'{type(T_exoplanet)}')
                log.info(f'{isinstance(T_exoplanet,numpy.ndarray)}')
                log.info(f'{isinstance(T_exoplanet,numpy.float64)}')

            if isinstance(T_exoplanet,numpy.ndarray):
                for index, i_exoplanet in enumerate(T_exoplanet):
                    axs.vlines([i_exoplanet*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{\mathrm{exoplanet "+f'{index}'+"}}$")
                    axs.vlines([i_exoplanet*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed",label = r"$\frac{1}{2} x T_{\mathrm{exoplanet "+f'{index}'+"}}$")
                    if i_exoplanet-T_star !=0:
                        T_synodique = (i_exoplanet*T_star)/abs(i_exoplanet-T_star)
                        axs.vlines([T_synodique*24],   (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y',label = r"$T_{\mathrm{synodic exoplanet "+f'{index}'+"}}$")
                        axs.vlines([T_synodique*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y', linestyles="dashed",label = r"$\frac{1}{2} x T_{\mathrm{synodic exoplanet "+f'{index}'+"}}$")
            else:
                axs.vlines([T_exoplanet*24], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', label = r"$T_{\mathrm{exoplanet}}$")
                axs.vlines([T_exoplanet*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='r', linestyles="dashed",label = r"$\frac{1}{2} x T_{\mathrm{exoplanet}}$")
                if T_exoplanet-T_star !=0:
                    T_synodique = (T_exoplanet*T_star)/abs(T_exoplanet-T_star)
                    axs.vlines([T_synodique*24],   (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y',label = r"$T_{synodic}$")
                    axs.vlines([T_synodique*24/2], (power_LS[index_freq]).min(), (power_LS[index_freq]).max(), colors='y', linestyles="dashed",label = r"$\frac{1}{2} x T_{synodic}$")

        #axs.xaxis.set_minor_locator(MultipleLocator(1))
        #axs.xaxis.set_major_locator(MultipleLocator(5))

        axs.legend()
        if x_limits == None:
            #axs.set_xlim([(numpy.mean(T_exoplanet)/10)*24,(numpy.mean(T_exoplanet)*2)*24])
            axs.set_xlim(numpy.min(1/(f_LS[index_freq])/60/60), numpy.max(1/(f_LS[index_freq])/60/60))
        else:
            axs.set_xlim(x_limits[0], x_limits[-1])
        axs.set_xlabel("Periodicity (Hours)")
        plt.tight_layout()
        pdf_file.savefig()
        plt.close()
        


    pdf_file.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Calculation of Lomb Scargle Periodogram for observations of a given radio emitter (planet, exoplanet or star)")
    parser.add_argument('-key_project', dest = 'key_project', required = True, type = str, help = "NenuFAR Key Project number (02, or 07)")
    parser.add_argument('-target', dest = 'target', required = True, type = str, help = "Observation Target Name (planet, exoplanet or star)")
    parser.add_argument('--level_processing', dest = 'level_processing', type = str, default = 'L1', help = "Level of processing to be used")
    parser.add_argument('--main_directory_path', dest = 'root', default = './data/', type = str, help = "Main directory path where the observation are stored")
    parser.add_argument('--stokes', dest = 'stokes', default = 'V', type = str, help = "Stokes parameter to be study. Choices: I, V, V+, V-, Q, U, L.")
    parser.add_argument('--threshold', dest = 'threshold', default = None, type = float, help = "Threshold to be applied for LS")
    parser.add_argument('--apply_rfi_mask', dest = 'apply_rfi_mask', default = False, action = 'store_true', help = "Apply RFI mask")
    parser.add_argument('--rfi_mask_level', dest = 'rfi_mask_level', default = None, type = int, help = "Level of the RFI mask to apply (needed if --apply_rfi_mask True). Option are 0, 1, 2, or 3")
    parser.add_argument('--rfi_mask_level0_percentage', dest = 'rfi_mask_level0_percentage', default = 10, type = float, help = "Percentage (i.e. threshold) of the RFI mask level to apply (needed if --apply_rfi_mask True and rfi_mask_level is 0). Values can be between 0 and 100 %")
    parser.add_argument('--off_beams', dest = 'off_beams', default = False, action = 'store_true', help = "Set as True to do the analysis on the off beam(s) observation")
    parser.add_argument('--interpolation_in_time', dest = 'interpolation_in_time', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_time_value', dest = 'interpolation_in_time_value', default = 1, type = float, help = "Value in second over which data need to be interpolated")
    parser.add_argument('--interpolation_in_frequency', dest = 'interpolation_in_frequency', default = False, action = 'store_true', help = "Interpolate in time")
    parser.add_argument('--interpolation_in_frequency_value', dest = 'interpolation_in_frequency_value', default = 1, type = float, help = "Value in MegaHertz (MHz) over which data need to be interpolaed")
    parser.add_argument('--frequency_interval', dest = 'frequency_interval', nargs = 2, type = float, default = [10,90], help = "Half-open Minimal and Maximal (i.e., [Minimal to Maximal)) frequency range over which the Lomb Scargle analysis has to be done")
    parser.add_argument('--verbose', dest = 'verbose', default = False, action = 'store_true', help = "To print on screen the log infos")
    parser.add_argument('--log_infos', dest = 'log_infos', default = False, action = 'store_true', help = "To print on screen the dask computing info, and control graphics after computation")
    
    parser.add_argument('--lombscargle_calculation', dest = 'lombscargle_calculation', default = False, action = 'store_true', help = "Set this as False if you don't want to calculate the lomb scargle periodogram. Only processed data will be saved.")
    parser.add_argument('--periodicity_stacking_calculation', dest = 'periodicity_stacking_calculation', default = False, action = 'store_true', help = "Set this as True if you want to calculate the stacked (per exoplanet(s) revolution and star rotation periods) timeseries.")

    parser.add_argument('--lombscargle_function', dest = 'lombscargle_function', type = str, default = 'scipy', help = "LombScargle package to be used. Options are 'scipy' or 'astropy'")
    parser.add_argument('--normalize_LS', dest = 'normalize_LS', default = False, action = 'store_true', help = "Normalization of the Lomb-Scargle periodogram")
    parser.add_argument('--remove_background_to_LS', dest = 'background', default = False, action='store_true', help="Set True to remove a background to the Lomb Scargle plots (per LS frequency)")
    parser.add_argument('--save_as_hdf5', dest = 'save_as_hdf5', default = False, action = 'store_true', help = "To save results in an hdf5 file")
    parser.add_argument('--plot_results', dest = 'plot_results', default = False, action = 'store_true', help = "To plot and save results")
    parser.add_argument("--figsize", dest = 'figsize', nargs = 2, type = int, default = None, help = "Figure size")
    parser.add_argument('--plot_x_lim', dest = 'plot_x_lim', default = None, nargs = 2, type = float, help = "x limits for Lomb-Scargle periodogram plot")

    parser.add_argument('--plot_only', dest = 'plot_only', default = False, action = 'store_true', help = "Set this as True if you only want to plot the results from pre-calculated data stored in an hdf5 file")
    parser.add_argument('--reprocess_LS_periodogram', dest = 'reprocess_LS_periodogram', default = False, action = 'store_true', help = "Set this as True if you want to read from an hdf5 file data already rebinned, and re-process the Lomb Scargle calculation")
    parser.add_argument('--input_hdf5_file', dest = 'input_hdf5_file', default = None, type = str, help = "HDF5 file path containing pre-calculated data. Required if --plot_only or --reprocess_LS_periodogram is set as True")
    parser.add_argument('--beam_number', dest = 'beam_number', default = None, type = int, help = "Beam number to be reprocessed in case --reprocess_LS_periodogram is set as True")
    parser.add_argument('--output_directory', dest = 'output_directory', default = './', type = str, help = "Output directory where to save hdf5 and/or plots")
    parser.add_argument('--only_data_during_night', dest = 'only_data_during_night', default = False, action = 'store_true', help = "To select only data during night time")
    args = parser.parse_args()
    

    if args.periodicity_stacking_calculation:
        args.lombscargle_calculation = False

    if (args.plot_only == False):
        if args.log_infos:
            log = configure_logging(args)
        else:
            log = None
        if args.reprocess_LS_periodogram == False:
            if args.key_project == '07':
                sub_path = "*/*/*/"
                target_type = 'star'
                beam_on = ['0','1']
            else:
                filename_list_type_target = f'{args.root}/list_type_target.txt'
                list_type_target = read_csv_to_dict(filename_list_type_target)
                target_type = get_planet_target_type(args.target, list_type_target)

                sub_path = f"*/*/*/{args.level_processing}/"

                if target_type == 'star':
                    beam_on = ['0','1']
                    beam_off = ['2', '3']
                elif target_type == 'exoplanet':
                    beam_on = ['0']
                    beam_off = ['1','2','3']
                else:
                    if args.log_infos:
                        log.info(f"It seems that the target you are looking for isn't in the '{filename_list_type_target}' file you provide. Please check Target name and/or add the target type ('star' or 'exoplanet') info to the file.")
                    raise RuntimeError(f"It seems that the target you are looking for isn't in the '{filename_list_type_target}' file you provide. Please check Target name and/or add the target type ('star' or 'exoplanet') info to the file.")

            if args.off_beams:
                beam_list = beam_off
            else:
                beam_list = beam_on
            
            if (target_type == 'star') or ((target_type == 'exoplanet') and (args.off_beams == False)) :
                if args.apply_rfi_mask and args.rfi_mask_level > 0:
                    rfi_fits_file_paths = [[
                            filename
                            for beam_number in beam_list
                                for filename in glob.iglob(
                                    f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*_{beam_number}.rfi*.fits',
                                recursive=True
                            )
                        ]]
                    
                    data_fits_file_paths = [
                                            [
                                                ifile.split('rfimask')[0] + 'spectra' + ifile.split('rfimask')[-1]
                                                for ifile in beam_files
                                            ]
                                            for beam_files in rfi_fits_file_paths
                                            ]
                else:       
                    data_fits_file_paths = [[
                            filename
                            for beam_number in beam_list
                                for filename in glob.iglob(
                                    f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*_{beam_number}.spectra*.fits',
                                    recursive=True
                            )
                        ]]

                    rfi_fits_file_paths = [[]]
            else:  #if ((target_type == 'exoplanet') and (args.off_beams == True)) 
                if args.apply_rfi_mask and args.rfi_mask_level > 0:
                    rfi_fits_file_paths = {
                                            beam_number: [
                                                        filename
                                                        for filename in glob.iglob(
                                                            f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*_{beam_number}.rfi*.fits',
                                                            recursive=True
                                                            )
                                                        ]
                                            for beam_number in beam_list
                                        }
                    rfi_fits_file_paths = [rfi_fits_file_paths[beam_number] for beam_number in beam_list]
                    
                    data_fits_file_paths = [
                                            [
                                                ifile.split('rfimask')[0] + 'spectra' + ifile.split('rfimask')[-1]
                                                for ifile in beam_files
                                            ]
                                            for beam_files in rfi_fits_file_paths
                                            ]

                else:       
                    data_fits_file_paths = {
                                            beam_number: [
                                                        filename
                                                        for filename in glob.iglob(
                                                            f'{args.root}/*{args.key_project}/{sub_path}*{args.target.upper()}*_{beam_number}.spectra*.fits',
                                                            recursive=True
                                                            )
                                                        ]
                                            for beam_number in beam_list
                                        }
                    data_fits_file_paths = [data_fits_file_paths[beam_number] for beam_number in beam_list]
                    rfi_fits_file_paths = [[], [], []]
            for i_beam in range(len(data_fits_file_paths)):
                if args.log_infos:
                    if args.off_beams:
                        log.info(f"Starting reading files for OFF beam number {beam_list[i_beam]}")
                        log.info(f"{len(data_fits_file_paths[i_beam])} OFF beam files will be read (x2 if users asked for RFI mask > 0 to be removed)")
                    else:
                        log.info(f"{len(data_fits_file_paths[i_beam])} ON beam files will be read (x2 if users asked for RFI mask > 0 to be removed)")
                    
                
                lazy_loader = LazyFITSLoader(data_fits_file_paths[i_beam], rfi_fits_file_paths[i_beam],
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
                T_star = lazy_loader.star_period

                extra_name = ''
                if args.off_beams:
                    beam_type = 'OFF'
                else:
                    beam_type = 'ON'
                if (target_type == 'star') or ((target_type == 'exoplanet') and (args.off_beams == False)):
                    beam_number = ''
                else:
                    beam_number = f'{beam_list[i_beam]}'
                
                if args.apply_rfi_mask != False:
                    if args.rfi_mask_level == 0:
                        extra_name = '_masklevel'+str(int(args.rfi_mask_level))+'_'+str(int(args.rfi_mask_level0_percentage))+'percents'
                    else:
                        extra_name = '_masklevel'+str(int(args.rfi_mask_level))
                else:
                    extra_name = '_nomaskapplied'
                extra_name = extra_name+'_'+f'{int(args.frequency_interval[0])}-{int(args.frequency_interval[1])}MHz_{beam_type}{beam_number}'

                if args.save_as_hdf5:
                    save_preliminary_data_to_hdf5(time,
                                        frequencies,
                                        data_final,
                                        args.stokes,
                                        args.output_directory,
                                        args.key_project,
                                        args.target,
                                        T_exoplanet,
                                        T_star,
                                        extra_name = extra_name)


                if args.lombscargle_calculation:
                    extra_name_LS = extra_name+f'_{args.lombscargle_function}LS_{args.normalize_LS}'

                    if args.only_data_during_night:
                        len_former_time = len(time)
                        mask = ((time/(24*60*60)-(time/(24*60*60)).astype(int))*24 > 4) * ((time/(24*60*60)-(time/(24*60*60)).astype(int))*24 < 22) #(* is and, + is or)
                        mask_2D = numpy.repeat(mask[:, None], data_final.shape[1], axis = 1)
                        time = time[mask == 0]
                        data_final = data_final[mask == 0,:]
                        if args.log_infos:
                            log.info(f"{len(time)} / {len_former_time} are selected for this time observation window")

                    args_list = [(
                                lazy_loader,
                                time,
                                20 * numpy.log10(data_final[:, index_freq]) if args.stokes.upper() in ('I', 'RM') else data_final[:, index_freq],
                                args.threshold,
                                args.normalize_LS,
                                args.lombscargle_function,
                                args.log_infos)
                                for index_freq in range(len(frequencies))
                                ]


                    with multiprocessing.Pool() as pool:
                        results = pool.map(calculate_LS, args_list)

                    f_LS = [result[0] for result in results]
                    power_LS = [result[1] for result in results]

                
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
                                T_star,
                                extra_name = extra_name_LS)


                    if args.plot_results:
                        plot_LS_periodogram(frequencies,
                                            f_LS,
                                            power_LS,
                                            args.stokes,
                                            args.output_directory,
                                            background = args.background,
                                            T_exoplanet = T_exoplanet,
                                            T_star = T_star,
                                            target = args.target,
                                            key_project = args.key_project,
                                            figsize = args.figsize,
                                            extra_name = extra_name_LS,
                                            x_limits = args.plot_x_lim, 
                                            log = log)
                
                if args.periodicity_stacking_calculation:
                    extra_name_PS = extra_name+f'_stacking_by_revolution_period'
                    # need to take the exoplanet revolution period
                    T_exoplanet = lazy_loader.exoplanet_period*24 # in hours
                    # At some point it'll be needed to differentiate cases where T_exoplanet is a numpy.ndarray, or a numpy.float
                    # then stack data_final by this period, using time
                    time_hours = time_unix / 3600
                    # Calculate the phase of the exoplanet
                    phase = time_hours % T_exoplanet

                    # Sort the data based on the phase
                    sorted_indices = np.argsort(phase)
                    sorted_phase = phase[sorted_indices]
                    sorted_data = data[sorted_indices]
                    # Stack the data based on the phase of the exoplanet
                    #stacked_data = []
                    #unique_phases = np.unique(sorted_phase)
                    #for phase_value in unique_phases:
                    #    phase_data = sorted_data[sorted_phase == phase_value]
                    #    stacked_data.append(phase_data)
                    

                    # Calculate the phase bins
                    phase_bins = np.linspace(0, T_exoplanet, num=T_exoplanet*24)  # Adjust the number of bins as needed


                    # Compute the sum of intensity encountered at each phase
                    stack_data = np.zeros_like(phase_bins)

                    # Iterate over each phase bin and accumulate the intensity
                    for i in range(len(phase_bins) - 1):
                        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                        stack_data[i] = np.sum(data[mask])

            




        elif args.reprocess_LS_periodogram == True:

            if args.off_beams:
                beam_type = 'OFF'
            else:
                beam_type = 'ON'

                
            if args.input_hdf5_file == None:
                raise RuntimeError("An hdf5 file containing pre-calculated data needs to be given with the --input_hdf5_file argument if --reprocess_LS_periodogram is set as True")
            if args.beam_number == None:
                raise RuntimeError("An beam number needs to be given with the --beam_number argument if --reprocess_LS_periodogram is set as True")
            (time_datetime,
                frequencies,
                data_final,
                stokes,
                key_project,
                target,
                T_exoplanet,
                T_star
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
            extra_name = extra_name+'_'+f'{int(args.frequency_interval[0])}-{int(args.frequency_interval[1])}MHz_{beam_type}{args.beam_number}'

            if args.lombscargle_calculation:
                extra_name = extra_name+f'_{args.lombscargle_function}LS_{args.normalize_LS}'
                if args.only_data_during_night:
                    len_former_time = len(time)
                    mask = ((time/(24*60*60)-(time/(24*60*60)).astype(int))*24 > 4) * ((time/(24*60*60)-(time/(24*60*60)).astype(int))*24 < 22) #(* is and, + is or)
                    mask_2D = numpy.repeat(mask[:, None], data_final.shape[1], axis = 1)
                    time = time[mask == 0]
                    data_final = data_final[mask == 0,:]
                    if args.log_infos:
                        log.info(f"{len(time)} / {len_former_time} are selected for this time observation window")

                args_list = [(
                            lazy_loader,
                            time,
                            20 * numpy.log10(data_final[:, index_freq]) if args.stokes.upper() in ('I', 'RM') else data_final[:, index_freq],
                            args.threshold,
                            args.normalize_LS,
                            args.lombscargle_function,
                            args.log_infos)
                            for index_freq in range(len(frequencies))
                            ]


                with multiprocessing.Pool() as pool:
                    results = pool.map(calculate_LS, args_list)

                f_LS = [result[0] for result in results]
                power_LS = [result[1] for result in results]
                
        
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
                            T_star,
                            extra_name = extra_name)

                if args.plot_results:
                    extra_name_LS = extra_name+f'_{args.lombscargle_function}LS_{args.normalize_LS}'
                    plot_LS_periodogram(frequencies,
                                    f_LS,
                                    power_LS,
                                    args.stokes,
                                    args.output_directory,
                                    background = args.background,
                                    T_exoplanet = T_exoplanet,
                                    T_star = T_star,
                                    target = args.target,
                                    key_project = args.key_project,
                                    figsize = args.figsize,
                                    extra_name = extra_name,
                                    x_limits = args.plot_x_lim, 
                                    log = log)

   
    

    if args.plot_only:
        if args.input_hdf5_file == None:
            raise RuntimeError("An hdf5 file containing pre-calculated data needs to be given with the --input_hdf5_file argument if --plot_only is set as True")
        

        if args.log_infos:
            log = configure_logging(args)
        else:
            log = None
        if args.lombscargle_calculation:
            (time_datetime, frequency_obs, frequency_LS, power_LS, stokes, key_project, target, T_exoplanet, T_star) = read_hdf5_file(args.input_hdf5_file, dataset = False, LS_dataset = True)
            plot_LS_periodogram(frequency_obs,
                            frequency_LS,
                            power_LS,
                            stokes,
                            args.output_directory,
                            background = args.background,
                            T_exoplanet = T_exoplanet,
                            T_star = T_star,
                            target = target,
                            key_project = key_project,
                            figsize = args.figsize,
                            x_limits = args.plot_x_lim, 
                            filename = args.input_hdf5_file.split('.')[0].split('/')[-1],
                            log = log)
