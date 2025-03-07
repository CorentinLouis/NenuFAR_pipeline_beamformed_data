import glob
import os
import numpy
import spacepy.pycdf
import datetime
from astropy.time import Time

# Create a list of ExPRES CDF files matching the specified pattern for years 2019 to 2024, for a given source (e.g., Io), for a given B_Model (e.g. jrmo09)
#Example usage
#base_dir = "~/Volumes/kronos/serpe/data/earth/"  # Change this to your actual base directory path
#cdf_files = create_cdf_file_list(base_dir)
import glob
import os

@numpy.vectorize
def datetime_to_timestamp(datetime_table):
    ### Function to return time in floating format (from a datetime object)
    return Time(datetime_table, format="datetime").unix

@numpy.vectorize
def timestamp_to_datetime(timestamp_table):
    ### Function to return time in datetime format (from a timestamp object)
    result = Time(timestamp_table, format="unix").datetime
    return (result)


def create_cdf_file_list(base_dir, years=(2019, 2025), source='io', B_model='jrm09', version='v01'):
    """
    Generates a sorted list of CDF file paths from subdirectories based on the specified pattern.
    
    Args:
        base_dir (str): The base directory containing the data for the years[0]-years[1].
        years (tuple): The range of years to explore.
        source (str): The source, default='io'. Options include 'io', 'europa', 'ganymede', or '' for all.
        B_model (str): The magnetic field model, default='jrm09'.
        version (str): The version of the files to search for, default='v01'.
        
    Returns:
        list of str: A sorted list of file paths matching the pattern.
    """
    cdf_files = []
    
    # Expand base directory if it uses ~
    base_dir = os.path.expanduser(base_dir)
    
    for year in range(years[0], years[1] + 1):
        print(f"Searching in year: {year}")
        for month in range(1, 13):
            # Construct the directory path
            dir_path = os.path.join(base_dir, str(year), f"{month:02d}")
            
            # Check if the directory exists
            if not os.path.exists(dir_path):
                print(f"Directory does not exist: {dir_path}")
                continue
            
            # Create the pattern to match the CDF files
            pattern = os.path.join(dir_path, f"*{source.lower()}_{B_model.lower()}*{version.lower()}.cdf")
            print(f"Using pattern: {pattern}")
            
            # Add files matching the pattern to the list
            matched_files = glob.glob(pattern)
            if matched_files:
                print(f"Found {len(matched_files)} files in {dir_path}")
            else:
                print(f"No files found in {dir_path} matching pattern: {pattern}")
            
            cdf_files.extend(matched_files)
    
    # Sort the list of file paths
    cdf_files = sorted(cdf_files)
    
    if not cdf_files:
        print("No CDF files found for the specified criteria.")
    else:
        print(f"Total CDF files found: {len(cdf_files)}")
    
    return cdf_files





# -*- coding: utf-8 -*-
import spacepy.pycdf
import numpy as np
import h5py

# Example usage:
# freq_table, freq_label, epoch, polarization, theta = extract_cdf_data(cdf_files, save_as_hdf5="output_data.hdf5")
def extract_cdf_data(cdf_files, save_as_hdf5=None):
    """
    Extracts data from a collection of CDF files and concatenates it into arrays.
    Optionally saves the concatenated data to an HDF5 file.
    
    Args:
        cdf_files (list of str): A list of file paths to CDF files.
        save_as_hdf5 (str, optional): The path to save the output as an HDF5 file. If None, data is not saved.
        
    Returns:
        tuple: A tuple containing:
            - freq_table (numpy.ndarray): The concatenated Frequency array from all CDFs.
            - freq_label (numpy.ndarray): The concatenated Frequency Label array from all CDFs.
            - epoch (numpy.ndarray): Concatenated Epoch array from all CDFs.
            - polarization (numpy.ndarray): Concatenated Polarization array from all CDFs.
            - theta (numpy.ndarray): Concatenated Theta array from all CDFs.
    """
    # Initialize empty lists to store concatenated data
    freq_label_all = []
    freq_table_all = []
    epoch_all = []
    polarization_all = []
    theta_all = []

    freq_table = None
    # Loop through the CDF files
    for cdf_path in cdf_files:
        # Open the CDF file
        with spacepy.pycdf.CDF(cdf_path) as cdf:
            # Extract the required data
            if freq_table is None:
                freq_label = cdf['Freq_Label'][:]
                freq_table = cdf['Frequency'][:]
            
            epoch_datetime = cdf['Epoch'][:]
            epoch = datetime_to_timestamp(epoch_datetime)
            polarization = cdf['Polarization'][:]
            theta = cdf['Theta'][:]

            # Append to the respective lists
            epoch_all.append(epoch)
            polarization_all.append(polarization)
            theta_all.append(theta)

    # Convert the lists to numpy arrays and concatenate over the time dimension
    freq_label_combined = np.concatenate(freq_label_all, axis=0).astype('S')
    freq_table_combined = np.concatenate(freq_table_all, axis=0)
    epoch_combined = np.concatenate(epoch_all, axis=0)
    polarization_combined = np.concatenate(polarization_all, axis=0)
    theta_combined = np.concatenate(theta_all, axis=0)


    # Save to HDF5 if the option is provided
    if save_as_hdf5:
        with h5py.File(save_as_hdf5, 'w') as h5f:
            h5f.create_dataset('Frequency', data=freq_table_combined)
            h5f.create_dataset('Freq_Label', data=freq_label_combined)
            h5f.create_dataset('Epoch', data=epoch_combined)
            h5f.create_dataset('Polarization', data=polarization_combined)
            h5f.create_dataset('Theta', data=theta_combined)
        print(f"Data saved to HDF5 file: {save_as_hdf5}")

    return freq_table_combined, freq_label_combined, epoch_combined, polarization_combined, theta_combined


# -*- coding: utf-8 -*-
import h5py
import numpy as np


def read_hdf5_data(file_paths):
    """
    Reads data from one or more HDF5 files and returns the datasets.
    
    Args:
        file_paths (str or list): Path(s) to the HDF5 file(s).
        
    Returns:
        dict: A dictionary with the concatenated data for most variables, 
              and 'Frequency' as a separate entry since it's the same for all files.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    data = {
        "Epoch": [],
        "Polarization": [],
        "Theta": [],
    }
    freq_table = None  # To store Frequency (not concatenated)
    
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            print(f"Processing file '{file_path}' with datasets:")
        
            # Read frequency once
            if freq_table is None:
                freq_table = h5f['Frequency'][:]
                freq_label = h5f['Freq_Label'][:]  # Assuming Freq_Label is also not concatenated

            # Concatenate other variables
            data["Epoch"].extend(h5f['Epoch'][:])
            data["Polarization"].extend(h5f['Polarization'][:])
            data["Theta"].extend(h5f['Theta'][:])

    # Convert lists to arrays
    data["Epoch"] = timestamp_to_datetime(np.array(data["Epoch"]))
    data["Polarization"] = np.array(data["Polarization"])
    data["Theta"] = np.array(data["Theta"])
    
    # Add Frequency and Freq_Label separately
    data["Frequency"] = freq_table
    data["Freq_Label"] = freq_label

    return data


def timestamp_to_datetime_2(epoch_timestamp):
    """Convert epoch timestamps to datetime objects."""

    return [datetime.datetime.utcfromtimestamp(ts) for ts in epoch_timestamp]

# Accessing the data
