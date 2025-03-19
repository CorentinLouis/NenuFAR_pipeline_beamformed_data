# A Pipeline to process NenuFAR beamformed data and calculate Lomb-Scargle Periodograms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This pipeline process and analyse the NenuFAR radio beamformed data of a given possible radio emitter (such as a planet, exoplanet, or star). It allows various processing steps ( including RFI masking, interpolation in time and frequency), and analyzing methods (Lomb-Scargle Periodogram calculation, periodicity stacking)

## Usage
The script requires specifying key parameters such as the observation target and Key Project (KP) number. Various optional flags allow customization of data processing, masking, interpolation, and output formatting.

### Basic Command
```bash
python3 script.py -key_project <project_number> -target <target_name>
```

## Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `-key_project` | NenuFAR Key Project number (e.g., `02` or `07`). |
| `-target` | Observation Target Name (e.g., planet, exoplanet, or star). |

### Optional Arguments

#### Data Processing
| Argument | Default | Description |
|----------|---------|-------------|
| `--level_processing` | `L1` | Level of processing to be used (`L1` most of the time, sometime `L1a` e.g. for some specific target such as AD Leo). |
| `--main_directory_path` | `./data/` | Path where observations are stored. |
| `--stokes` | `V` | Stokes parameter to analyze (`I`, `V`, `V+`, `V-`, `VI`, `Q`, `U`, `L`). |
| `--threshold` | `None` | Threshold to be applied for Lomb-Scargle calculation. All data above this threshold won't be taken into account |
| `--frequency_interval` | `[10, 90]` | Half-open minimal and maximal frequency range for Lomb-Scargle analysis. |
| `--only_data_during_night` | `False` | Select only data recorded during night time (>4 hours, <22 hours UT). |
| `--off_beams` | `False` | Analyze off-beam observations. |


#### RFI Masking
| Argument | Default | Description |
|----------|---------|-------------|
| `--apply_rfi_mask` | `False` | Apply RFI mask. |
| `--rfi_mask_level` | `None` | RFI mask level (`0`, `1`, `2`, or `3`). Required if `--apply_rfi_mask` is `True`. |
| `--rfi_mask_level0_percentage` | `10` | Percentage threshold for RFI mask level `0`. Values: `0-100`. |

#### Interpolation
| Argument | Default | Description |
|----------|---------|-------------|
| `--interpolation_in_time` | `False` | Enable time interpolation. |
| `--interpolation_in_time_value` | `1` sec | Time interpolation interval (in seconds). |
| `--interpolation_in_frequency` | `False` | Enable frequency interpolation. |
| `--interpolation_in_frequency_value` | `1` MHz | Frequency interpolation interval (in MHz). |

#### Lomb-Scargle Calculation
| Argument | Default | Description |
|----------|---------|-------------|
| `--lombscargle_calculation` | `False` | Perform Lomb-Scargle periodogram calculation. |
| `--lombscargle_function` | `scipy` | Lomb-Scargle package to use (`scipy` or `astropy`). |
| `--normalize_LS` | `False` | Normalize Lomb-Scargle periodogram. |
| `--remove_background_to_LS` | `False` | Remove background from Lomb-Scargle plots. |
| `--periodicity_stacking_calculation` | `False` | Compute stacked periodicity for exoplanet revolution or star rotation. |

#### Logging & Verbosity
| Argument | Default | Description |
|----------|---------|-------------|
| `--verbose` | `False` | Print log information to the screen. |
| `--log_infos` | `False` | Print Dask computing info and control graphics after computation. |


#### Plotting & Output
| Argument | Default | Description |
|----------|---------|-------------|
| `--plot_results` | `False` | Generate and save plots. |
| `--figsize` | `None` | Set figure size. |
| `--plot_x_lim` | `None` | Set x-axis limits for Lomb-Scargle plots. |
| `--save_as_hdf5` | `False` | Save results in HDF5 format at different steps of the pipeline. |
| `--output_directory` | `./` | Output directory for results. |

#### Pre-Processed Data Handling
| Argument | Default | Description |
|----------|---------|-------------|
| `--plot_only` | `False` | Plot results from pre-computed HDF5 data. |
| `--reprocess_LS_periodogram` | `False` | Re-process Lomb-Scargle calculation from HDF5 data. |
| `--input_hdf5_file` | `None` | Path to pre-computed HDF5 file. Required if `--plot_only` or `--reprocess_LS_periodogram` is `True`. |
| `--beam_number` | `None` | Beam number to reprocess if `--reprocess_LS_periodogram` is `True`. |

## Example Commands

### Process data from KP 07 observations of Jupiter with Default Parameters and save results 
```bash
python script.py -key_project 07 -target Jupiter --save_as_hdf5
```

### Process data from KP O7 observations of Jupiter and apply RFI Mask level 2 and save results
```bash
python script.py -key_project 07 -target Jupiter --apply_rfi_mask --rfi_mask_level 2 --save_as_hdf5
```
### Process data from KP O7 observations of Jupiter and apply RFI Mask level 2 and calculate Lomb Scargle Periodogram
```bash
python script.py -key_project 07 -target Jupiter --apply_rfi_mask --rfi_mask_level 2 --lombscargle_calculation --save_as_hdf5
```

### Generate Only Plots from Pre-Processed Data
```bash
python script.py --plot_only --input_hdf5_file ./data/processed/processed_data_file.hdf5
```

## Example of Lomb Scargle Periodograms of Louis et al. (2025) "Detection method for periodic radio emissions from an exoplanet's magnetosphere or a star--planet interaction" publication
### Example with a simulated signal
The `normal_distribution_study.ipynb` Jupyter Notebook allows to understand how the Lomb–Scargle periodogram performs on a simulated signal. Initially, it use a sine wave with a known periodicity and random observation gaps. Next, it produce a similar simulated signal, but this time we control the observation windows and gaps between observations to study the impact of observaion regularity on the Lomb–Scargle periodogram, hence mimicking real observing conditions. Finally, it embed the signal in random noise with a normal distribution and study how varying the signal-to-noise ratio affects the detection of the underlying periodic signal.

This Jupyter notebook allows to reproduce the Figures of Section 2 the Louis et al. paper (2025) "Detection method for periodic radio emissions from an exoplanet's magnetosphere or a star--planet interaction" publication

### Example with real NenuFAR data from KP O7 observations of Jupiter

First, the NenuFAR KP07 data were processed using the following command
```bash
python3 lomb_scargle_calculation.py -key_project 07 -target Jupiter --stokes VI --main_directory_path './data/' --apply_rfi_mask --rfi_mask_level 0  --rfi_mask_level0_percentage 50 --interpolation_in_time --interpolation_in_time_value 600 --interpolation_in_frequency --interpolation_in_frequency_value 1 --save_as_hdf5 --output_directory './outputs/' --frequency_interval 8 90 --log_infos
```
Once the processed file has been created (accessible upon request to corentin.louis@obspm.fr), the specific lomb-scargle periodograms displayed in Louis et al. (2025) can be calculated using the `Louis_2025_LS_Jupiter.ipynb` Jupyter Notebook.
