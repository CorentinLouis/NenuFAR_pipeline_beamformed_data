# Pipeline to process NenuFAR LT 02 beamformed data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Routines to process and analyse the NenuFAR LT 02 Stellar/Exoplanets radio beamformed data

The `pipeline_beamformed_LT02.ipynb` file gives a hint on how to use the differents routines.

`ReadFits_data.py` allows to read and open L1 or L1a data FITS files from ES/LT02 NenuFAR program;  
`ReadFits_rfimask.py` allows to read and open L1 or L1a mask FITS files from ES/LT02 NenuFAR program;  
More to come, stay tuned..

Requirements:  
- python 3.8.10
- `pip install -r requirements.txt`

Use:
How do I pre-process the data before applying Lomb-Scargle.

Use lomb_scargle_calculation.py function. 


Example for the data from NenuFAR KP 07 (Jupiter) dataset

```python3 lomb_scargle_calculation.py -key_project 07 -target Jupiter --main_directory_path './data/' --apply_rfi_mask --rfi_mask_level 0 --interpolation_in_time --interpolation_in_time_value 600 --interpolation_in_frequency --interpolation_in_frequency_value 1 --save_as_hdf5 --output_directory './outputs/' --frequency_interval 8 90 --plot_results --log_infos --rfi_mask_level0_percentage 50 --stokes VI --lombscargle_function 'astropy'```
