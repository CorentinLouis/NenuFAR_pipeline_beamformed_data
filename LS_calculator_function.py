import os
import dask.array as da
import numpy


def calculate_LS(args):
    lazy_loader, time, data, normalize_LS, lombscargle_function, log_infos = args    
    #lazy_loader, index_freq, time, data_final, normalize_LS= args
    pid = os.getpid()
    print(f"Worker PID: {pid}")
    f_LS, power_LS = lazy_loader.LS_calculation(
                            time, 20*numpy.log10(data), normalize_LS, log_infos, lombscargle_function
                            )
    
    return f_LS, power_LS #time, frequencies, data_final, f_LS, power_LS


