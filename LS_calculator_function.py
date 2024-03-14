import multiprocessing
import os

def calculate_LS(args):
    lazy_loader, index_freq, time, data_final, normalize_LS= args
    pid = os.getpid()
    print(f"Worker PID: {pid}")


    f_LS_, power_LS_ = lazy_loader.LS_calculation(
        time, data_final[:, index_freq], normalize_LS
    )
    return f_LS_, power_LS_


