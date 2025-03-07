import numpy
from astropy.timeseries import LombScargle as LombScargle_astropy
from lomb_scargle_calculation import read_hdf5_file, datetime_to_timestamp, timestamp_to_datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.colors as mcolors

from matplotlib import gridspec
import matplotlib.dates as mdates


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - leftMin) / (leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def calculate_LS_periodogram(time, signal, exoplanet_period_in_hours, ls_object = False):

    nout=100000
    T_exoplanet = numpy.mean(exoplanet_period_in_hours)*60*60 # T needs to be in seconds #numpy.mean() is taken, in case T_exoplanet is a list
    T1 = T_exoplanet/10     # Period min
    T2 = T_exoplanet*10     # Period max
    w1 = 2*numpy.pi/T1      # Pulsation max
    w2 = 2*numpy.pi/T2      # Pulsation min
    #f_LS = numpy.logspace(numpy.log10(w2), numpy.log10(w1), nout)  / (2 * numpy.pi) #Frequencies at which to search for periodicity with LombScargle
    f_LS = numpy.linspace(w2, w1, nout)  / (2 * numpy.pi) #Frequencies at which to search for periodicity with LombScargle


    normalization='standard'
    method = 'auto'
    fit_mean = False # improve the accuracy of results, especially in the case of incomplete phase coverage
    center_data = False 
    
    ls_tmp = LombScargle_astropy(time, signal, fit_mean = fit_mean)
    #frequency_LS, power_LS = ls_tmp.autopower(method=method, minimum_frequency = f_LS[0], maximum_frequency = f_LS[-1], samples_per_peak=100)
    frequency_LS = f_LS
    power_LS = ls_tmp.power(f_LS)
        
    #frequency_LS, power_LS = LombScargle_astropy(time, signal, fit_mean = fit_mean, normalization= normalization).autopower(method = method, minimum_frequency = f_LS[0], maximum_frequency = f_LS[-1], samples_per_peak=100)
    
    if ls_object:
        return ls_tmp, frequency_LS, power_LS
    else:
        return frequency_LS, power_LS


def randomization_test(t_observed, y_observed, Period_exoplanet, n_iterations = 100):
    max_powers_randomized = []

    frequency_LS, power_LS = calculate_LS_periodogram(t_observed, y_observed, Period_exoplanet)
    max_power_original = numpy.max(power_LS)
    for _ in range(n_iterations):
        # shuffle the values
        y_randomized = numpy.random.permutation(y_observed)

        # Compute the LS periodogram with shuffled data
        frequency_LS_randomized, power_LS_randomized = calculate_LS_periodogram(t_observed, y_randomized, Period_exoplanet)

        # Store the maximum power of the shuffled data
        max_powers_randomized.append(numpy.max(power_LS_randomized))

    # Calculate the confidence levels (percentiles)
    conf_levels = numpy.percentile(max_powers_randomized, [50, 90, 95, 99, 99.9])

    # Compute p-value or confidence level
    #p_value = numpy.sum(max_powers_randomized >= max_power_original) / n_iterations
    #confidence_level = 1 - p_value

    return(frequency_LS, power_LS, conf_levels)

def randomization_test_using_LS_package(t_observed, y_observed, Period_exoplanet, method = 'baluev'):
    (ls_object, frequency_LS, power_LS) = calculate_LS_periodogram(t_observed, y_observed, Period_exoplanet, ls_object = True)
        
    # method : 'baluev', 'davies', 'naive', 'bootstrap'
    
    # Find the period corresponding to the peak power
    best_frequency = frequency_LS[numpy.argmax(power_LS)]
    best_period = 1 / best_frequency / 3600
    print(f"Best period: {best_period:.4f}")

    # Compute the False Alarm Probability (FAP) for all powers
    fap = ls_object.false_alarm_probability(power_LS, method = method)  # Pass the entire power array

    # Find the FAP for the peak power
    peak_fap = fap[numpy.argmax(power_LS)]
    print(f"False Alarm Probability (FAP) for peak: {peak_fap:.4e}")

    # Find power levels corresponding to specific FAP thresholds
    fap_levels = [0.50, 0.90, 0.95, 0.99, 0.999]
    false_alarm_levels = ls_object.false_alarm_level(fap_levels, method = method)
    for level, fap_level in zip(fap_levels, false_alarm_levels):
        print(f"Power threshold for FAP {level*100:.2f}%: {fap_level:.4f}")
    
    return(frequency_LS, power_LS, fap_levels, false_alarm_levels)


def calculate_and_plot_LS_distrib(t_observed, y_observed, Period_exoplanet, main_title, real_data = False, x_zoomin = None, y_zoomin = None, add_p_values = None, add_extra_T = None, log_x_scale = False, log_y_scale = False, savefig = False, filename_savedfile = 'LS_plot.pdf', T_title = None, y_T_arrow = None):
    # x_zoomin and y_zoomin: either None, or [min, max]
    # add_extra_T needs to be a dictionnary like: {T_value: float, T_name: string}/
    #     e.g., add_extra_T = {
    #                           'Europa': {'T_value': 11.2321, 'T_name': 'Europa'},
    #                           'Ganymede': {'T_value': 10.5330, 'T_name': 'Ganymede'}
    #                          }

    if add_p_values != None:
        if add_p_values.lower() == 'p-test':
            (frequency_LS, power_LS, confidence_level) = randomization_test(t_observed, y_observed, Period_exoplanet)
            confidence_level_labels = numpy.array([50, 90, 95, 99, 99.9])
        if add_p_values.lower() == 'from_ls':
            (frequency_LS, power_LS, fap_levels, confidence_level) = randomization_test_using_LS_package(t_observed, y_observed, Period_exoplanet, method = 'baluev')
            confidence_level_labels = numpy.array([99.9, 99, 95, 90, 50])
    else:
        (frequency_LS, power_LS) = calculate_LS_periodogram(t_observed, y_observed, Period_exoplanet)

    
    T_Synodic_Io = 0.5394862621777665 * 24 * 3600
    T_Jupiter = 9.95*3600
    T_Moon = 1.035050109661509 * 24 *3600
    T_Day = 0.99726968 * 24 * 3600

    fig, ax = plt.subplots()
    ax.plot(1/frequency_LS/60/60, power_LS)
    
    if add_p_values:
        ax.plot

    if x_zoomin != None:
        x_min = x_zoomin[0]
        x_max = x_zoomin[1]
    else:
        x_min = (1/frequency_LS/60/60).min()
        x_max = (1/frequency_LS/60/60).max()
    ax.set_xlim(x_min,x_max)
    if y_zoomin != None:
        y_min = y_zoomin[0]
        y_max = y_zoomin[1]
    else:
        if numpy.isfinite(numpy.max(power_LS)):
            y_min = -0.1*numpy.max(power_LS)
            y_max = 1.7*numpy.max(power_LS)
        else:
            y_min = 0
            y_max = 1
    ax.set_ylim(y_min,y_max)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("LS Power")
    ax.set_title(main_title)

    #ax.annotate("", xy=(T_Jupiter/60/60, 1.1*numpy.max(power_LS)), xytext=(T_Jupiter/60/60, 1.25*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="darkgreen"))

    #ax.annotate(r"T$_{Jupiter}$", xy=(T_Jupiter/60/60,  1.3*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="darkgreen")
    if T_title == None:
        if real_data != True :
            T_title = r"T$_{input}$"
        else:
            T_title = r"T$_{Io}$"
    if y_T_arrow == None:
        y_T_arrow = numpy.max(power_LS)

    ax.annotate("", xy=(Period_exoplanet, 1.1*y_T_arrow), xytext=(Period_exoplanet, 1.4*y_T_arrow),
                arrowprops=dict(arrowstyle="->", color="dodgerblue"))
    ax.annotate(T_title, xy=(Period_exoplanet, 1.4*y_T_arrow),
                ha='left', va='bottom',
                color="dodgerblue")

    if add_extra_T != None:
        ind_extra_T = 0
        for ind, value in add_extra_T.items():
            ind_extra_T=ind_extra_T+1
            ax.annotate("",
                 xy=(value['T_value'], 1.1*y_T_arrow),
                 xytext=(value['T_value'], (1.4+ind_extra_T*0.1)*y_T_arrow),
                 arrowprops=dict(arrowstyle="->", color="dodgerblue"))
            T_title = f"{value['T_name']}"
            ax.annotate(T_title,
                 xy=(value['T_value'], (1.4+ind_extra_T*0.1)*y_T_arrow),
                 ha='left', va='bottom',
                 color="dodgerblue")

                 
    
    if log_x_scale:
        ax.set_xscale("log", base=10)
    if log_y_scale:
        ax.set_yscale("log", base=10)
    #plt.tight_layout()
    if add_p_values:
        # Plot the confidence levels
        for i, conf_level in enumerate(confidence_level):
            ax.axhline(y=conf_level, color='C{}'.format(i+1), linestyle='--', label=f'{confidence_level_labels[i]}%')
        ax.legend(title = "Confidence Level")
        if savefig:
                plt.savefig(filename_savedfile, format='pdf', transparent = True, dpi='figure')
        else:
            plt.show()
        plt.close()
        return(frequency_LS, power_LS, confidence_level)
    else:
        if savefig:
            plt.savefig(filename_savedfile, format='pdf', transparent = True, dpi='figure')
        else:
            plt.show()
        plt.close()
        return(frequency_LS, power_LS)
    
    #ax.annotate("", xy=(T_Moon/60/60, 1.35*numpy.max(power_LS)), xytext=(T_Moon/60/60, 1.5*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="slategray"))

    #ax.annotate(r"T$_{Moon}$", xy=(T_Moon/60/60, 1.5*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="slategray")
                
    #ax.annotate("", xy=(T_Day/60/60, 1.1*numpy.max(power_LS)), xytext=(T_Day/60/60, 1.25*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="orange"))

    #ax.annotate(r"T$_{Sun}$", xy=(T_Day/60/60, 1.3*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="orange")


def calculate_and_plot_LS_distrib_ON_minus_OFF(t_observed_ON, y_observed_ON, t_observed_OFF, y_observed_OFF, Period_exoplanet, main_title, real_data = False, x_zoomin = None, y_zoomin = None, add_p_values = None, add_extra_T = None, log_x_scale = False, savefig = False, filename_savedfile = 'LS_plot.pdf', T_title = None, y_T_arrow = None):
    # x_zoomin and y_zoomin: either None, or [min, max]
    # add_extra_T needs to be a dictionnary like: {T_value: float, T_name: string}/
    #     e.g., add_extra_T = {
    #                           'Europa': {'T_value': 11.2321, 'T_name': 'Europa'},
    #                           'Ganymede': {'T_value': 10.5330, 'T_name': 'Ganymede'}
    #                          } in hours

    
    (frequency_LS_ON, power_LS_ON) = calculate_LS_periodogram(t_observed_ON, y_observed_ON, Period_exoplanet)
    (frequency_LS_OFF, power_LS_OFF) = calculate_LS_periodogram(t_observed_OFF, y_observed_OFF, Period_exoplanet)

    power_LS = power_LS_ON-power_LS_OFF
    frequency_LS = frequency_LS_ON
    
    T_Synodic_Io = 0.5394862621777665 * 24 * 3600
    T_Jupiter = 9.95*3600
    T_Moon = 1.035050109661509 * 24 *3600
    T_Day = 0.99726968 * 24 * 3600

    fig, ax = plt.subplots()
    ax.plot(1/frequency_LS/60/60, power_LS)
    
    if add_p_values:
        ax.plot

    if x_zoomin != None:
        x_min = x_zoomin[0]
        x_max = x_zoomin[1]
    else:
        x_min = (1/frequency_LS/60/60).min()
        x_max = (1/frequency_LS/60/60).max()
    ax.set_xlim(x_min,x_max)
    if y_zoomin != None:
        y_min = y_zoomin[0]
        y_max = y_zoomin[1]
    else:
        if numpy.isfinite(numpy.max(power_LS)):
            y_min = -0.1*numpy.max(power_LS)
            y_max = 1.7*numpy.max(power_LS)
        else:
            y_min = 0
            y_max = 1
    ax.set_ylim(y_min,y_max)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("LS Power")
    ax.set_title(main_title)

    #ax.annotate("", xy=(T_Jupiter/60/60, 1.1*numpy.max(power_LS)), xytext=(T_Jupiter/60/60, 1.25*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="darkgreen"))

    #ax.annotate(r"T$_{Jupiter}$", xy=(T_Jupiter/60/60,  1.3*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="darkgreen")
    if T_title == None:
        if real_data != True :
            T_title = r"T$_{input}$"
        else:
            T_title = r"T$_{Io}$"
    if y_T_arrow == None:
        y_T_arrow = numpy.max(power_LS)

    ax.annotate("", xy=(Period_exoplanet, 1.1*y_T_arrow), xytext=(Period_exoplanet, 1.4*y_T_arrow),
                arrowprops=dict(arrowstyle="->", color="dodgerblue"))
    ax.annotate(T_title, xy=(Period_exoplanet, 1.4*y_T_arrow),
                ha='left', va='bottom',
                color="dodgerblue")

    if add_extra_T != None:
        ind_extra_T = 0
        for ind, value in add_extra_T.items():
            ind_extra_T=ind_extra_T+1
            ax.annotate("",
                 xy=(value['T_value'], 1.1*y_T_arrow),
                 xytext=(value['T_value'], (1.4+ind_extra_T*0.1)*y_T_arrow),
                 arrowprops=dict(arrowstyle="->", color="dodgerblue"))
            T_title = f"{value['T_name']}"
            ax.annotate(T_title,
                 xy=(value['T_value'], (1.4+ind_extra_T*0.1)*y_T_arrow),
                 ha='left', va='bottom',
                 color="dodgerblue")
    
    if log_x_scale:
        ax.set_xscale("log", base=10)

    #plt.tight_layout()
    if add_p_values:
        # Plot the confidence levels
        for i, conf_level in enumerate(confidence_level):
            ax.axhline(y=conf_level, color='C{}'.format(i+1), linestyle='--', label=f'{confidence_level_labels[i]}%')
        ax.legend(title = "Confidence Level")
        if savefig:
                plt.savefig(filename_savedfile, format='pdf', transparent = True, dpi='figure')
        return(frequency_LS, power_LS, confidence_level)
    else:
        if savefig:
            plt.savefig(filename_savedfile, format='pdf', transparent = True, dpi='figure')
        return(frequency_LS, power_LS)
    
    #ax.annotate("", xy=(T_Moon/60/60, 1.35*numpy.max(power_LS)), xytext=(T_Moon/60/60, 1.5*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="slategray"))

    #ax.annotate(r"T$_{Moon}$", xy=(T_Moon/60/60, 1.5*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="slategray")
                
    #ax.annotate("", xy=(T_Day/60/60, 1.1*numpy.max(power_LS)), xytext=(T_Day/60/60, 1.25*numpy.max(power_LS)),
    #            arrowprops=dict(arrowstyle="->", color="orange"))

    #ax.annotate(r"T$_{Sun}$", xy=(T_Day/60/60, 1.3*numpy.max(power_LS)),
    #            ha='center', va='bottom',
    #            color="orange")

def plot_LS_1D_and_2D(time_datetime, frequencies, data, 
                        target = '', # target name
                        target_type = 'exoplanet', # exoplanet or star
                        T_search = 1.22, # Expected periodicity in days (LS will look for periodicity x/ 10 this value)
                        extra_title='',
                        x_zoomin = None, y_zoomin = None,
                        log_x_scale = False,
                        y_zoomin_2D_periodogram = None,
                        vmin = None, vmax = None,
                        cmap = 'inferno',
                        y_T_arrow = None,
                        T_title = None,
                        add_extra_T = None,
                        savefig = False,
                        filename = 'LS_periodograms.pdf',
                        add_p_values = None):
 
    for i_freq, freq in enumerate(frequencies):
        time = datetime_to_timestamp(time_datetime)
        data_ = data[:, frequencies==freq][:,0]
        data_[numpy.abs(data_) > 1] = numpy.nan # unphysical points
        data_[numpy.isnan(data_)] = 0
        data_[numpy.isinf(data_)] = 0

        print(f'data: {data_.min()}, {data_.max()}')
        title = f"Lomb Scargle (LS) Periodogram, f = {freq}"+'\n'+f'{extra_title}'
        if T_title == None:
            T_title = f'{target}'
    
        if add_p_values:
            (frequency_LS, power_LS, confidence_level) = calculate_and_plot_LS_distrib(time, data_, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T=add_extra_T, add_p_values=add_p_values)
        else:
            (frequency_LS, power_LS) = calculate_and_plot_LS_distrib(time, data_, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T=add_extra_T)
        if i_freq == 0:
            power_LS_full = numpy.zeros((frequencies.shape[0], frequency_LS.shape[0]))
        power_LS_full[i_freq, : ] = power_LS
    
    power_LS_full[numpy.isnan(power_LS_full)] = 0
    power_LS_full[numpy.isinf(power_LS_full)] = 0
    print(f'power_LS: {power_LS_full.min()}, {power_LS_full.max()}')
    plot_LS_2D_periodogram(frequency_LS, frequencies, power_LS_full,
                        x_zoomin = x_zoomin,
                        y_zoomin = y_zoomin_2D_periodogram,
                        log_x_scale = log_x_scale,
                        T_search = T_search, T_name = T_title, add_extra_T = add_extra_T,
                        vmin = vmin, vmax = vmax, cmap = cmap, savefig = savefig, filename = filename)

    return(frequency_LS, frequencies, power_LS_full)

def read_data_and_plot_LS(path_to_data = './',
                          file_name = '',
                          target = '', # target name
                          target_type = 'exoplanet', # exoplanet or star
                          T_search = 1.22, # Expected periodicity in days (LS will look for periodicity x/ 10 this value)
                          extra_title='',
                          beam_on = True, beam_off = False,
                          beam_off_number = '', # empty string or number (int or string format)
                          x_zoomin = None, y_zoomin = None,
                          log_x_scale = False,
                          y_zoomin_2D_periodogram = None,
                          vmin = None, vmax = None,
                          cmap = 'inferno',
                          y_T_arrow = None,
                          T_title = None,
                          add_extra_T = None,
                          savefig = False,
                          filename = 'LS_periodograms.pdf',
                          add_p_values = None):
    
    ext = file_name.split('.')[-1]
    main_filename = file_name.split('.')[0]
    main_filename_no_beam_name = main_filename.split('_O')[0]

    if beam_on:
        file_NenuFAR_observations_beam_ON = f'{main_filename_no_beam_name}_ON.{ext}'
        (time_datetime_beam_ON,
            frequencies_beam_ON,
            data_final_beam_ON,
            stokes_beam_ON,
            key_project_beam_ON,
            target_beam_ON,
            T_exoplanet_beam_ON,
            T_star_beam_ON
            ) = read_hdf5_file(path_to_data+file_NenuFAR_observations_beam_ON, dataset=True, LS_dataset = False)
        frequencies = frequencies_beam_ON

        sorted_indices = numpy.argsort(time_datetime_beam_ON)

        data_final_beam_ON = data_final_beam_ON[sorted_indices, :]
        time_datetime_beam_ON = time_datetime_beam_ON[sorted_indices]
        frequencies = frequencies_beam_ON

            

    if beam_off:
        file_NenuFAR_observations_beam_OFF = f'{main_filename_no_beam_name}_OFF{beam_off_number}.{ext}'
        (time_datetime_beam_OFF,
            frequencies_beam_OFF,
            data_final_beam_OFF,
            stokes_beam_OFF,
            key_project_beam_OFF,
            target_beam_OFF,
            T_exoplanet_beam_OFF,
            T_star_beam_OFF
            ) = read_hdf5_file(path_to_data+file_NenuFAR_observations_beam_OFF, dataset=True, LS_dataset = False)
        frequencies = frequencies_beam_OFF

        sorted_indices = numpy.argsort(time_datetime_beam_OFF)

        data_final_beam_OFF = data_final_beam_OFF[sorted_indices,:]
        time_datetime_beam_OFF = time_datetime_beam_OFF[sorted_indices]

     
    for i_freq, freq in enumerate(frequencies):
        if beam_on:
            time_real_beam_ON = datetime_to_timestamp(time_datetime_beam_ON)
            data_for_LS_beam_ON = data_final_beam_ON[:, frequencies_beam_ON==freq][:,0]
            data_for_LS_beam_ON[numpy.abs(data_for_LS_beam_ON) > 1] = numpy.nan # unphysical points
            data_for_LS_beam_ON[numpy.isnan(data_for_LS_beam_ON)] = 0
            data_for_LS_beam_ON[numpy.isinf(data_for_LS_beam_ON)] = 0

        if beam_off:
            time_real_beam_OFF = datetime_to_timestamp(time_datetime_beam_OFF)
            data_for_LS_beam_OFF = data_final_beam_OFF[:, frequencies_beam_OFF==freq][:,0]
            data_for_LS_beam_OFF[numpy.abs(data_for_LS_beam_OFF) > 1] = numpy.nan # unphysical points
            data_for_LS_beam_OFF[numpy.isnan(data_for_LS_beam_OFF)] = 0
            data_for_LS_beam_OFF[numpy.isinf(data_for_LS_beam_OFF)] = 0

        title = f"Lomb Scargle (LS) Periodogram, f = {freq}"+'\n'+f'{extra_title}'
        if T_title == None:
            T_title = f'{target}'
        if beam_on and beam_off:
            if add_p_values:
                (frequency_LS, power_LS, confidence_level) = calculate_and_plot_LS_distrib_ON_minus_OFF(time_real_beam_ON, data_for_LS_beam_ON, time_real_beam_OFF, data_for_LS_beam_OFF, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin =y_zoomin, y_T_arrow = 0.010, add_extra_T = add_extra_T, add_p_values= add_p_values, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')
            else:
                (frequency_LS, power_LS) = calculate_and_plot_LS_distrib_ON_minus_OFF(time_real_beam_ON, data_for_LS_beam_ON, time_real_beam_OFF, data_for_LS_beam_OFF, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin =y_zoomin, y_T_arrow = 0.010, add_extra_T = add_extra_T, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')
        elif beam_on ==True and beam_off == False:
            if add_p_values:
                (frequency_LS, power_LS, confidence_level) = calculate_and_plot_LS_distrib(time_real_beam_ON, data_for_LS_beam_ON, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T=add_extra_T, add_p_values=add_p_values, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')
            else:
                (frequency_LS, power_LS) = calculate_and_plot_LS_distrib(time_real_beam_ON, data_for_LS_beam_ON, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T=add_extra_T, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')
        elif beam_on ==False and beam_off == True:
            if add_p_values:
                (frequency_LS, power_LS, confidence_level) = calculate_and_plot_LS_distrib(time_real_beam_OFF, data_for_LS_beam_OFF, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T = add_extra_T, add_p_values=add_p_values, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')
            else:
                (frequency_LS, power_LS) = calculate_and_plot_LS_distrib(time_real_beam_OFF, data_for_LS_beam_OFF, T_search*24, title, real_data = True, log_x_scale=log_x_scale, T_title = T_title, x_zoomin=x_zoomin, y_zoomin = y_zoomin, y_T_arrow = y_T_arrow, add_extra_T = add_extra_T, savefig = savefig, filename_savedfile = f'LS_freq_{freq}.pdf')

        if i_freq == 0:
            power_LS_full = numpy.zeros((frequencies.shape[0], frequency_LS.shape[0]))
        power_LS_full[i_freq, : ] = power_LS
        
    
    #plot_LS_2D_periodogram(frequency_LS, frequencies, power_LS_full,
    plot_SNR_LS_periodogram_2D(frequency_LS, frequencies, power_LS_full,
                        x_zoomin = x_zoomin,
                        y_zoomin = y_zoomin_2D_periodogram,
                        log_x_scale = log_x_scale,
                        T_search = T_search, T_name = T_title, add_extra_T = add_extra_T,
                        vmin = vmin, vmax = vmax, cmap = cmap, savefig = savefig, filename = filename)

    return(frequency_LS, frequencies, power_LS_full)
    


def calculate_and_plot_LS_periodogram_over_samples_in_time(time_datetime_beam_ON, frequencies_beam_ON, data_final_beam_ON,
                                                exoplanet_period_in_hours = 12.993,
                                                hline_label = r'T$_{Input}$',
                                                LS_window_size = 1000,  # Number of points in each window
                                                LS_step_size = 50,
                                                cmap = 'bwr',
                                                color_hline = None,
                                                x_zoomin = None, # datetime object
                                                LS_y_zoomin = None,
                                                log_y_scale = False, 
                                                add_extra_T = None,
                                                vmin = None, vmax = None,
                                                savefig = False, filename = 'LS_periodogram_over_time.png'):
    
                 # add_extra_T needs to be a dictionnary like: {T_value: float, T_name: string}/
    #     e.g., add_extra_T = {
    #                           'Europa': {'T_value': 11.2321, 'T_name': 'Europa'},
    #                           'Ganymede': {'T_value': 10.5330, 'T_name': 'Ganymede'}
    #                          }

    n_panels = len(frequencies_beam_ON) + 1
    

    

    fig = plt.figure(figsize=(10,5*n_panels))
    gs = gridspec.GridSpec(n_panels, 2, width_ratios=[4, 1], height_ratios=[1]*n_panels, wspace=0.3)
    axes = []
    ax = plt.subplot(gs[0,0])
    axes.append(ax)
        

    i_ax = 1
    for i_freq in frequencies_beam_ON:    
        

        time_real_beam_ON = datetime_to_timestamp(time_datetime_beam_ON)
        #time_numeric = (time_real_beam_ON - time_real_beam_ON[0])  # Convert to seconds since start

        data_for_LS_beam_ON = data_final_beam_ON[:, frequencies_beam_ON==i_freq][:,0]
        data_for_LS_beam_ON[numpy.abs(data_for_LS_beam_ON) > 1] = numpy.nan # unphysical points
        data_for_LS_beam_ON[numpy.isnan(data_for_LS_beam_ON)] = 0
        data_for_LS_beam_ON[numpy.isinf(data_for_LS_beam_ON)] = 0
        
        axes[0].scatter(time_datetime_beam_ON, data_for_LS_beam_ON, s=10, label=f'{i_freq:.4f} MHz', alpha=0.5)
        
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0],sharey=axes[1] if i_ax > 1 else None)
        

        # Prepare storage for results
        power_results = []
        times_results = []

        # Sliding window approach

        # Method 1 is step of observed point
        for start in range(0, len(time_real_beam_ON) - LS_window_size, LS_step_size): 
            end = start + LS_window_size
            # Select the data for the current window
            print(timestamp_to_datetime(time_real_beam_ON[start]), timestamp_to_datetime(time_real_beam_ON[end]))
            time_window = time_real_beam_ON[start:end]
            data_window = data_for_LS_beam_ON[start:end]

            # Calculate Lomb-Scargle power
            frequency_LS, power_LS = calculate_LS_periodogram(time_window, data_window, exoplanet_period_in_hours)
            
            # Store results
            power_results.append(power_LS)
            # Store the center time of the current window
            times_results.append(numpy.mean(time_window))

        
        # Convert results to arrays for plotting
        power_results = numpy.array(power_results)
        times_results = numpy.array(times_results)
        times_plot = timestamp_to_datetime(times_results)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm)

        
        ax.set_ylabel(f'Periodicities (hours)'+'\n'+f'(for freq. {i_freq:.2f} MHz)')
        #ax.grid(True)
        #ax.set_yscale('log')
        #ax.set_title('Wavelet Power Spectrum')
        
        ax_cb = plt.subplot(gs[i_ax, 1])
        pos_cb = ax_cb.get_position()  # Get current position of the colorbar axis
        ax_cb.set_position([pos_cb.x0-0.05, pos_cb.y0, 0.02, pos_cb.height])  # Adjust width (0.02) and keep the same height

        cbar = plt.colorbar(img, cax=ax_cb, label='Lomb-Scargle Power')

        i_ax = i_ax+1
        axes.append(ax)

    axes[0].set_ylabel('Stokes V/I')
    axes[0].set_title(f'Beam On')
    axes[0].set_ylim(-1,1)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))#, title="Legend")
    axes[0].grid(True)
    # Remove x-axis labels and ticks for all but the last subplot
    for i_freq, ax in enumerate(axes[1:]):
        #ax.tick_params(axis='x', which='both', labelbottom=False)  # Remove ticks and labels
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel('LS Periodicities (hours)'+'\n'+f'(freq. {frequencies_beam_ON[i_freq]:.02f} MHz)')

        if log_y_scale:
            ax.set_yscale("log", base=10)

        if color_hline == None:
            if cmap == 'inferno':
                color_hline = 'red'
            else:
                color_hline = 'k'
        ax.axhline(y=exoplanet_period_in_hours, color=color_hline, linestyle=':', label=hline_label)

        
        if add_extra_T != None:
            ind_extra_T = 0
            for ind, value in add_extra_T.items():
                ind_extra_T=ind_extra_T+1
                ax.axhline(y=value['T_value'], color=color_hline, linestyle=':', label=f"{value['T_name']}")

        ax.legend()


    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month



    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month


    if x_zoomin != None:
        axes[-1].set_xlim(x_zoomin)    
    if LS_y_zoomin != None:
        axes[-1].set_ylim(LS_y_zoomin)    
    axes[-1].set_xlabel('Time (years)')
    
    plt.tight_layout()
    if savefig:
        plt.savefig(filename, transparent=True)
    else:
        plt.show()

    plt.close()



def read_and_plot_timeseries(path_to_data = './',
                          file_name = '',
                          target = '', # target name
                          target_type = 'exoplanet', # exoplanet or star
                          extra_title='',
                          beam_on = True, beam_off = False,
                          beam_off_number = '', # empty string or number (int or string format)
                          x_zoomin = None, y_zoomin = None,
                          fontsize = 20,
                          savefig = False,
                          filename = 'timeseries.pdf'):

    ext = file_name.split('.')[-1]
    main_filename = file_name.split('.')[0]
    main_filename_no_beam_name = main_filename.split('_O')[0]

    if beam_on:
        file_NenuFAR_observations_beam_ON = f'{main_filename_no_beam_name}_ON.{ext}'
        (time_datetime_beam_ON,
            frequencies_beam_ON,
            data_final_beam_ON,
            stokes_beam_ON,
            key_project_beam_ON,
            target_beam_ON,
            T_exoplanet_beam_ON,
            T_star_beam_ON
            ) = read_hdf5_file(path_to_data+file_NenuFAR_observations_beam_ON, dataset=True, LS_dataset = False)

        sorted_indices = numpy.argsort(time_datetime_beam_ON)

        data_final_beam_ON = data_final_beam_ON[sorted_indices,:]
        time_datetime_beam_ON = time_datetime_beam_ON[sorted_indices]
        frequencies = frequencies_beam_ON

    

    if beam_off:
        file_NenuFAR_observations_beam_OFF = f'{main_filename_no_beam_name}_OFF{beam_off_number}.{ext}'
        (time_datetime_beam_OFF,
            frequencies_beam_OFF,
            data_final_beam_OFF,
            stokes_beam_OFF,
            key_project_beam_OFF,
            target_beam_OFF,
            T_exoplanet_beam_OFF,
            T_star_beam_OFF
            ) = read_hdf5_file(path_to_data+file_NenuFAR_observations_beam_OFF, dataset=True, LS_dataset = False)
        frequencies = frequencies_beam_OFF

        sorted_indices = numpy.argsort(time_datetime_beam_OFF)

        data_final_beam_ON = data_final_beam_OFF[sorted_indices,:]
        time_datetime_beam_OFF = time_datetime_beam_OFF[sorted_indices]


    if (beam_on == True) and (beam_off==True):
        figsize = (20, 20)
        nrows = 2
    else:
        nrows = 1
        figsize = (20, 10)
    fig, ax_fig = plt.subplots(nrows,1,figsize=figsize)
    fig.suptitle('Observed Signal', fontsize = fontsize+4)

    for i_freq in frequencies:
        if beam_on:
            if beam_off == True:
                ax = ax_fig[0]
            else:
                ax = ax_fig
            time_real_beam_ON = datetime_to_timestamp(time_datetime_beam_ON)
            data_for_LS_beam_ON = data_final_beam_ON[:, frequencies_beam_ON==i_freq][:,0]
            data_for_LS_beam_ON[numpy.abs(data_for_LS_beam_ON) > 1] = numpy.nan # unphysical points
            data_for_LS_beam_ON[numpy.isnan(data_for_LS_beam_ON)] = 0
            data_for_LS_beam_ON[numpy.isinf(data_for_LS_beam_ON)] = 0

            ax.scatter(time_datetime_beam_ON, data_for_LS_beam_ON, s=10, label=f'{int(i_freq)} MHz', alpha=0.5)
        
            
            ax.legend(
                loc='upper left',  
                bbox_to_anchor=(1.05, 1),  
                borderaxespad=0,  
                fontsize=fontsize-2,
                ) 
            ax.set_title(f'Beam On', fontsize = fontsize+2)
            

        if beam_off:
            if beam_on == True:
                ax = ax_fig[1]
            else:
                ax = ax_fig
            
            time_real_beam_OFF = datetime_to_timestamp(time_datetime_beam_OFF)
            data_for_LS_beam_OFF = data_final_beam_OFF[:, frequencies_beam_OFF==i_freq][:,0]
            data_for_LS_beam_OFF[numpy.abs(data_for_LS_beam_OFF) > 1] = numpy.nan # unphysical points
            data_for_LS_beam_OFF[numpy.isnan(data_for_LS_beam_OFF)] = 0
            data_for_LS_beam_OFF[numpy.isinf(data_for_LS_beam_OFF)] = 0

            ax.scatter(time_datetime_beam_OFF, data_for_LS_beam_OFF, s=10, label=f'{i_freq:.4f} kHz', alpha=0.5)
            
            if beam_on == False:    
                ax.legend(
                loc='upper left',  
                bbox_to_anchor=(1.05, 1),  
                borderaxespad=0,  
                fontsize=fontsize-2,
                ) 
            ax.set_title(f'Beam Off', fontsize = fontsize+2)

        ax.grid(True)        
        ax.set_xlabel('Time (years)', fontsize = fontsize)
        ax.set_ylabel('Stokes V/I', fontsize = fontsize)
        ax.set_ylim(-1,1)
        ax.tick_params(axis='x', labelsize=fontsize)  # X-axis tick labels
        ax.tick_params(axis='y', labelsize=fontsize)
        plt.tight_layout()
        plt.savefig(filename, format='pdf', transparent = True, dpi='figure')
    


def calculate_and_plot_LS_periodogram_over_time(time_datetime_beam_ON, frequencies_beam_ON, data_final_beam_ON,
                                                exoplanet_period_in_hours = 12.993,
                                                hline_label = r'T$_{Input}$',
                                                LS_window_size = 10000, # Number in minutes
                                                LS_step_size = 500, # Number in minutes
                                                observed_frequency_limits = [8,80],
                                                title_main = f'Beam On',
                                                cmap = 'bwr',
                                                color_hline = None,
                                                x_zoomin = None, # datetime object
                                                LS_y_zoomin = None,
                                                log_y_scale = False, 
                                                add_extra_T = None,
                                                vmin = None, vmax = None,
                                                savefig = False, filename = 'LS_periodogram_over_time.png',
                                                stokes = 'V/I',
                                                extra_panel_data = None,
                                                extra_panel_ytitle = None,
                                                figsize = None):
    
                 # add_extra_T needs to be a dictionnary like: {T_value: float, T_name: string}/
    #     e.g., add_extra_T = {
    #                           'Europa': {'T_value': 11.2321, 'T_name': 'Europa'},
    #                           'Ganymede': {'T_value': 10.5330, 'T_name': 'Ganymede'}
    #                          }


    if len(observed_frequency_limits) == 1:
        frequencies_to_be_used = frequencies_beam_ON[frequencies_beam_ON == observed_frequency_limits[0]]
    
    else:
        # Find the index of the first value less than or equal to the lower limit
        lower_index = numpy.max(numpy.where(frequencies_beam_ON <= observed_frequency_limits[0]))

        # Find the index of the first value greater than or equal to the upper limit
        upper_index = numpy.min(numpy.where(frequencies_beam_ON >= observed_frequency_limits[1]))

        # Select the values that bound the range
        frequencies_to_be_used = frequencies_beam_ON[lower_index:upper_index + 1]


    n_panels = len(frequencies_to_be_used) + 1
    
    if extra_panel_data != None:
        n_panels = n_panels+1
        height_ratios = [1] * (n_panels - 1) + [0.5]
    else:
        height_ratios = [1]*n_panels
    
    if figsize == None:
        figsize=(10,5*n_panels)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_panels, 2, width_ratios=[4, 1], height_ratios=height_ratios, wspace=0.3)
    axes = []
    ax = plt.subplot(gs[0,0])
    axes.append(ax)
    

    i_ax = 1
    
    for i_freq in frequencies_to_be_used:  
        print(f'Frequency: {i_freq} MHZ')
        time_real_beam_ON = datetime_to_timestamp(time_datetime_beam_ON)
        #time_numeric = (time_real_beam_ON - time_real_beam_ON[0])  # Convert to seconds since start

        data_for_LS_beam_ON = data_final_beam_ON[:, frequencies_beam_ON==i_freq][:,0]
        if stokes == 'V/I':
            mask_nonphysical = (numpy.abs(data_for_LS_beam_ON) > 1) + (numpy.isinf(data_for_LS_beam_ON)) + (numpy.isnan(data_for_LS_beam_ON))
            #data_for_LS_beam_ON[numpy.abs(data_for_LS_beam_ON) > 1] = numpy.nan # unphysical points
            #data_for_LS_beam_ON[numpy.isnan(data_for_LS_beam_ON)] = 0
            #data_for_LS_beam_ON[numpy.isinf(data_for_LS_beam_ON)] = 0
            
            time_real_for_LS_beam_ON = time_real_beam_ON[~mask_nonphysical]
            time_datetime_for_LS_beam_ON = time_datetime_beam_ON[~mask_nonphysical]
            data_for_LS_beam_ON = data_for_LS_beam_ON[~mask_nonphysical]

        elif stokes == 'V':
            mask_nonphysical = (numpy.abs(data_for_LS_beam_ON) > 1e9) + (numpy.isinf(data_for_LS_beam_ON)) + (numpy.isnan(data_for_LS_beam_ON))
            time_real_for_LS_beam_ON = time_real_beam_ON[~mask_nonphysical]
            time_datetime_for_LS_beam_ON = time_datetime_beam_ON[~mask_nonphysical]
            data_for_LS_beam_ON = data_for_LS_beam_ON[~mask_nonphysical]
            print(data_for_LS_beam_ON.shape, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), data_for_LS_beam_ON.mean())
            #data_for_LS_beam_ON = translate(data_for_LS_beam_ON, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), -1, 1)
            #print(data_for_LS_beam_ON.shape, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), data_for_LS_beam_ON.mean())
            

        axes[0].scatter(time_datetime_for_LS_beam_ON, data_for_LS_beam_ON, s=10, label=f'{i_freq:.4f} MHz', alpha=0.5)
        
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0],sharey=axes[1] if i_ax > 1 else None)
        

        # Prepare storage for results
        power_results = []
        times_results = []

        # Sliding window approach
        time_delta_minutes_total = int((time_real_for_LS_beam_ON.max()-time_real_for_LS_beam_ON.min())/60)
        for start in range(0, time_delta_minutes_total - LS_window_size, LS_step_size):
            
            time_window_start = time_real_for_LS_beam_ON.min()+start*60
            time_window_end = time_window_start+LS_window_size*60
            print(f'{timestamp_to_datetime(time_window_start)}, {timestamp_to_datetime(time_window_end)}')
            # Select the data for the current window
            t_mask = (time_real_for_LS_beam_ON >= time_window_start) * (time_real_for_LS_beam_ON < time_window_end)
            time_window = time_real_for_LS_beam_ON[t_mask]
            data_window = data_for_LS_beam_ON[t_mask]

            if time_window.size != 0:
                # Calculate Lomb-Scargle power
                frequency_LS, power_LS = calculate_LS_periodogram(time_window, data_window, exoplanet_period_in_hours)
            
                # Store results
                power_results.append(power_LS)
                # Store the center time of the current window
                times_results.append(numpy.mean(time_window))


        # Convert results to arrays for plotting
        power_results = numpy.array(power_results)
        times_results = numpy.array(times_results)
        times_plot = timestamp_to_datetime(times_results)

        if vmin==None:
            vmin = numpy.min(power_results)
        if vmax==None:
            vmax = numpy.max(power_results)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        #img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm, levels=20)
        img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, cmap=cmap, levels=numpy.linspace(vmin,vmax,20), extend='min', vmin=vmin, vmax=vmax)
        
        ax.set_ylabel(f'Periodicities (hours)'+'\n'+f'(for freq. {i_freq:.2f} MHz)')
        #ax.grid(True)
        #ax.set_yscale('log')
        #ax.set_title('Wavelet Power Spectrum')
        
        ax_cb = plt.subplot(gs[i_ax, 1])
        pos_cb = ax_cb.get_position()  # Get current position of the colorbar axis
        ax_cb.set_position([pos_cb.x0-0.05, pos_cb.y0, 0.02, pos_cb.height])  # Adjust width (0.02) and keep the same height

        cbar = plt.colorbar(img, cax=ax_cb, label='Lomb-Scargle Power')

        i_ax = i_ax+1
        axes.append(ax)

    axes[0].set_ylabel('Stokes V/I')
    axes[0].set_title(title_main)
    axes[0].set_ylim(-1,1)
    axes[0].legend(
                loc='upper left',  
                bbox_to_anchor=(1.05, 1),  
                borderaxespad=0,  
                #fontsize=fontsize-2,
                ) 
    axes[0].grid(True)

    for i_freq, ax in enumerate(axes[1:]):
        #ax.tick_params(axis='x', which='both', labelbottom=False)  # Remove ticks and labels
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel('LS Periodicities (hours)'+'\n'+f'(freq. {frequencies_to_be_used[i_freq]:.02f} MHz)')

        if log_y_scale:
            ax.set_yscale("log", base=10)

        if color_hline == None:
            if cmap == 'inferno':
                color_hline = 'red'
            else:
                color_hline = 'k'

        color_hline = 'red'
        #ax.axhline(y=exoplanet_period_in_hours, color=color_hline, linestyle='--', label=hline_label)

        
        if add_extra_T != None:
            ind_extra_T = 0
            for ind, value in add_extra_T.items():
                ind_extra_T=ind_extra_T+1
                #ax.axhline(y=value['T_value'], color=color_hline, linestyle='--', alpha=0.7, label=f"{value['T_name']}")
                line, = ax.plot(
                    [times_plot[0], times_plot[-1]],
                    [value['T_value'], value['T_value']],
                    color=color_hline,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    label=f"{value['T_name']}"
                )
                line.set_dashes([10, 20])  # [dash length, space length]

                # Adding arrows at the edges
                # Left arrow
                ax.annotate(
                    '',  # No text
                    xy=(times_plot[0] , value['T_value']),  # Arrowhead position
                    xytext=(times_plot[0] - 0.05 * (times_plot[-1] - times_plot[0]), value['T_value']),  # Start of the arrow
                    textcoords='data',  # Position in axes fraction coordinates
                    arrowprops=dict(arrowstyle='-|>', color=color_hline, lw=1.5)
                )

                # Right arrow
                ax.annotate(
                    '',  # No text
                    xy=(times_plot[-1], value['T_value']),  # Arrowhead position
                    xytext=(times_plot[-1] + 0.05 * (times_plot[-1] - times_plot[0]), value['T_value']),  # Start of the arrow
                    textcoords='data',  # Position in axes fraction coordinates
                    arrowprops=dict(arrowstyle='-|>', color=color_hline, lw=1.5)
                )



        ax.legend()
        ax.legend(
                loc='upper left',  
                bbox_to_anchor=(1.35, 1),  
                borderaxespad=0,  
#                fontsize=fontsize-2,
                ) 


    #ax.xaxis.set_major_locator(mdates.YearLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    #ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month


    if x_zoomin != None:
        axes[-1].set_xlim(x_zoomin)    
    if LS_y_zoomin != None:
        axes[-1].set_ylim(LS_y_zoomin)    
    
    if extra_panel_data != None:
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0])
        ax.plot(extra_panel_data[0], extra_panel_data[-1])
        if extra_panel_ytitle != None:
            ax.set_ylabel(extra_panel_ytitle)
        i_ax = i_ax+1
        axes.append(ax)

    
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month    
    axes[-1].set_xlabel('Time (years)')
    
    
    plt.tight_layout()
    if savefig:
        plt.savefig(filename, transparent=True)
    else:
        plt.show()

    plt.close()


def calculate_and_plot_SNR_LS_periodogram_over_time(time_datetime_beam_ON, frequencies_beam_ON, data_final_beam_ON,
                                                exoplanet_period_in_hours = 12.993,
                                                hline_label = r'T$_{Input}$',
                                                LS_window_size = 10000, # Number in minutes
                                                LS_step_size = 500, # Number in minutes
                                                observed_frequency_limits = [8,80],
                                                title_main = f'Beam On',
                                                cmap = 'bwr',
                                                color_hline = None,
                                                x_zoomin = None, # datetime object
                                                LS_y_zoomin = None,
                                                log_y_scale = False, 
                                                add_extra_T = None,
                                                vmin = None, vmax = None,
                                                savefig = False, filename = 'SNR_LS_periodogram_over_time.png',
                                                stokes = 'V/I',
                                                extra_panel_data = None,
                                                extra_panel_ytitle = None,
                                                figsize = None):
    
                 # add_extra_T needs to be a dictionnary like: {T_value: float, T_name: string}/
    #     e.g., add_extra_T = {
    #                           'Europa': {'T_value': 11.2321, 'T_name': 'Europa'},
    #                           'Ganymede': {'T_value': 10.5330, 'T_name': 'Ganymede'}
    #                          }


    if len(observed_frequency_limits) == 1:
        frequencies_to_be_used = frequencies_beam_ON[frequencies_beam_ON == observed_frequency_limits[0]]
    
    else:
        # Find the index of the first value less than or equal to the lower limit
        lower_index = numpy.max(numpy.where(frequencies_beam_ON <= observed_frequency_limits[0]))

        # Find the index of the first value greater than or equal to the upper limit
        upper_index = numpy.min(numpy.where(frequencies_beam_ON >= observed_frequency_limits[1]))

        # Select the values that bound the range
        frequencies_to_be_used = frequencies_beam_ON[lower_index:upper_index + 1]


    n_panels = len(frequencies_to_be_used) + 1
    
    if extra_panel_data != None:
        n_panels = n_panels+1
        height_ratios = [1] * (n_panels - 1) + [0.5]
    else:
        height_ratios = [1]*n_panels
    
    if figsize == None:
        figsize=(10,5*n_panels)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_panels, 2, width_ratios=[4, 1], height_ratios=height_ratios, wspace=0.3)
    axes = []
    ax = plt.subplot(gs[0,0])
    axes.append(ax)
    

    i_ax = 1
    
    for i_freq in frequencies_to_be_used:  
        print(f'Frequency: {i_freq} MHZ')
        time_real_beam_ON = datetime_to_timestamp(time_datetime_beam_ON)
        #time_numeric = (time_real_beam_ON - time_real_beam_ON[0])  # Convert to seconds since start

        data_for_LS_beam_ON = data_final_beam_ON[:, frequencies_beam_ON==i_freq][:,0]
        if stokes == 'V/I':
            mask_nonphysical = (numpy.abs(data_for_LS_beam_ON) > 1) + (numpy.isinf(data_for_LS_beam_ON)) + (numpy.isnan(data_for_LS_beam_ON))
            #data_for_LS_beam_ON[numpy.abs(data_for_LS_beam_ON) > 1] = numpy.nan # unphysical points
            #data_for_LS_beam_ON[numpy.isnan(data_for_LS_beam_ON)] = 0
            #data_for_LS_beam_ON[numpy.isinf(data_for_LS_beam_ON)] = 0
            
            time_real_for_LS_beam_ON = time_real_beam_ON[~mask_nonphysical]
            time_datetime_for_LS_beam_ON = time_datetime_beam_ON[~mask_nonphysical]
            data_for_LS_beam_ON = data_for_LS_beam_ON[~mask_nonphysical]

        elif stokes == 'V':
            mask_nonphysical = (numpy.abs(data_for_LS_beam_ON) > 1e9) + (numpy.isinf(data_for_LS_beam_ON)) + (numpy.isnan(data_for_LS_beam_ON))
            time_real_for_LS_beam_ON = time_real_beam_ON[~mask_nonphysical]
            time_datetime_for_LS_beam_ON = time_datetime_beam_ON[~mask_nonphysical]
            data_for_LS_beam_ON = data_for_LS_beam_ON[~mask_nonphysical]
            print(data_for_LS_beam_ON.shape, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), data_for_LS_beam_ON.mean())
            #data_for_LS_beam_ON = translate(data_for_LS_beam_ON, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), -1, 1)
            #print(data_for_LS_beam_ON.shape, data_for_LS_beam_ON.min(), data_for_LS_beam_ON.max(), data_for_LS_beam_ON.mean())
            

        axes[0].scatter(time_datetime_for_LS_beam_ON, data_for_LS_beam_ON, s=10, label=f'{i_freq:.4f} MHz', alpha=0.5)
        
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0],sharey=axes[1] if i_ax > 1 else None)
        

        # Prepare storage for results
        power_results = []
        SNR_LS_results = []
        times_results = []

        # Sliding window approach
        time_delta_minutes_total = int((time_real_for_LS_beam_ON.max()-time_real_for_LS_beam_ON.min())/60)
        for start in range(0, time_delta_minutes_total - LS_window_size, LS_step_size):
            
            time_window_start = time_real_for_LS_beam_ON.min()+start*60
            time_window_end = time_window_start+LS_window_size*60
            print(f'{timestamp_to_datetime(time_window_start)}, {timestamp_to_datetime(time_window_end)}')
            # Select the data for the current window
            t_mask = (time_real_for_LS_beam_ON >= time_window_start) * (time_real_for_LS_beam_ON < time_window_end)
            time_window = time_real_for_LS_beam_ON[t_mask]
            data_window = data_for_LS_beam_ON[t_mask]

            if time_window.size != 0:
                # Calculate Lomb-Scargle power
                frequency_LS, power_LS = calculate_LS_periodogram(time_window, data_window, exoplanet_period_in_hours)
            
                SNR_tmp =  numpy.nanstd(power_LS) 
                SNR_LS = power_LS/SNR_tmp

                # Store results
                SNR_LS_results.append(SNR_LS)
                power_results.append(power_LS)
                # Store the center time of the current window
                times_results.append(numpy.mean(time_window))


        # Convert results to arrays for plotting
        power_results = numpy.array(power_results)
        SNR_LS_results = numpy.array(SNR_LS_results)
        times_results = numpy.array(times_results)
        times_plot = timestamp_to_datetime(times_results)

        

        if vmin==None:
            vmin = numpy.nanmin(SNR_LS_results)
        if vmax==None:
            vmax = numpy.nanmax(SNR_LS_results)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        #img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm, levels=20)
        img = ax.contourf(times_plot, 1/frequency_LS/60/60, SNR_LS_results.T, cmap=cmap, levels=numpy.linspace(vmin,vmax,20), extend='min', vmin=vmin, vmax=vmax)
        
        ax.set_ylabel(f'Periodicities (hours)'+'\n'+f'(for freq. {i_freq:.2f} MHz)')
        #ax.grid(True)
        #ax.set_yscale('log')
        #ax.set_title('Wavelet Power Spectrum')
        
        ax_cb = plt.subplot(gs[i_ax, 1])
        pos_cb = ax_cb.get_position()  # Get current position of the colorbar axis
        ax_cb.set_position([pos_cb.x0-0.05, pos_cb.y0, 0.02, pos_cb.height])  # Adjust width (0.02) and keep the same height

        cbar = plt.colorbar(img, cax=ax_cb, label='SNR')

        i_ax = i_ax+1
        axes.append(ax)

    axes[0].set_ylabel('Stokes V/I')
    axes[0].set_title(title_main)
    axes[0].set_ylim(-1,1)
    axes[0].legend(
                loc='upper left',  
                bbox_to_anchor=(1.05, 1),  
                borderaxespad=0,  
                #fontsize=fontsize-2,
                ) 
    axes[0].grid(True)

    for i_freq, ax in enumerate(axes[1:]):
        #ax.tick_params(axis='x', which='both', labelbottom=False)  # Remove ticks and labels
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel('LS Periodicities (hours)'+'\n'+f'(freq. {frequencies_to_be_used[i_freq]:.02f} MHz)')

        if log_y_scale:
            ax.set_yscale("log", base=10)

        if color_hline == None:
            if cmap == 'inferno':
                color_hline = 'red'
            else:
                color_hline = 'k'

        color_hline = 'red'
        #ax.axhline(y=exoplanet_period_in_hours, color=color_hline, linestyle='--', label=hline_label)

        
        if add_extra_T != None:
            ind_extra_T = 0
            for ind, value in add_extra_T.items():
                ind_extra_T=ind_extra_T+1
                #ax.axhline(y=value['T_value'], color=color_hline, linestyle='--', alpha=0.7, label=f"{value['T_name']}")
                line, = ax.plot(
                    [times_plot[0], times_plot[-1]],
                    [value['T_value'], value['T_value']],
                    color=color_hline,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    label=f"{value['T_name']}"
                )
                line.set_dashes([10, 20])  # [dash length, space length]

                # Adding arrows at the edges
                # Left arrow
                ax.annotate(
                    '',  # No text
                    xy=(times_plot[0] , value['T_value']),  # Arrowhead position
                    xytext=(times_plot[0] - 0.05 * (times_plot[-1] - times_plot[0]), value['T_value']),  # Start of the arrow
                    textcoords='data',  # Position in axes fraction coordinates
                    arrowprops=dict(arrowstyle='-|>', color=color_hline, lw=1.5)
                )

                # Right arrow
                ax.annotate(
                    '',  # No text
                    xy=(times_plot[-1], value['T_value']),  # Arrowhead position
                    xytext=(times_plot[-1] + 0.05 * (times_plot[-1] - times_plot[0]), value['T_value']),  # Start of the arrow
                    textcoords='data',  # Position in axes fraction coordinates
                    arrowprops=dict(arrowstyle='-|>', color=color_hline, lw=1.5)
                )



        ax.legend()
        ax.legend(
                loc='upper left',  
                bbox_to_anchor=(1.35, 1),  
                borderaxespad=0,  
#                fontsize=fontsize-2,
                ) 


    #ax.xaxis.set_major_locator(mdates.YearLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    #ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month


    if x_zoomin != None:
        axes[-1].set_xlim(x_zoomin)    
    if LS_y_zoomin != None:
        axes[-1].set_ylim(LS_y_zoomin)    
    
    if extra_panel_data != None:
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0])
        ax.plot(extra_panel_data[0], extra_panel_data[-1])
        if extra_panel_ytitle != None:
            ax.set_ylabel(extra_panel_ytitle)
        i_ax = i_ax+1
        axes.append(ax)

    
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month    
    axes[-1].set_xlabel('Time (years)')
    
    
    plt.tight_layout()
    if savefig:
        plt.savefig(filename, transparent=True)
    else:
        plt.show()

    plt.close()

def plot_SNR_LS_periodogram_2D(frequency_LS, frequencies, power_LS_2D,
                            T_search = 12.992/24, T_name = r'Io$_{syn}$',
                            add_extra_T = None, x_zoomin = None, y_zoomin = None, vmin = None, vmax = None,
                            log_x_scale = False,
                            cmap = 'inferno',
                            savefig = False, filename = 'SNR_LS_2D_periodogram.pdf'):

    SNR_LS_2D = numpy.zeros(power_LS_2D.shape)
    for i_freq, freq in enumerate(frequencies):
        SNR_tmp =  numpy.nanstd(power_LS_2D[i_freq,:]) 
        SNR_LS_2D[i_freq,:] = power_LS_2D[i_freq,:]/SNR_tmp

    fig, ax = plt.subplots(figsize=(10, 6))

    if vmin==None:
        vmin = numpy.nanmin(SNR_LS_2D)
    if vmax==None:
        vmax = numpy.nanmax(SNR_LS_2D)
    print(vmin,vmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm, levels=20)        
    im = ax.contourf(1/frequency_LS/60/60, frequencies, SNR_LS_2D, cmap=cmap, levels=numpy.linspace(vmin,vmax,20), extend='min', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='SNR')

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    if x_zoomin == None:
        ax.set_xlim(numpy.min(1/frequency_LS/60/60), numpy.max(1/frequency_LS/60/60))
    else:
        ax.set_xlim(x_zoomin)
    if y_zoomin == None:
        ax.set_ylim(numpy.min(frequencies), numpy.max(frequencies))
    else:
        ax.set_ylim(y_zoomin)

    if log_x_scale:
        ax.set_xscale("log", base=10)

    specific_periodicities = [T_search*24]
    specific_periodicities_name = [T_name]
    if add_extra_T != None:
        ind_extra_T = 0
        for ind, value in add_extra_T.items():
            ind_extra_T=ind_extra_T+1
            specific_periodicities.append(value['T_value'])
            specific_periodicities_name.append(value['T_name'])
        # Convert periodicities to your x-axis units
    # Create a secondary x-axis at the top for periodicities
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(specific_periodicities)  # Set the tick positions
    secax.set_xticklabels([f'{p}' for p in specific_periodicities_name])  # Label the ticks (e.g., "1h", "5h", "10h")
    secax.set_xlabel('Searched Periodicities (hours)')  # Label for the top axis

    ax.set_xlabel('Periodicities (hours)')
    ax.set_ylabel('Frequency (MHz)')

    if savefig:
        plt.savefig(filename, format='png', transparent = True, dpi=500)
    else:
        plt.show()

    

def plot_LS_2D_periodogram(frequency_LS, frequencies, power_LS_2D,
                            T_search = 12.992/24, T_name = r'Io$_{syn}$',
                            add_extra_T = None, x_zoomin = None, y_zoomin = None, vmin = None, vmax = None,
                            log_x_scale = False,
                            cmap = 'inferno',
                            savefig = False, filename = 'LS_2D_periodogram.pdf'):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    if vmin==None:
        vmin = numpy.nanmin(power_LS_2D)
    if vmax==None:
        vmax = numpy.nanmax(power_LS_2D)
    print(vmin,vmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm, levels=20)        
    im = ax.contourf(1/frequency_LS/60/60, frequencies, power_LS_2D, cmap=cmap, levels=numpy.linspace(vmin,vmax,20), extend='min', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Lomb-Scargle Power')

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    if x_zoomin == None:
        ax.set_xlim(numpy.min(1/frequency_LS/60/60), numpy.max(1/frequency_LS/60/60))
    else:
        ax.set_xlim(x_zoomin)
    if y_zoomin == None:
        ax.set_ylim(numpy.min(frequencies), numpy.max(frequencies))
    else:
        ax.set_ylim(y_zoomin)

    if log_x_scale:
        ax.set_xscale("log", base=10)

    specific_periodicities = [T_search*24]
    specific_periodicities_name = [T_name]
    if add_extra_T != None:
        ind_extra_T = 0
        for ind, value in add_extra_T.items():
            ind_extra_T=ind_extra_T+1
            specific_periodicities.append(value['T_value'])
            specific_periodicities_name.append(value['T_name'])
        # Convert periodicities to your x-axis units
    # Create a secondary x-axis at the top for periodicities
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(specific_periodicities)  # Set the tick positions
    secax.set_xticklabels([f'{p}' for p in specific_periodicities_name])  # Label the ticks (e.g., "1h", "5h", "10h")
    secax.set_xlabel('Searched Periodicities (hours)')  # Label for the top axis

    ax.set_xlabel('Periodicities (hours)')
    ax.set_ylabel('Frequency (MHz)')

    if savefig:
        plt.savefig(filename, format='png', transparent = True, dpi=500)
    else:
        plt.show()
        

    
