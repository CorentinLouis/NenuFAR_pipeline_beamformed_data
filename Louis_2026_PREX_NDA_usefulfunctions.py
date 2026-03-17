import numpy
import math
from scipy.io import readsav

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.colors as mcolors

import datetime

from processing_data import read_hdf5_file, datetime_to_timestamp, timestamp_to_datetime

def doy_specific_year_to_yyyyddd(doy, origin):  
### Function to change "day of a specific year" format to yyyyddd ###

    aa = numpy.arange(61, dtype=float)+origin  # array of years starting from year of origin
    deb = numpy.zeros([61], dtype=float)  # zeros
    for i in range(1, len(deb)):  # categorising start point for each year
        if i % 4 == 1:
            deb[i:] = deb[i:]+366.
        else:
            deb[i:] = deb[i:]+365.

    yyyyddd = numpy.zeros(len(doy), dtype=float)

    for i in range(0, len(doy)):
        j = doy[i]-deb

        if j[0]>=1 == False:
            yyyyddd[i] = (aa[j >= 1][-1])*1000.+j[j >= 1][-1]
        else:
            yyyyddd[i] = (aa[j >= 0][-1])*1000.+1+j[j >= 0][-1]

    return(yyyyddd)


def doy_to_datetime(time_doy):
    ### Function to change a doy (in floating yyyyddd format) to a datetime object datetime.datetime(YYYY, mm, dd, HH, MM, SS)
    time_hours = [int((itime-int(itime))*24) for itime in (time_doy)]
    time_minutes = [int(((time_doy[itime]-int(time_doy[itime]))*24-time_hours[itime])*60) for itime in range(len(time_doy))]
    time_seconds = [int((((time_doy[itime]-int(time_doy[itime]))*24-time_hours[itime])*60-time_minutes[itime])*60) for itime in range(len(time_doy))]
    time = [datetime.datetime.strptime(f'{int(time_doy[itime])}T{time_hours[itime]:02d}:{time_minutes[itime]:02d}:{time_seconds[itime]:02d}', "%Y%jT%H:%M:%S") for itime in range(len(time_doy))]
    return(time)

@numpy.vectorize
def datetime_to_timestamp_old(datetime_table):
    ### Function to return time in floating format (from a datetime object)
    return datetime_table.timestamp()





def restore_cat(yyyy_min, yyyy_max, duration_min, intensity_min, fmax_min, fmax_max,
                catalog_path = "/Users/clouis/Documents/Data/NDA/catalogue/new_catalog_.sav"):
    # Load the .sav file
    catalog = readsav(catalog_path)
    
    emg = catalog['emg']
    emeph = catalog['emeph']
    obsg = catalog['obsg']
    obs = catalog['obs']
    obseph = catalog['obseph']
    num = catalog['num']
    em = catalog['em']
    nobs = catalog['nobs']
    nemg = catalog['nemg']
    
    # Calculate duration
    duration = emg['t_max'] - emg['t_min']
    
    # Filter the emg and emeph arrays
    w = numpy.where((emg['ymd'] >= yyyy_min * 10000) &
                 (emg['ymd'] <= yyyy_max * 10000 + 1231) &
                 (duration >= duration_min) &
                 (emg['iav'] >= intensity_min) &
                 (emg['f_max'] >= fmax_min) &
                 (emg['f_max'] <= fmax_max))[0]
    
    emg = emg[w]
    emeph = emeph[w]
    n_emg = len(w)
    duration = duration[w]
    
    # Filter the obsg, obs, and obseph arrays
    w = numpy.where((obsg['ymd'] >= yyyy_min * 10000) &
                 (obsg['ymd'] <= yyyy_max * 10000 + 1231))[0]
    
    obs = obs[w]
    obsg = obsg[w]
    obseph = obseph[w]
    n_obs = len(w)
    num = num[:n_obs]
    
    # Filter the nobs, em, and nemg arrays
    w = numpy.where((nobs >= min(w)) & (nobs <= max(w)))[0]
    
    nobs = nobs[w]
    em = em[w]
    nemg = nemg[w]
    n_em = len(w)
    
    return obs, em, obsg, emg, obseph, emeph, n_obs, n_em, n_emg, nemg, nobs, num


def plot_SNR_LS_over_time(times_results,
                          frequency_LS,
                          SNR_LS_results,
                          frequencies_beam_ON,
                          observed_frequency_limits = [8,80],
                          extra_panel_data = None,
                          figsize = None,
                          vmin = None,
                          vmax = None,
                          vmin_auto_percentile_value = 99.99,
                          x_zoomin = None,
                          LS_y_zoomin = None,
                          extra_panel_ytitle = None,
                          cmap = 'Greys',
                          log_y_scale= None,
                          color_hline = None,
                          add_extra_T = None,
                          fontsize = 20,
                          hlines = None,
                          vlines_regular = None
                          ):
    times_plot = timestamp_to_datetime(times_results)

    if len(observed_frequency_limits) == 1:
        frequencies_to_be_used = frequencies_beam_ON[frequencies_beam_ON == observed_frequency_limits[0]]
    
    else:
        # Find the index of the first value less than or equal to the lower limit
        lower_index = numpy.max(numpy.where(frequencies_beam_ON <= observed_frequency_limits[0]))

        # Find the index of the first value greater than or equal to the upper limit
        upper_index = numpy.min(numpy.where(frequencies_beam_ON >= observed_frequency_limits[1]))

        # Select the values that bound the range
        frequencies_to_be_used = frequencies_beam_ON[lower_index:upper_index + 1]


    n_panels = len(frequencies_to_be_used)
    
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
    

    i_ax = 0

        
    for i_freq in frequencies_to_be_used:  
        
        
        if vmin == "auto":
            vmin = numpy.nanpercentile(SNR_LS_results,vmin_auto_percentile_value)
        if vmin==None:
            vmin = numpy.nanmin(SNR_LS_results)
        if vmax==None:
            vmax = numpy.nanmax(SNR_LS_results)
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        #img = ax.contourf(times_plot, 1/frequency_LS/60/60, power_results.T, extend='both', cmap=cmap, zorder=0, norm = norm, levels=20)
        img = ax.contourf(
                    times_plot, 1/frequency_LS/60/60, numpy.ma.masked_less(SNR_LS_results, vmin).T,
                    cmap=cmap,
                    levels=numpy.linspace(vmin,vmax,20),
                    extend='max',
                    vmin=vmin, vmax=vmax,
                    zorder = 1)
        #ax.grid(True)
        #ax.set_yscale('log')
        #ax.set_title('Wavelet Power Spectrum')
        
        ax_cb = plt.subplot(gs[i_ax, 1])
        pos_cb = ax_cb.get_position()  # Get current position of the colorbar axis
        ax_cb.set_position([pos_cb.x0-0.05, pos_cb.y0, 0.02, pos_cb.height])  # Adjust width (0.02) and keep the same height

        cbar = plt.colorbar(img, cax=ax_cb, label='SNR')

        i_ax = i_ax+1
        axes.append(ax)


    for i_freq, ax in enumerate(axes[0:]):
        #ax.tick_params(axis='x', which='both', labelbottom=False)  # Remove ticks and labels
        
        if len(frequencies_to_be_used) != 1:
            ax.set_ylabel('LS Periodicities (hours)'+'\n'+f'(freq. {frequencies_to_be_used[i_freq]:.02f} MHz)', fontsize = fontsize+2)
        else:
            ax.set_ylabel('LS Periodicities (hours)', fontsize= fontsize+2)

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
                    label=f"{value['T_name']}",
                    zorder = 0
                )
                #line.set_dashes([10, 20])  # [dash length, space length]
                line.set_dashes([5, 2.5])  # [dash length, space length]

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
            ax.set_xlim(x_zoomin)    
        if LS_y_zoomin != None:
            ax.set_ylim(LS_y_zoomin) 
            major_loc = (LS_y_zoomin[-1]-LS_y_zoomin[0])/5
            minor_loc = major_loc/10
        else:
            major_loc = 5
            minor_loc = 1
        ax.yaxis.set_major_locator(MultipleLocator(major_loc))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_loc))   
            
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Set formatter to show only the year
        ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month    
        ax.set_xlabel('Time (years)', fontsize=fontsize+2)

    if hlines is not None:
        for i_hlines in hlines:
            ax.hlines(i_hlines, xmin = times_plot[0], xmax = times_plot[-1])
    if vlines_regular is not None:
        n_vlines = int(((times_results[-1]-times_results[0])/60/60/24)/vlines_regular)
        print(f'Number of Vertical Lines is {n_vlines}')
        for i_vlines in range(n_vlines):
            ax.vlines(times_plot[0]+datetime.timedelta(days=vlines_regular*i_vlines),
                      ymin= 1/frequency_LS[-1]/60/60,
                      ymax = 1/frequency_LS[0]/60/60,
                      colors='k',
                      linestyles='dashed')

    if extra_panel_data != None:
        ax = plt.subplot(gs[i_ax,0], sharex=axes[0])
        ax.plot(extra_panel_data[0], extra_panel_data[-1])
        if extra_panel_ytitle != None:
            ax.set_ylabel(extra_panel_ytitle)
        i_ax = i_ax+1
        axes.append(ax)
        
    plt.tight_layout()


def organize_catalogue_for_Louis_PREX_2026(obs, em, obsg, emg, obseph, emeph, n_obs, n_em, n_emg, nemg, nobs, num ):



    # Compute the minimum t90
    tmin = numpy.min(obsg['t90'])

    # Calculate nmin
    nmin = (numpy.max(obsg['t90']) - tmin + 1) * 1440.0
    nmin = math.ceil(nmin / 10000) * 10000

    # Initialize x array
    x = numpy.zeros(int(nmin), dtype=float)
    fmin = numpy.zeros(int(nmin), dtype = float)
    fmax = numpy.zeros(int(nmin), dtype = float)
    x_io = numpy.zeros(int(nmin), dtype = float)
    x_non_io = numpy.zeros(int(nmin), dtype = float)

    # Check the type and filter emg
    type = 'all'  # or set it according to your context
    if type.lower() == 'all':
        w = numpy.where((emg['source'] >= 0) & (emg['source'] <= 11))[0]

    nw = len(w)

    # Assuming INT and TAV are defined
    INT = False  # or set according to your context
    TAV = 0  # or set according to your context

    if not INT and TAV == 0:
        for i in range(nw):
            indices = (emg[w[i]]['t90'] - tmin) * 1440 + emg[w[i]]['time'][:emg[w[i]]['npts']]
            #ALL
            if emg[w[i]]['source'] in (1, 2, 3, 4, 5, 8, 9):
                x[indices.astype(int)] = -1.0
            else:
                x[indices.astype(int)] = 1.0
            if emg[w[i]]['source'] == 0:
                x[indices.astype(int)] = 0
            

            # Io only
            if emg[w[i]]['source'] in (1, 2, 3, 4, 5, 6, 7):
                if emg[w[i]]['source'] in (1, 2, 3, 4, 5):
                    x_io[indices.astype(int)] = -1.0
                if emg[w[i]]['source'] in (6, 7):
                    x_io[indices.astype(int)] = 1.0
            else:
                x_io[indices.astype(int)] = 0

            # Non Io only
            if emg[w[i]]['source'] in (0, 1, 2, 3, 4, 5, 6, 7):
                x_non_io[indices.astype(int)] = 0
            if emg[w[i]]['source'] in (8,9):
                x_non_io[indices.astype(int)] = -1
            if emg[w[i]]['source'] in (10, 11):
                x_non_io[indices.astype(int)] = 1

            fmin[indices.astype(int)] = emg[w[i]]['fmin'][:emg[w[i]]['npts']]
            fmax[indices.astype(int)] = emg[w[i]]['fmax'][:emg[w[i]]['npts']]            

    xt = numpy.zeros(int(nmin), dtype=float)
    for i in range(n_obs):
        indices = (obsg[i]['t90'] - tmin) * 1440 + obsg[i]['time'][:obsg[i]['npts']]
        xt[indices.astype(int)] = 1.0

    w = numpy.where(xt == 1.0)[0]
    t_obsg = numpy.arange(int(nmin), dtype=float)
    t = t_obsg[w]
    x = x[w]
    x_non_io = x_non_io[w]
    x_io = x_io[w]
    fmin = fmin[w]
    fmax = fmax[w]

    return (t_obsg, t, x, x_non_io, x_io, fmin, fmax)