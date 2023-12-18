# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy

import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import get_cmap
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_dynamic_spectrum(self, stokes: str = 'V', decibel: bool = True, masked_data: bool = False, **kwargs):

    """ Plots the dynamical spectrum corresponding to the given Stokes parameter. 
        If 'figname' is given, the plot will be saved with the given path.
        :param self:
            Contains data, time and frequency arrays
        type self:
            ReadFits_data object
        :param stokes:
            Stokes parameter to plot.
        :type stokes:
            str
        :param decibel:
            If set, the dynamical spectrum will be plot in a logarithmic scale. Default is True.
        :type decibel:
            bool
        :param masked_data:
            Keyword to plot the L1 data or the masked data
        :type masked_data:
            bool
    """

    if stokes[0] != 'L':
        stokes_index = {
            'I': 0,
            'Q': 1,
            'U': 2,
            'V': 3,
            }
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    if masked_data:
        data = self.data_masked
    else:
        data = self.data
    data[data == 0] = numpy.nan
    if decibel:
        data_to_plot = 10*numpy.log10(data[:, :, stokes_index[stokes[0]]])
        unit_label = 'dB'
    else:
        data_to_plot = data[:, :, stokes_index[stokes[0]]]
        unit_label = 'Amp'
    
    if stokes[1:3] == '/I':
        data_to_plot = data_to_plot / data[:, :, stokes_index['I']]
    if stokes[0] != 'I':
        vmin=kwargs.get("vmin", -numpy.nanpercentile(data_to_plot, 95))
        vmax=kwargs.get("vmax", numpy.nanpercentile(data_to_plot, 95))
    else:
        vmin=kwargs.get("vmin", numpy.nanpercentile(data_to_plot, 1))
        vmax=kwargs.get("vmax", numpy.nanpercentile(data_to_plot, 95))
    im = ax.pcolormesh(
        self.time.datetime,
        self.frequency.to(u.MHz).value,
        data_to_plot.T,
        shading="auto",
        cmap=kwargs.get("cmap", "viridis"),
        vmin=vmin,
        vmax=vmax
    )

    plt.xlabel(f"Time (hours of day {self.time[0].datetime.strftime('%Y-%m-%d')})")
    plt.ylabel('Frequency [MHz]')
    plt.title(kwargs.get("title","SPDYN"))

    dateFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)

    cax = inset_axes(
        ax,
        width='3%',
        height='100%',
        loc='lower left',
        bbox_to_anchor=(1.03, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(f'{stokes} ({unit_label})')


    figname = kwargs.get("figname", "")
    if figname != "":
        plt.savefig(
            figname,
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )
    plt.show()
    plt.close('all')