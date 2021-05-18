"""
Some helper plot functions based on matplotlib.
These are just some code snippets that I use
regularly and not supposed to be of great merit otherwise.

Install matplotlib and bfmplot to use this module

.. code::

    matplotlib>=3.0.0
    bfmplot>=0.1.0
"""

import matplotlib.pyplot as pl
import matplotlib as mpl

import numpy as np

from epipack.colors import hex_colors, palettes

colors = [hex_colors[c] for c in  palettes['dark']]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

def strip_axis(ax,horizontal='right'):
    """Remove the right and the top axis"""
    if horizontal == 'right':
        anti_horizontal = 'left'
    else:
        anti_horizontal = 'right'
        ax.yaxis.set_label_position("right")

    ax.spines[horizontal].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position(anti_horizontal)
    ax.xaxis.set_ticks_position('bottom')

def plot(t,result,ax=None, curve_label_format='{}',figsize=None):
    """
    Plot an epipack result.

    Parameters
    ==========
    t : numpy.ndarray
        Sampling times.
    result : dict
        Mapping compartments to incidence time series.
    ax : matplotlib.axis.Axis, default = None
        The axis on which to plot
    curve_label_format : str, default = '{}'
        How to display a curve label
    figsize : tuple, default = None
        A tuple containing width and height of the figure
        that's produced

    Returns
    =======
    ax : matplotlib.axis.Axis
        The axis on which results were drawn
    """
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=figsize)
    _res = list(result.values())[0]
    N = np.zeros_like(_res)
    for C, timeseries in result.items():
        ax.plot(t, timeseries, label=curve_label_format.format(str(C)))
        N += timeseries
    strip_axis(ax)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim([0,N.max()])
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    ax.get_figure().tight_layout()
    return ax
