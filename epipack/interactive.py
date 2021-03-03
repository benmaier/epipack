"""
Interactive Jupyter widgets for SymbolicEpiModels.
"""

import copy
from collections import OrderedDict
from math import log10

import numpy as np
import sympy

import ipywidgets as widgets

import matplotlib.pyplot as pl

from epipack.colors import palettes, hex_colors

def get_box_layout():
    """Return default box layout"""
    return widgets.Layout(
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
    )

class Range(dict):
    """
    Defines a value range for an interactive linear
    value slider.

    Parameters
    ==========
    min : float
        Minimal value of parameter range
    max : float
        Maximal value of parameter range
    step_count : int, default = 100
        Divide the parameter space into that
        many intervals
    value : float, default = None
        Initial value. If ``None``, defaults to the
        mean of ``min`` and ``max``.
    """

    def __init__(self,
                 min,
                 max,
                 step_count=100,
                 value=None):

        super().__init__()

        assert(max > min)
        assert(step_count>0)

        self['min'] = min
        self['max'] = max
        if value is None:
            self['value'] = 0.5*(max+min)
        else:
            assert(min <= value and max >= value)
            self['value'] = value
        self['step'] = (max-min)/step_count

    def __float__(self):
        return float(self['value'])

    def __add__(self, other):
        return other + float(self)

    def __radd__(self, other):
        return other + float(self)

    def __mul__(self, other):
        return other * float(self)

    def __rmul__(self, other):
        return other * float(self)

    def __truediv__(self, other):
        return float(self) / other

    def __rtruediv__(self, other):
        return other / float(self)

    def __pow__(self, other):
        return float(self)**other

    def __rpow__(self, other):
        return other**float(self)

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)

class LogRange(dict):
    """
    Defines a value range for an interactive logarithmic
    value slider.

    Parameters
    ==========
    min : float
        Minimal value of parameter range
    max : float
        Maximal value of parameter range
    step_count : int, default = 100
        Divide the exponent space into that
        many intervals
    base : float, default = 10
        Base of the logarithm
    value : float, default = None
        Initial value. If ``None``, defaults to the
        geometric mean of ``min`` and ``max``.
    """

    def __init__(self,
                 min,
                 max,
                 step_count=100,
                 value=None,
                 base=10,
                 ):

        super().__init__()

        assert(max > min)
        assert(step_count>0)
        assert(base>0)

        def logB(x):
            return np.log(x) / np.log(base)

        self['min'] = logB(min)
        self['max'] = logB(max)

        if value is None:
            self['value'] = np.sqrt(max*min)
        else:
            assert(min <= value and max >= value)
            self['value'] = value

        self['step'] = (logB(max)-logB(min))/step_count
        self['base'] = base

    def __float__(self):
        return float(self['value'])

    def __add__(self, other):
        return other + float(self)

    def __radd__(self, other):
        return other + float(self)

    def __mul__(self, other):
        return other * float(self)

    def __rmul__(self, other):
        return other * float(self)

    def __truediv__(self, other):
        return float(self) / other

    def __rtruediv__(self, other):
        return other / float(self)

    def __pow__(self, other):
        return float(self)**other

    def __rpow__(self, other):
        return other**float(self)

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)

class InteractiveIntegrator(widgets.HBox):
    """
    An interactive widget that lets you control parameters
    of a SymbolicEpiModel and shows you the output.

    Based on this tutorial: https://kapernikov.com/ipywidgets-with-matplotlib/

    Parameters
    ==========
    model : epipack.symbolic_epi_models.SymbolicEpiModel
        An instance of ``SymbolicEpiModel`` that has been initiated
        with initial conditions
    parameter_values : dict
        A dictionary that maps parameter symbols to single, fixed values
        or ranges (instances of :class:`epipack.interactive.Range` or
        :class:`epipack.interactive.LogRange`).
    t : numpy.ndarray
        The time points over which the model will be integrated
    return_compartments : list, default = None
        A list of compartments that should be displayed.
        If ``None``, all compartments will be displayed.
    return_derivatives : list, default = None
        A list of derivatives that should be displayed
        If ``None``, no derivatives will be displayed.
    figsize : tuple, default = (4,4)
        Width and height of the created figure.
    palette : str, default = 'dark'
        A palette from ``epipack.colors``. Choose from

        .. code:: python

            [ 'dark', 'light', 'dark pastel', 'light pastel',
              'french79', 'french79 pastel', 'brewer light',
              'brewer dark', 'brewer dark pastel', 'brewer light pastel'
            ]
    integrator : str, default = 'dopri5'
        Either ``euler`` or ``dopri5``.
    continuous_update : bool, default = False
        If ``False``, curves will be updated only if the mouse button
        is released. If ``True``, curves will be continuously updated.
    show_grid : bool, default = False
        Whether or not to display a grid

    Attributes
    ==========
    model : epipack.symbolic_epi_models.SymbolicEpiModel
        An instance of ``SymbolicEpiModel`` that has been initiated
        with initial conditions.
    fixed_parameters : dict
        A dictionary that maps parameter symbols to single, fixed values
    t : numpy.ndarray
        The time points over which the model will be integrated
    return_compartments : list
        A list of compartments that will be displayed.
    colors : list
        A list of hexstrings.
    fig : matplotlib.Figure
        The figure that will be displayed.
    ax : matplotlib.Axis
        The axis that will be displayed.
    lines : dict
        Maps compartments to line objects
    children : list
        Contains two displayed boxes (controls and output)
    continuous_update : bool, default = False
        If ``False``, curves will be updated only if the mouse button
        is released. If ``True``, curves will be continuously updated.
    """

    def __init__(self,
                 model,
                 parameter_values,
                 t,
                 return_compartments=None,
                 return_derivatives=None,
                 figsize=(4,4),
                 palette='dark',
                 integrator='dopri5',
                 continuous_update=False,
                 show_grid=False,
                ):

        super().__init__()

        self.model = model
        self.t = np.array(t)
        self.colors = [ hex_colors[colorname] for colorname in palettes[palette] ]
        if return_compartments is None:
            self.return_compartments = self.model.compartments
        else:
            self.return_compartments = return_compartments
        self.return_derivatives = return_derivatives
        self.integrator = integrator
        self.lines = None
        self.continuous_update = continuous_update

        output = widgets.Output()

        with output:
            self.fig, self.ax = pl.subplots(constrained_layout=True, figsize=figsize)
            self.ax.set_xlabel('time')
            self.ax.set_ylabel('frequency')
            self.ax.grid(show_grid)


        self.fig.canvas.toolbar_position = 'bottom'

        # define widgets
        self.fixed_parameters = {}
        self.sliders = {}
        for parameter, value in parameter_values.items():
            self.fixed_parameters[parameter] = float(value)
            if type(value) not in [Range, LogRange]:
                continue
            else:
                these_vals = copy.deepcopy(value)
                these_vals['description'] = r'\(' + sympy.latex(parameter) + r'\)'
                these_vals['continuous_update'] = self.continuous_update
                if type(value) == LogRange:
                    slider = widgets.FloatLogSlider(**these_vals)
                else:
                    slider = widgets.FloatSlider(**these_vals)
            self.sliders[parameter] = slider

        checkb_xscale = widgets.Checkbox(
            value=False,
            description='logscale time',
        )
        checkb_yscale = widgets.Checkbox(
            value=False,
            description='logscale frequency',
        )

        controls = widgets.VBox(
            list(self.sliders.values()) + [
                checkb_xscale,
                checkb_yscale,
        ])
        controls.layout = get_box_layout()

        out_box = widgets.Box([output])
        output.layout = get_box_layout()

        for parameter, slider in self.sliders.items():
            slider.observe(self.update_parameters, 'value')
        checkb_xscale.observe(self.update_xscale, 'value')
        checkb_yscale.observe(self.update_yscale, 'value')

        self.children = [controls, output]

        self.update_parameters()

    def update_parameters(self, *args, **kwargs):
        """Update the current values of parameters as given by slider positions."""
        parameters = copy.deepcopy(self.fixed_parameters)
        for parameter, slider in self.sliders.items():
            parameters[parameter] = slider.value

        self.update_plot(parameters)

    def update_plot(self, parameters):
        """Recompute and -draw the epidemic curves with updated parameter values"""

        self.model.set_parameter_values(parameters)

        if self.return_derivatives is None:
            res = self.model.integrate(
                    self.t,
                    return_compartments=self.return_compartments,
                    integrator=self.integrator)

        else:
            res = self.model.integrate_and_return_by_index(
                    self.t,
                    integrator=self.integrator)
            ndx = [ self.model.get_compartment_id(C) for C in self.return_derivatives ]
            dydt = self.model.get_numerical_dydt()
            derivatives = np.array([ dydt(t,res[:,it]) for it, t in enumerate(self.t) ]).T
            res = {C: res[self.model.get_compartment_id(C),:] for C in self.return_compartments}
            der = {C: derivatives[self.model.get_compartment_id(C),:] for C in self.return_derivatives}

        is_initial_run = self.lines is None
        if is_initial_run:
            self.lines = {}

        # plot compartments 
        for iC, C in enumerate(self.return_compartments):
            ydata = res[C]
            if is_initial_run:
                self.lines[C], = self.ax.plot(self.t,ydata,label=str(C),color=self.colors[iC])
            else:
                self.lines[C].set_ydata(ydata)

        # plot derivatives
        if self.return_derivatives is not None:

            for iC, C in enumerate(self.return_derivatives):
                ydata = der[C]
                _C = 'd' + str(C) + '/dt'
                if is_initial_run:
                    self.lines[_C], = self.ax.plot(self.t,ydata,ls='--',label=_C,color=self.colors[iC])
                else:
                    self.lines[_C].set_ydata(ydata)


        if is_initial_run:
            self.ax.legend()

        self.fig.canvas.draw()

    def update_xscale(self, change):
        """Update the scale of the x-axis. For "log", pass an object ``change`` that has ``change.new=True``"""
        scale = 'linear'
        if change.new:
            scale = 'log'
        self.ax.set_xscale(scale)

    def update_yscale(self, change):
        """Update the scale of the y-axis. For "log", pass an object ``change`` that has ``change.new=True``"""
        scale = 'linear'
        if change.new:
            scale = 'log'
        self.ax.set_yscale(scale)

class GeneralInteractiveWidget(widgets.HBox):
    """
    An interactive widget that lets you control parameters
    that are passed to a custom function which returns a result
    dictionary.

    Based on this tutorial: https://kapernikov.com/ipywidgets-with-matplotlib/

    Parameters
    ==========
    result_function : func
        A function that returns a result dictionary when passed
        parameter values as ``result_function(**parameter_values)``.
    parameter_values : dict
        A dictionary that maps parameter names to single, fixed values
        or ranges (instances of :class:`epipack.interactive.Range` or
        :class:`epipack.interactive.LogRange`).
    t : numpy.ndarray
        The time points corresponding to values in the result dictionary.
    return_keys : list, default = None
        A list of result keys that should be shown.
        If ``None``, all compartments will be displayed.
    figsize : tuple, default = (4,4)
        Width and height of the created figure.
    palette : str, default = 'dark'
        A palette from ``epipack.colors``. Choose from

        .. code:: python

            [ 'dark', 'light', 'dark pastel', 'light pastel',
              'french79', 'french79 pastel', 'brewer light',
              'brewer dark', 'brewer dark pastel', 'brewer light pastel'
            ]
    continuous_update : bool, default = False
        If ``False``, curves will be updated only if the mouse button
        is released. If ``True``, curves will be continuously updated.
    show_grid : bool, default = False
        Whether or not to display a grid
    ylabel : str, default = 'frequency'
        What to name the yaxis
    label_converter : func, default = str
        A function that returns a string when passed a result key
        or parameter name.


    Attributes
    ==========
    result_function : func
        A function that returns a result dictionary when passed
        parameter values as ``result_function(**parameter_values)``.
    fixed_parameters : dict
        A dictionary that maps parameter names to fixed values
    t : numpy.ndarray
        The time points corresponding to values in the result dictionary.
    return_keys : list
        A list of result dictionary keys of which the result
        will be displayed.
    colors : list
        A list of hexstrings.
    fig : matplotlib.Figure
        The figure that will be displayed.
    ax : matplotlib.Axis
        The axis that will be displayed.
    lines : dict
        Maps compartments to line objects
    children : list
        Contains two displayed boxes (controls and output)
    continuous_update : bool, default = False
        If ``False``, curves will be updated only if the mouse button
        is released. If ``True``, curves will be continuously updated.
    lbl : func, default = str
        A function that returns a string when passed a result key
        or parameter name.
    """

    def __init__(self,
                 result_function,
                 parameter_values,
                 t,
                 return_keys=None,
                 figsize=(4,4),
                 palette='dark',
                 continuous_update=False,
                 show_grid=False,
                 ylabel='frequency',
                 label_converter=str,
                ):

        super().__init__()

        self.t = t
        self.get_result = result_function
        self.colors = [ hex_colors[colorname] for colorname in palettes[palette] ]
        self.return_keys = return_keys
        self.lines = None
        self.continuous_update = continuous_update
        self.lbl = label_converter
        output = widgets.Output()

        with output:
            self.fig, self.ax = pl.subplots(constrained_layout=True, figsize=figsize)
            self.ax.set_xlabel('time')
            self.ax.set_ylabel(ylabel)
            self.ax.grid(show_grid)

        self.fig.canvas.toolbar_position = 'bottom'

        # define widgets
        self.fixed_parameters = {}
        self.sliders = {}
        for parameter, value in parameter_values.items():
            self.fixed_parameters[parameter] = float(value)
            if type(value) not in [Range, LogRange]:
                continue
            else:
                these_vals = copy.deepcopy(value)
                these_vals['description'] = self.lbl(parameter) or parameter
                these_vals['continuous_update'] = self.continuous_update
                if type(value) == LogRange:
                    slider = widgets.FloatLogSlider(**these_vals)
                else:
                    slider = widgets.FloatSlider(**these_vals)
            self.sliders[parameter] = slider

        checkb_xscale = widgets.Checkbox(
            value=False,
            description='logscale time',
        )
        checkb_yscale = widgets.Checkbox(
            value=False,
            description='logscale frequency',
        )

        controls = widgets.VBox(
            list(self.sliders.values()) + [
                checkb_xscale,
                checkb_yscale,
        ])
        controls.layout = get_box_layout()

        out_box = widgets.Box([output])
        output.layout = get_box_layout()

        for parameter, slider in self.sliders.items():
            slider.observe(self.update_parameters, 'value')
        checkb_xscale.observe(self.update_xscale, 'value')
        checkb_yscale.observe(self.update_yscale, 'value')

        self.children = [controls, output]

        self.update_parameters()

    def update_parameters(self, *args, **kwargs):
        """Update the current values of parameters as given by slider positions."""
        parameters = copy.deepcopy(self.fixed_parameters)
        for parameter, slider in self.sliders.items():
            parameters[parameter] = slider.value

        self.update_plot(parameters)

    def update_plot(self, parameters):
        """Recompute and -draw the epidemic curves with updated parameter values"""

        res = self.get_result(**parameters)

        is_initial_run = self.lines is None
        if is_initial_run:
            self.lines = {}

        if self.return_keys is None:
            keys = res.keys()
        else:
            keys = self.return_keys

        # plot compartments 
        for iC, C in enumerate(keys):
            ydata = res[C]
            if is_initial_run:
                self.lines[C], = self.ax.plot(self.t,ydata,label=self.lbl(C),color=self.colors[iC])
            else:
                self.lines[C].set_ydata(ydata)

        if is_initial_run:
            self.ax.legend()

        self.fig.canvas.draw()

    def update_xscale(self, change):
        """Update the scale of the x-axis. For "log", pass an object ``change`` that has ``change.new=True``"""
        scale = 'linear'
        if change.new:
            scale = 'log'
        self.ax.set_xscale(scale)

    def update_yscale(self, change):
        """Update the scale of the y-axis. For "log", pass an object ``change`` that has ``change.new=True``"""
        scale = 'linear'
        if change.new:
            scale = 'log'
        self.ax.set_yscale(scale)


if __name__=="__main__": # pragma: no cover

    A = LogRange(0.1,1,value=0.5)

    print(A + 2)
    print(2 + A)
    print(A * 2)
    print(2 * A)
    print(A / 2)
    print(2 / A)
    print(A**2)
    print(2**A)
    print(A - 2)
    print(2 - A)
