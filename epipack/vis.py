"""
Visualizations of simulations with pyglet.
"""

from copy import deepcopy
from itertools import chain

import numpy as np

import pyglet
from pyglet import shapes
from pyglet.window import key, mouse, Window
from pyglet.gl import *

from epipack.networks import get_random_layout, get_grid_layout
from epipack import colors as col
from epipack.colors import colors

_colors = list(colors.values())


class SimulationStatus():
    """
    Saves information about the current simulation.

    Parameters
    ==========
    N : int
        Number of nodes
    sampling_dt : float
        The amount of simulation time that's supposed
        to pass during a single update

    Attributes
    ==========
    old_node_status : numpy.ndarray
        An array containing node statuses of the previous update
    sampling_dt : float
        The amount of simulation time that's supposed
        to pass during a single update
    simulation_ended : bool
        Whether or not the simulation is over
    paused : bool
        Whether or not the simulation is paused
    """

    def __init__(self,N,sampling_dt):
        self.old_node_status = -1e300*np.ones((N,))
        self.simulation_ended = False
        self.sampling_dt = sampling_dt
        self.paused = False

    def update(self,old_node_status):
        """
        Update the nodes statuses.
        """
        self.old_node_status = np.array(old_node_status)

    def set_simulation_status(self,simulation_ended):
        """
        Trigger the simulation to be over.
        """
        self.simulation_ended = simulation_ended


class App(pyglet.window.Window):
    """

    A pyglet Window class that makes zooming and panning convenient
    and tracks user input.

    Adapted from Peter Varo's solution
    at https://stackoverflow.com/a/19453006/4177832

    Parameters
    ==========
    width : float
        Width of the app window
    height : float
        Height of the app window
    simulation_status : SimulationStatus
        An object that tracks the simulation. Here,
        it's used to pause or increase the simulation speed.
    """

    def __init__(self, width, height, simulation_status, *args, **kwargs):
        #conf = Config(sample_buffers=1,
        #              samples=4,
        #              depth_size=16,
        #              double_buffer=True)
        self.left   = 0
        self.right  = width
        self.bottom = 0
        self.top    = height
        super().__init__(width, height, *args, **kwargs)
        self.batches = []
        self.batch_funcs = []

        #Initialize camera values
        self.left   = 0
        self.right  = width
        self.bottom = 0
        self.top    = height
        self.zoom_level = 1
        self.zoomed_width  = width
        self.zoomed_height = height

        # Set window values
        self.width  = width
        self.height = height

        self.orig_left = self.left
        self.orig_right = self.right
        self.orig_bottom = self.bottom
        self.orig_top = self.top
        self.orig_zoom_level = self.zoom_level
        self.orig_zoomed_width = self.zoomed_width
        self.orig_zoomed_height = self.zoomed_height

        self.simulation_status = simulation_status


    def add_batch(self,batch,prefunc=None):
        """
        Add a batch that needs to be drawn.
        Optionally, also pass a function that's
        triggered before this batch is drawn.
        """
        self.batches.append(batch)
        self.batch_funcs.append(prefunc)


    def on_draw(self):
        """
        Clear and draw all batches
        """
        self.clear()

        for batch, func in zip(self.batches, self.batch_funcs):
            if func is not None:
                func()
            batch.draw()

    def init_gl(self, width, height):
        # Set viewport
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        #try:
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )
        #except AttributeError as e:
        #    print(self.name) 

    def on_resize(self, width, height):
        """Rescale."""
        # Set window values
        self.width  = width
        self.height = height

        # Initialize OpenGL context
        self.init_gl(width, height)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Pan."""
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

    def on_mouse_scroll(self, x, y, dx, dy):
        """Zoom."""
        # Get scale factor
        f = 1+dy/50
        # If zoom_level is in the proper range
        if .02 < self.zoom_level*f < 10:

            self.zoom_level *= f

            mouse_x = x/self.width
            mouse_y = y/self.height

            mouse_x_in_world = self.left   + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width  *= f
            self.zoomed_height *= f

            self.left   = mouse_x_in_world - mouse_x*self.zoomed_width
            self.right  = mouse_x_in_world + (1 - mouse_x)*self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y*self.zoomed_height
            self.top    = mouse_y_in_world + (1 - mouse_y)*self.zoomed_height

        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )


    def on_key_press(self, symbol, modifiers):
        """
        Check for keyboard input.
        Current inputs:

        - backspace or CMD+0: reset view
        - up : increase simulation speed
        - down : decrease simulation speed
        - space : pause simulation
        """
        #if symbol & key.BACKSPACE or (symbol & key._0 and (modifiers & MOD_COMMAND or modifiers & MOD_CTRL)):
        if symbol == key.BACKSPACE or (symbol == key._0 and (modifiers & key.MOD_COMMAND)):
            self.left = self.orig_left
            self.right = self.orig_right
            self.bottom = self.orig_bottom
            self.top = self.orig_top
            self.zoom_level = self.orig_zoom_level
            self.zoomed_width = self.orig_zoomed_width
            self.zoomed_height = self.orig_zoomed_height

            glMatrixMode( GL_PROJECTION )
            glLoadIdentity()
            glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )
        elif symbol == key.UP:
            self.simulation_status.sampling_dt *= 1.2
        elif symbol == key.DOWN:
            self.simulation_status.sampling_dt /= 1.2
        elif symbol == key.SPACE:
            self.simulation_status.paused = not self.simulation_status.paused



class Scale():
    """
    A scale that maps all its connected graphics objects
    to world (window) dimensions.

    Parameters
    ==========
    bound_increase_factor : float, default = 1.0
        By how much the respective bound should increase
        once it's reached.

    Attributes
    ==========
    bound_increase_factor : float
        By how much the respective bound should increase
        once it's reached.
    x0 : float
        lower bound of data x-dimension
    x1 : float
        upper bound of data x-dimension
    y0 : float
        lower bound of data y-dimension
    y1 : float
        upper bound of data y-dimension
    left : float
        lower bound of world x-dimension
    right : float
        upper bound of world x-dimension
    bottom : float
        lower bound of world y-dimension
    top : float
        upper bound of world y-dimension
    scaling_objects : list
        A list of objects that need to be rescaled
        once the data or world dimensions change.
        Each entry of this list is assumed to be 
        an object that has a method called ``rescale()``.
    """

    def __init__(self,bound_increase_factor=1.0):

        self.x0 = np.nan
        self.y0 = np.nan
        self.x1 = np.nan
        self.y1 = np.nan
        self.left = np.nan
        self.right = np.nan
        self.bottom = np.nan
        self.top = np.nan

        self.bound_increase_factor = bound_increase_factor

        self.scaling_objects = []

    def extent(self,left,right,top,bottom):
        """
        Define the world (window) dimensions.
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self._calc()
        return self

    def domain(self,x0,x1,y0,y1):
        """
        Define the data dimensions.
        """
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self._calc()
        return self

    def _calc(self):
        """
        calculate scalars
        """
        self.mx = (self.right-self.left)/(self.x1 - self.x0)
        self.my = (self.top-self.bottom)/(self.y1 - self.y0)

    def scale(self,x,y):
        """
        Scale data.
        """
        _x = self.scalex(x)
        _y = self.scaley(y)

        return _x, _y

    def scalex(self,x):
        """
        Scale x-data
        """
        if type(x) == list:
            _x = list(map(lambda _x: self.mx * (_x-self.x0) + self.left, x ))
        else:
            _x = self.mx * (x-self.x0) + self.left

        return _x

    def scaley(self,y):
        """
        Scale y-data
        """
        if type(y) == list:
            _y = list(map(lambda _y: self.my * (_y-self.y0) + self.bottom, y ))
        else:
            _y = self.my * (y-self.y0) + self.bottom

        return _y

    def check_bounds(self,xmin,xmax,ymin,ymax):
        """
        Check whether the global data dimensions have changed
        considered updated data dimensions of a single instance.
        If this is the case, trigger rescaling of all connected
        instances.
        """
        changed = False
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        if xmin < self.x0:
            xvec = xmin - self.x1
            x0 = self.x1 + xvec * self.bound_increase_factor
            changed = True
        if ymin < self.y0:
            yvec = ymin - self.y1
            y0 = self.y1 + yvec * self.bound_increase_factor
            changed = True
        if xmax > self.x1:
            xvec = xmax - self.x0
            x1 = self.x0 + xvec * self.bound_increase_factor
            changed = True
        if ymax > self.y1:
            yvec = ymax - self.y0
            y1 = self.y0 + yvec * self.bound_increase_factor
            changed = True

        if changed:
            self.domain(x0,x1,y0,y1)
            for obj in self.scaling_objects:
                obj.rescale()

    def add_scaling_object(self,obj):
        """
        Append an object that depends on this Scale instance.
        """
        self.scaling_objects.append(obj)



class Curve():
    """
    A class that draws an OpenGL
    curve to a pyglet Batch instance
    with easy methods to update data.

    Parameters
    ==========
        x : list
            x data
        y : list
            y data
        color : list 
            List of 3 integers between 0 and 255 (RGB color list)
        scale : Scale
            An instance of a Scale that maps the data dimensions
            to an area in a pyglet Window

    Attributes
    ==========
        batch : pyglet.graphics.Batch
            The batch instance in which this curve is drawn
        scale : Scale
            An instance of a Scale that maps the data dimensions
            to an area in a pyglet Window
        vertex_list : pyglet.graphics.VertexList
            Contains the vertex list in window coordinates.
            format strings are ``v2f`` and ``c3B``.
        color : list
            as described above
        xmin : float
            lower bound of x-dimension
        xmax : float
            upper bound of x-dimension
        ymin : float
            lower bound of y-dimension
        ymax : float
            upper bound of y-dimension
    """

    def __init__(self,x,y,color,scale,batch):
        self.batch = batch
        self.vertex_list = batch.add(0, GL_LINE_STRIP, None,
                    'v2f',
                    'c3B',
                )
        self.scale = scale
        scale.add_scaling_object(self)
        self.color = color
        self.xmin = 1e300
        self.ymin = 1e300
        self.xmax = -1e300
        self.ymax = -1e300
        self.set(x,y)

    def set(self,x,y):
        """
        Set the data of this curve.
        """
        _x = list(x)
        _y = list(y)

        # GL_LINE_STRIP needs to have the first
        # and last vertex as duplicates,
        # save data
        self.x = _x[:1] + _x + _x[-1:]
        self.y = _y[:1] + _y + _y[-1:]

        # get min/max values of this update
        xmin = min(_x)
        xmax = max(_x)
        ymin = min(_y)
        ymax = max(_y)

        # scale and zip together the new numbers
        _x, _y = self.scale.scale(self.x, self.y)
        xy = list(chain.from_iterable(zip(_x, _y)))

        # resize vertex list, set vertices and colors
        self.vertex_list.resize(len(_x))
        self.vertex_list.vertices = xy
        self.vertex_list.colors = self.color * len(_x)

        # check whether or not the bounds of
        # the scale need to be updated
        self.update_bounds(xmin,xmax,ymin,ymax)

    def append_single_value(self,x,y):
        """
        Append a single data point to this curve.
        Note that if the bounds change with this
        update, the connect Scale-instance will be updated
        automatically.
        """
        self.append_list([x], [y])

    def append_list(self,x,y):
        """
        Append a list of data points to this curve.
        Note that if the bounds change with this
        update, the connect Scale-instance will be updated
        automatically.
        """
        _x = x + x[-1:]
        _y = y + y[-1:]

        # remember that self.x contains the last
        # vertex twice for GL_LINE_STRIP.
        # We have to pop the duplicate of the
        # formerly last entry and append the new list
        self.x.pop()
        self.x.extend(_x)
        self.y.pop()
        self.y.extend(_y)

        xmin = min(_x)
        xmax = max(_x)
        ymin = min(_y)
        ymax = max(_y)

        _x, _y = self.scale.scale(_x, _y)
        xy = list(chain.from_iterable(zip(_x, _y)))

        self.vertex_list.resize(self.vertex_list.get_size() + len(_x) -1 )
        self.vertex_list.vertices[-len(xy):] = xy
        self.vertex_list.colors[-3*len(_x):] = self.color * len(_x)

        self.update_bounds(xmin,xmax,ymin,ymax)

    def rescale(self):
        """
        Rescale this curve's data according to
        the connected Scale-instance
        """
        _x, _y = self.scale.scale(self.x, self.y)
        xy = list(chain.from_iterable(zip(_x, _y)))
        self.vertex_list.vertices = xy

    def update_bounds(self,xmin,xmax,ymin,ymax):
        """
        Compute the bounds of this curves data and
        update the scale accordingly
        """
        self.xmin = min(self.xmin,xmin)
        self.ymin = min(self.ymin,ymin)
        self.xmax = max(self.xmax,xmax)
        self.ymax = max(self.ymax,ymax)

        self.scale.check_bounds(self.xmin,self.xmax,self.ymin,self.ymax)

def get_network_batch(stylized_network,
                      yoffset,
                      draw_links=True,
                      draw_nodes=True,
                      draw_nodes_as_rectangles=False,
                      n_circle_segments=16):
    """
    Create a batch for a network visualization.

    Parameters
    ----------
    stylized_network : dict
        The network properties which are returned from the
        interactive visualization.
    draw_links : bool, default : True
        Whether the links should be drawn
    draw_nodes : bool, default : True
        Whether the nodes should be drawn
    n_circle_segments : bool, default = 16
        Number of segments a circle will be constructed of.

    Returns
    -------
    network_objects : dict
        A dictionary containing all the necessary objects to draw and
        update the network.
        - `lines` : a list of pyglet-line objects (one entry per link)

        - `disks` : a list of pyglet-circle objects (one entry per node)
        - `circles` : a list of pyglet-circle objects (one entry per node)
        - `nodes_to_lines` : a dictionary mapping a node to a list of
          pairs. Each pair's first entry is the focal node's neighbor
          and the second entry is the index of the line-object that
          connects the two
        - `batch` : the pyglet Batch instance that contains all of the objects
    """

    batch = pyglet.graphics.Batch()

    pos = { node['id']: np.array([node['x_canvas'], node['y_canvas'] + yoffset]) for node in stylized_network['nodes'] }

    lines = []
    disks = []
    circles = []
    node_to_lines = { node['id']: [] for node in stylized_network['nodes'] }

    if draw_links:

        for ilink, link in enumerate(stylized_network['links']):
            u, v = link['source'], link['target']
            node_to_lines[u].append((v, ilink))
            node_to_lines[v].append((u, ilink))
            if 'color' in link.keys():
                this_color = link['color']
            else:
                this_color = stylized_network['linkColor']
            lines.append(shapes.Line(
                pos[u][0],
                pos[u][1],
                pos[v][0],
                pos[v][1],
                width=link['width'],
                color=tuple(bytes.fromhex(this_color[1:])),
                batch=batch,
                         )
                    )
            lines[-1].opacity = int(255*stylized_network['linkAlpha'])


    if draw_nodes:
        disks = [None for n in range(len(stylized_network['nodes']))]
        circles = [None for n in range(len(stylized_network['nodes']))]


        for node in stylized_network['nodes']:
            if not draw_nodes_as_rectangles:
                disks[node['id']] = \
                        shapes.Circle(node['x_canvas'],
                                      node['y_canvas']+yoffset,
                                      node['radius'],
                                      segments=n_circle_segments,
                                      color=tuple(bytes.fromhex(node['color'][1:])),
                                      batch=batch,
                                      )

                circles[node['id']] = \
                        shapes.Arc(node['x_canvas'],
                                      node['y_canvas']+yoffset,
                                      node['radius'],
                                      segments=n_circle_segments+1,
                                      color=tuple(bytes.fromhex(stylized_network['nodeStrokeColor'][1:])),
                                      batch=batch,
                                      )
            else:
                r = node['radius']
                disks[node['id']] = \
                        shapes.Rectangle(
                                      node['x_canvas']-r,
                                      node['y_canvas']+yoffset-r,
                                      2*r,
                                      2*r,
                                      color=tuple(bytes.fromhex(node['color'][1:])),
                                      batch=batch)

    return {'lines': lines, 'disks': disks, 'circles':circles, 'node_to_lines': node_to_lines, 'batch': batch}

_default_config = {
            'plot_sampled_curve': True,
            'draw_links':True,
            'draw_nodes':True,
            'n_circle_segments':16,
            'plot_height':120,
            'bgcolor':'#253237',
            'curve_stroke_width':4.0,
            'node_stroke_width':1.0,
            'link_color': '#4b5a62',
            'node_stroke_color':'#000000',
            'node_color':'#264653',
            'bound_increase_factor':1.0,
            'update_dt':0.04,
            'show_curves':True,
            'draw_nodes_as_rectangles':False,
            'show_legend': True,
            'legend_font_color':None,
            'legend_font_size':10,
            'padding':10,
            'compartment_colors':_colors,
            'palette': "dark",
        }

# light colors
#_default_config.update({
#            'bgcolor':'#fbfbef',
#            'link_color': '#8e9aaf',
#            'node_stroke_color':'#000000',
#            'legend_font_color':'#040414',
#        })


def visualize(model,
              network, 
              sampling_dt,
              ignore_plot_compartments=[],
              quarantine_compartments=[],
              config=None,
              ):
    """
    Start a visualization of a stochastic simulation.

    Parameters
    ==========
    model : epipack.stochastic_epi_models.StochasticEpiModel
        An initialized StochasticEpiModel.
    network: dict
        A stylized network in the netwulf-format
        (see https://netwulf.readthedocs.io/en/latest/python_api/post_back.html)
        where instead of 'x' and 'y', node positions are saved in 'x_canvas'
        and 'y_canvas'. Example:

        .. code:: python

            stylized_network = {
                "xlim": [0, 833],
                "ylim": [0, 833],
                "linkAlpha": 0.5,
                "nodeStrokeWidth": 0.75,
                "links": [
                    {"source": 0, "target": 1, "width": 3.0 }
                ],
                "nodes": [
                    {"id": 0,
                     "x_canvas": 436.0933431058901,
                     "y_canvas": 431.72418500564186,
                     "radius": 20
                     },
                    {"id": 1,
                     "x_canvas": 404.62184898400426,
                     "y_canvas": 394.8158724310507,
                     "radius": 20
                     }
                ]
            }

    sampling_dt : float 
        The amount of simulation time that's supposed to pass
        with a single update.
    ignore_plot_compartments : list, default = []
        List of compartment objects that are supposed to be
        ignored when plotted.
    quarantine_compartments : list, default = []
        List of compartment objects that are supposed to be
        resemble quarantine (i.e. temporarily 
        losing all connections)
    config : dict, default = None
        A dictionary containing all possible configuration
        options. Entries in this dictionary will overwrite
        the default config which is 

        .. code:: python

            _default_config = {
                        'plot_sampled_curve': True,
                        'draw_links':True,
                        'draw_nodes':True,
                        'n_circle_segments':16,
                        'plot_height':120,
                        'bgcolor':'#253237',
                        'curve_stroke_width':4.0,
                        'node_stroke_width':1.0,
                        'link_color': '#4b5a62',
                        'node_stroke_color':'#000000',
                        'node_color':'#264653',
                        'bound_increase_factor':1.0,
                        'update_dt':0.04,
                        'show_curves':True,
                        'draw_nodes_as_rectangles':False,
                        'show_legend': True,
                        'legend_font_color':'#fafaef',
                        'legend_font_size':10,
                        'padding':10,
                        'compartment_colors':_colors
                    }

    """

    # update the config and compute some helper variables
    cfg = deepcopy(_default_config)
    if config is not None:
        cfg.update(config)

    palette = cfg['palette']
    if type(palette) == str:
        if 'link_color' not in cfg:
            cfg['link_color'] = col.hex_link_colors[palette]
        if 'bgcolor' not in cfg:
            cfg['bgcolor'] = col.hex_bg_colors[palette]
        if 'compartment_colors' not in cfg:
            cfg['compartment_colors'] = [ col.colors[this_color] for this_color in col.palettes[palette] ]

    bgcolor = [ _/255 for _ in list(bytes.fromhex(cfg['bgcolor'][1:])) ] + [1.0]

    bgY = 0.2126*bgcolor[0] + 0.7152*bgcolor[1] + 0.0722*bgcolor[2]
    if cfg['legend_font_color'] is None:
        if bgY < 0.5:
            cfg['legend_font_color'] = '#fafaef'
        else:
            cfg['legend_font_color'] = '#232323'

    width = network['xlim'][1] - network['xlim'][0]
    height = network['ylim'][1] - network['ylim'][0]

    with_plot = cfg['show_curves'] and set(ignore_plot_compartments) != set(model.compartments)

    if with_plot:
        height += cfg['plot_height']
        plot_width = width
        plot_height = cfg['plot_height']
    else:
        plot_height = 0

    with_legend = cfg['show_legend']

    if with_legend:
        legend_batch = pyglet.graphics.Batch()
        #x, y = legend.get_location()
        #legend.set_location(x - width, y)
        # create a test label to get the actual dimensions
        test_label = pyglet.text.Label('Ag')
        dy = test_label.content_height * 1.1
        del(test_label)

        legend_circle_radius = dy/2/2
        distance_between_circle_and_label = 2*legend_circle_radius
        legend_height = len(model.compartments) * dy + cfg['padding']

        # if legend is shown in concurrence to the plot,
        # move the legend to be on the right hand side of the plot,
        # accordingly make the plot at least as tall as 
        # the demanded height or the legend height
        if with_plot:
            plot_height = max(plot_height, legend_height)
        legend_y_offset = legend_height

        max_text_width = 0
        legend_objects = [] # this is a hack so that the garbage collector doesn't delete our stuff 
        for iC, C in enumerate(model.compartments):
            this_y = legend_y_offset - iC * dy - cfg['padding']
            this_x = width + cfg['padding'] + legend_circle_radius
            label = pyglet.text.Label(str(C),
                              font_name=('Helvetica', 'Arial', 'Sans'),
                              font_size=cfg['legend_font_size'],
                              x=this_x + legend_circle_radius+distance_between_circle_and_label, 
                              y=this_y,
                              anchor_x='left', anchor_y='top',
                              color = list(bytes.fromhex(cfg['legend_font_color'][1:])) + [255],
                              batch = legend_batch
                              )
            legend_objects.append(label)

            #if not cfg['draw_nodes_as_rectangles']:
            if True:
                disk = shapes.Circle(this_x,
                                      this_y - (dy-1.25*legend_circle_radius)/2,
                                      legend_circle_radius,
                                      segments=64,
                                      color = cfg['compartment_colors'][iC],
                                      batch=legend_batch,
                                      )

                circle = shapes.Arc(this_x,
                                      this_y - (dy-1.25*legend_circle_radius)/2,
                                      legend_circle_radius,
                                      segments=64+1,
                                      color=list(bytes.fromhex(cfg['legend_font_color'][1:])),
                                      batch=legend_batch,
                                      )

                legend_objects.extend([disk,circle])
            #else:
            #    rect = shapes.Rectangle(this_x,
            #                          this_y - (dy-1.5*legend_circle_radius)/2,
            #                          2*legend_circle_radius,
            #                          2*legend_circle_radius,
            #                          color = _colors[iC],
            #                          batch=legend_batch,
            #                          )
            #    legend_objects.append(rect)

            max_text_width = max(max_text_width, label.content_width)

        legend_width =   2*cfg['padding'] \
                       + 2*legend_circle_radius \
                       + distance_between_circle_and_label \
                       + max_text_width

        # if legend is shown in concurrence to the plot,
        # move the legend to be on the right hand side of the plot,
        # accordingly make the plot narrower and place the legend
        # directly under the square network plot.
        # if not, make the window wider and show the legend on
        # the right hand side of the network plot.
        if with_plot:
            for obj in legend_objects:
                obj.x -= legend_width 
            plot_width = width - legend_width
        else:
            width += legend_width

    size = (width, height)


    # overwrite network style with the epipack default style
    network['linkColor'] = cfg['link_color']
    network['nodeStrokeColor'] = cfg['node_stroke_color']
    for node in network['nodes']:
        node['color'] = cfg['node_color']
    N = len(network['nodes'])

    # get the OpenGL shape objects that comprise the network
    network_batch = get_network_batch(network,
                                      yoffset=plot_height,
                                      draw_links=cfg['draw_links'],
                                      draw_nodes=cfg['draw_nodes'],
                                      draw_nodes_as_rectangles=cfg['draw_nodes_as_rectangles'],
                                      n_circle_segments=cfg['n_circle_segments'],
                                      )
    lines = network_batch['lines']
    disks = network_batch['disks']
    circles = network_batch['circles']
    node_to_lines = network_batch['node_to_lines']
    batch = network_batch['batch']

    # initialize a simulation state that has to passed to the app
    # so the app can change simulation parameters
    simstate = SimulationStatus(len(network['nodes']), sampling_dt)

    # intialize app
    window = App(*size,simulation_status=simstate,resizable=True)
    glClearColor(*bgcolor)

    # handle different strokewidths
    if 'nodeStrokeWidth' in network:
        node_stroke_width = network['nodeStrokeWidth'] 
    else:
        node_stroke_width = cfg['node_stroke_width']

    def _set_linewidth_nodes():
        glLineWidth(node_stroke_width)

    def _set_linewidth_curves():
        glLineWidth(cfg['curve_stroke_width'])

    def _set_linewidth_legend():
        glLineWidth(1.0)

    # add the network batch with the right function to set the linewidth
    # prior to drawing
    window.add_batch(batch,prefunc=_set_linewidth_nodes)

    if with_legend:
        # add the legend batch with the right function to set the linewidth
        # prior to drawing
        window.add_batch(legend_batch,prefunc=_set_linewidth_legend)

    # decide whether to plot all measured changes or only discrete-time samples
    discrete_plot = cfg['plot_sampled_curve']

    # find quarantined compartment ids
    # This set is needed for filtering later on.
    quarantined = set(model.get_compartment_id(C) for C in quarantine_compartments)

    # initialize time arrays
    t = 0
    discrete_time = [t]

    # initialize curves
    if with_plot:
        # find the maximal value of the
        # compartments that are meant to be plotted. 
        # These sets are needed for filtering later on.
        maxy = max([ model.y0[model.get_compartment_id(C) ] for C in (set(model.compartments) - set(ignore_plot_compartments))])
        scl = Scale(bound_increase_factor=cfg['bound_increase_factor'])\
                .extent(0,plot_width,plot_height-cfg['padding'],cfg['padding'])\
                .domain(0,20*sampling_dt,0,maxy)
        curves = {}
        for iC, C in enumerate(model.compartments):
            if C in ignore_plot_compartments:
                continue
            _batch = pyglet.graphics.Batch()
            window.add_batch(_batch,prefunc=_set_linewidth_curves)
            y = [np.count_nonzero(model.node_status==model.get_compartment_id(C))]
            curve = Curve(discrete_time,y,cfg['compartment_colors'][iC],scl,_batch)
            curves[C] = curve

    # define the pyglet-App update function that's called on every clock cycle
    def update(dt):

        # skip if nothing remains to be done
        if simstate.simulation_ended or simstate.paused:
            return

        # get sampling_dt
        sampling_dt = simstate.sampling_dt

        # Advance the simulation until time sampling_dt.
        # sim_time is a numpy array including all time values at which
        # the system state changed. The first entry is the initial state
        # of the simulation at t = 0 which we will discard later on
        # the last entry at `sampling_dt` will be missing so we
        # have to add it later on.
        # `sim_result` is a dictionary that maps a compartment
        # to a numpy array containing the compartment counts at
        # the corresponding time.
        sim_time, sim_result = model.simulate(sampling_dt,adopt_final_state=True)

        # compare the new node statuses with the old node statuses
        # and find the nodes that have changed status
        ndx = np.where(model.node_status != simstate.old_node_status)[0]

        # if nothing changed, evaluate the true total event rate
        # and if it's zero, do not do anything anymore
        did_simulation_end = len(ndx) == 0 and model.get_true_total_event_rate() == 0.0
        simstate.set_simulation_status(did_simulation_end)
        if simstate.simulation_ended:
            return

        # advance the current time as described above.
        # we save both all time values as well as just the sampled times.
        this_time = (discrete_time[-1] + sim_time[1:]).tolist() + [discrete_time[-1] + sampling_dt]
        discrete_time.append(discrete_time[-1] + sampling_dt)

        # if curves are plotted
        if with_plot:

            # iterate through result array
            for k, v in sim_result.items():
                # skip curves that should be ignored
                if k in ignore_plot_compartments:
                    continue
                # count occurrences of this compartment
                val = np.count_nonzero(model.node_status==model.get_compartment_id(k))
                if discrete_plot:
                    # in case only sampled curves are of interest,
                    # just add this single value
                    curves[k].append_single_value(discrete_time[-1], v[-1])
                else:
                    # otherwise, append the current value to the exact simulation list
                    # and append the whole dataset
                    val = (v[1:].tolist() + [v[-1]])
                    curves[k].append_list(this_time, val)


        # iterate through the nodes that have to be updated
        for node in ndx:
            status = model.node_status[node]
            if cfg['draw_nodes']:
                disks[node].color = cfg['compartment_colors'][status]

            # if a node becomes quarantined,
            # iterate through its attached links (lines)
            # and switch them off
            if status in quarantined:
                for neigh, linkid in node_to_lines[node]:
                    lines[linkid].visible = False
            # if it became unquarantined
            elif simstate.old_node_status[node] in quarantined:
                # check of the neighbor is unquarantined
                # and switch on the link if this is the case
                for neigh, linkid in node_to_lines[node]:
                    if model.node_status[neigh] not in quarantined:
                        lines[linkid].visible = True

        # save the current node statuses
        simstate.update(model.node_status)

    # schedule the app clock and run the app
    pyglet.clock.schedule_interval(update, cfg['update_dt'])
    pyglet.app.run()
    #pyglet.clock.unschedule(update)
    #window.close()
    #del(window)
    #print(pyglet.app.platform_event_loop)
    #pyglet.app.exit()
    #print("called exit..")


def visualize_reaction_diffusion(
              model,
              network,
              sampling_dt,
              node_compartments,
              value_extent=[0.0,1.0],
              integrator='euler',
              n_integrator_midpoints=0,
              config=None,
              ):
    """
    Start a visualization of a reaction-diffusion simulation.

    Parameters
    ==========
    model : epipack.deterministic_epi_models.DeterministicEpiModel
        An initialized DeterministicEpiModel.
    network: dict
        A stylized network in the netwulf-format
        (see https://netwulf.readthedocs.io/en/latest/python_api/post_back.html)
        where instead of 'x' and 'y', node positions are saved in 'x_canvas'
        and 'y_canvas'. Example:

        .. code:: python

            stylized_network = {
                "xlim": [0, 833],
                "ylim": [0, 833],
                "linkAlpha": 0.5,
                "nodeStrokeWidth": 0.75,
                "links": [
                    {"source": 0, "target": 1, "width": 3.0 }
                ],
                "nodes": [
                    {"id": 0,
                     "x_canvas": 436.0933431058901,
                     "y_canvas": 431.72418500564186,
                     "radius": 20
                     },
                    {"id": 1,
                     "x_canvas": 404.62184898400426,
                     "y_canvas": 394.8158724310507,
                     "radius": 20
                     }
                ]
            }

    sampling_dt : float
        The amount of simulation time that's supposed to pass
        with a single update.
    quarantine_compartments: list
        List of compartment objects that are supposed to be
        resemble quarantine (i.e. temporarily
        losing all connections)
    node_compartments: list
        The compartments for which to display the concentrations.
        Each entry `m` in this list is expected to be a compartment
        associated with node `m`. this list should therefore
        be as long as the number of nodes.
    config : dict, default = None
        A dictionary containing all possible configuration
        options. Entries in this dictionary will overwrite
        the default config which is

        .. code:: python

            _default_config = {
                        'plot_sampled_curve': True,
                        'draw_links':True,
                        'draw_nodes':True,
                        'n_circle_segments':16,
                        'plot_height':120,
                        'bgcolor':'#253237',
                        'curve_stroke_width':4.0,
                        'node_stroke_width':1.0,
                        'link_color': '#4b5a62',
                        'node_stroke_color':'#000000',
                        'node_color':'#264653',
                        'bound_increase_factor':1.0,
                        'update_dt':0.04,
                        'show_curves':True,
                        'draw_nodes_as_rectangles':False,
                        'show_legend': True,
                        'legend_font_color':'#fafaef',
                        'legend_font_size':10,
                        'padding':10,
                        'compartment_colors':_colors
                    }

    """

    # update the config and compute some helper variables
    cfg = deepcopy(_default_config)
    if config is not None:
        cfg.update(config)

    bgcolor = [ _/255 for _ in list(bytes.fromhex(cfg['bgcolor'][1:])) ] + [1.0]

    width = network['xlim'][1] - network['xlim'][0]
    height = network['ylim'][1] - network['ylim'][0]

    size = (width, height)



    # overwrite network style with the epipack default style
    network['linkColor'] = cfg['link_color']
    network['nodeStrokeColor'] = cfg['node_stroke_color']
    for node in network['nodes']:
        node['color'] = col.hex_bg_colors['light']
    N = len(network['nodes'])

    # get the OpenGL shape objects that comprise the network
    network_batch = get_network_batch(network,
                                      yoffset=0,
                                      draw_links=cfg['draw_links'],
                                      draw_nodes=cfg['draw_nodes'],
                                      draw_nodes_as_rectangles=cfg['draw_nodes_as_rectangles'],
                                      n_circle_segments=cfg['n_circle_segments'],
                                      )
    lines = network_batch['lines']
    disks = network_batch['disks']
    circles = network_batch['circles']
    node_to_lines = network_batch['node_to_lines']
    batch = network_batch['batch']

    # initialize a simulation state that has to passed to the app
    # so the app can change simulation parameters
    simstate = SimulationStatus(len(network['nodes']), sampling_dt)

    # intialize app
    window = App(*size,simulation_status=simstate,resizable=True)
    glClearColor(*bgcolor)

    # handle different strokewidths
    if 'nodeStrokeWidth' in network:
        node_stroke_width = network['nodeStrokeWidth']
    else:
        node_stroke_width = cfg['node_stroke_width']

    def _set_linewidth_nodes():
        glLineWidth(node_stroke_width)

    # add the network batch with the right function to set the linewidth
    # prior to drawing
    window.add_batch(batch,prefunc=_set_linewidth_nodes)

    # initialize time arrays
    t = 0
    discrete_time = [t]

    _cmin, _cmax = value_extent
    if not callable(_cmin):
        cmin = lambda _: _cmin
    else:
        cmin = _cmin
    if not callable(_cmax):
        cmax = lambda _: _cmax
    else:
        cmax = _cmax
    def _get_opacity(val,this_cmin,this_cmax):
        opacity = (val-this_cmin)/(this_cmax-this_cmin) + this_cmin
        if opacity > 1.0:
            opacity = 1.0
        if opacity < 0.0:
            opacity = 0.0
        return int(255*opacity)

    try:
        node_compartment_indices = np.array([ model.get_compartment_id(C) for C in node_compartments ])
    except AttributeError as e:
        node_compartment_indices = np.array(node_compartments)

    this_cmin, this_cmax = cmin(model.y0), cmax(model.y0)
    for node, idx in enumerate(node_compartment_indices):
        concentration = model.y0[idx]
        disks[node].opacity = _get_opacity(concentration, this_cmin, this_cmax)

    # define the pyglet-App update function that's called on every clock cycle
    def update(dt):

        # skip if nothing remains to be done
        if simstate.simulation_ended or simstate.paused:
            return

        # get sampling_dt
        sampling_dt = simstate.sampling_dt

        # Advance the simulation until time sampling_dt.
        # `sim_result` is a two-dimensional array
        # where index sim_result[iC, iT] gives the
        # concentration of compartment iC at time iT
        this_t = np.linspace(0, sampling_dt, n_integrator_midpoints+2)
        sim_result = model.integrate_and_return_by_index(
                            this_t,
                            integrator=integrator,
                            adopt_final_state=True,
                            )

        # it might happen that an external model only returns
        # the final state instead of for each time point
        if len(sim_result.shape) > 1:
            result = sim_result[node_compartment_indices,-1]
        else:
            result = sim_result[node_compartment_indices]

        # if nothing changed, evaluate the true total event rate
        # and if it's zero, do not do anything anymore
        #did_simulation_end = len(ndx) == 0 and model.get_true_total_event_rate() == 0.0
        #simstate.set_simulation_status(did_simulation_end)
        #if simstate.simulation_ended:
        #    return
        this_cmin, this_cmax = cmin(result), cmax(result)

        # detect significant changes (relative change)
        rel_change = np.abs(result-simstate.old_node_status)
        ndx = np.where(rel_change>1e-3)[0]
        #ndx = np.where(rel_change>0)[0]
        result = result[ndx]

        # iterate through the nodes that have to be updated
        #for node, idx in enumerate(node_compartment_indices):
        if cfg['draw_nodes']:
            for node, concentration in zip(ndx, result):
                    disks[node].opacity = _get_opacity(concentration,this_cmin,this_cmax)

        simstate.old_node_status[ndx] = result
        #simstate.update(result)


    # schedule the app clock and run the app
    pyglet.clock.schedule_interval(update, cfg['update_dt'])
    pyglet.app.run()


if __name__=="__main__":     # pragma: no cover

    import netwulf as nw
    from epipack import StochasticEpiModel

    network, config, _ = nw.load('/Users/bfmaier/pythonlib/facebook/FB.json')

    N = len(network['nodes'])

    edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]
    k0 = 2*len(edge_list)/len(network['nodes'])

    model = StochasticEpiModel(list("SIRXTQ"),
                               N=len(network['nodes']),
                               edge_weight_tuples=edge_list,
                               )
    Reff = 3
    R0 = 10
    recovery_rate = 1/8
    quarantine_rate = 1/16
    tracing_rate = 1/2
    waning_immunity_rate = 1/14
    infection_rate = Reff * (recovery_rate+quarantine_rate) / k0
    infection_rate = R0 * (recovery_rate) / k0
    model.set_node_transition_processes([
            ("I",recovery_rate,"R"),
            ("I",quarantine_rate,"T"),
            ("T",tracing_rate,"X"),
            ("Q",waning_immunity_rate,"S"),
            ("X",recovery_rate,"R"),
            ])
    model.set_link_transmission_processes([("I","S",infection_rate,"I","I")])
    model.set_conditional_link_transmission_processes({
        ("T", "->", "X") : [
                 ("X","I",0.5,"X","T"),
                 #("X","S",0.5,"X","Q"),
                 ],
        })
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 0.08

    visualize(model,network,sampling_dt,ignore_plot_compartments=['S'],quarantine_compartments=['X', 'T', 'Q'])

