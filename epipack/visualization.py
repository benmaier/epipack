"""
Classes to visualize a stochastic simulation with pyglet.
"""

try:
    import pyglet
except ModuleNotFoundError as e:
    print("Please install pyglet>=1.5. It's not installed automatically.")
    raise e

from copy import deepcopy
from itertools import chain

import numpy as np

from pyglet import shapes
from pyglet.window import key, mouse, Window
from pyglet.gl import *

_colors = [ [38,70,83], [231,111,81], [42,157,143],[131,227,119], [233,196,106], [100,100,100] ] * 2

class App(pyglet.window.Window):

    def __init__(self, width, height, *args, **kwargs):
        #conf = Config(sample_buffers=1,
        #              samples=4,
        #              depth_size=16,
        #              double_buffer=True)
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
        # Initialize OpenGL context
        #self.init_gl(width, height)

    def add_batch(self,batch,prefunc=None):
        self.batches.append(batch)
        self.batch_funcs.append(prefunc)


    def on_draw(self):
        self.clear()

        for batch, func in zip(self.batches, self.batch_funcs):
            if func is not None:
                func()
            batch.draw()

        #glPopMatrix()

    def init_gl(self, width, height):
        # Set viewport
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

    def on_resize(self, width, height):
        # Set window values
        self.width  = width
        self.height = height
        # Initialize OpenGL context
        self.init_gl(width, height)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

    def on_mouse_scroll(self, x, y, dx, dy):
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

class Scale():

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
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self._calc()
        return self

    def domain(self,x0,x1,y0,y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self._calc()
        return self

    def _calc(self):
        self.mx = (self.right-self.left)/(self.x1 - self.x0)
        self.my = (self.top-self.bottom)/(self.y1 - self.y0)

    def scale(self,x,y):
        _x = self.scalex(x)
        _y = self.scaley(y)

        return _x, _y

    def scalex(self,x):
        if type(x) == list:
            _x = list(map(lambda _x: self.mx * (_x-self.x0) + self.left, x ))
        else:
            _x = self.mx * (x-self.x0) + self.left

        return _x
        
    def scaley(self,y):
        if type(y) == list:
            _y = list(map(lambda _y: self.my * (_y-self.y0) + self.bottom, y ))
        else:
            _y = self.my * (y-self.y0) + self.bottom

        return _y

    def check_bounds(self,xmin,xmax,ymin,ymax):
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
        self.scaling_objects.append(obj)
        


class Curve():

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
        self.append_list([x], [y])

    def append_list(self,x,y):
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
        _x, _y = self.scale.scale(self.x, self.y)
        xy = list(chain.from_iterable(zip(_x, _y)))
        self.vertex_list.vertices = xy

    def update_bounds(self,xmin,xmax,ymin,ymax):
        self.xmin = min(self.xmin,xmin)
        self.ymin = min(self.ymin,ymin)
        self.xmax = max(self.xmax,xmax)
        self.ymax = max(self.ymax,ymax)

        self.scale.check_bounds(self.xmin,self.xmax,self.ymin,self.ymax)

class StatusSaver():

    def __init__(self,N):
        self.old_node_status = -1e300*np.ones((N,))

    def update(self,old_node_status):
        self.old_node_status = np.array(old_node_status)

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
    """

    batch = pyglet.graphics.Batch()

    pos = { node['id']: np.array([node['x_canvas'], node['y_canvas'] + yoffset]) for node in stylized_network['nodes'] }

    lines = []
    disks = []
    circles = []
    node_to_lines = { node['id']: [] for node in stylized_network['nodes'] }

    if draw_links:
        #zorder = max( _c.get_zorder() for _c in ax.get_children()) + 1

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
                                      color=tuple(bytes.fromhex(node['color'][1:])),# + [int(255*stylized_network['linkAlpha'])]),
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
                                      color=tuple(bytes.fromhex(node['color'][1:])),# + [int(255*stylized_network['linkAlpha'])]),
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
            'draw_nodes_as_rectangles':False,
        }

def get_grid_layout(nodes,edge_weight_tuples=[],windowwidth=400,linkwidth=1):

    w = h = windowwidth
    N = len(nodes)
    N_side = int(np.ceil(np.sqrt(N)))
    dx = w / N_side
    radius = dx/2

    network = {}
    stylized_network = {
        'xlim': [0, w],
        'ylim': [0, h],
        'linkAlpha': 0.5,
        'nodeStrokeWidth': 0.0001,
    }

    nodes = [ {
                'id': i*N_side + j,
                'x_canvas': i*dx + 0.5,
                'y_canvas': j*dx + 0.5,
                'radius': radius
              } for i in range(N_side) for j in range(N_side) ]
    links = [ {'source': u, 'target': v, 'width': linkwidth} for u, v, w in edge_weight_tuples ]

    stylized_network['nodes'] = nodes
    stylized_network['links'] = links

    return stylized_network

def visualize(model,
              network, 
              sampling_dt,
              ignore_plot_compartments=[],
              quarantine_compartments=[],
              config=None,
              ):

    cfg = deepcopy(_default_config)
    if config is not None:
        cfg.update(config)
        
    width = network['xlim'][1] - network['xlim'][0]
    height = network['ylim'][1] - network['ylim'][0]

    with_plot = set(ignore_plot_compartments) != set(model.compartments)

    if with_plot:
        size = (width, height+cfg['plot_height'])
    else:
        size = (width, height)

    network['linkColor'] = cfg['link_color']
    network['nodeStrokeColor'] = cfg['node_stroke_color']
    for node in network['nodes']:
        node['color'] = cfg['node_color']
    N = len(network['nodes'])

    network_batch = get_network_batch(network,
                                      cfg['plot_height'],
                                      cfg['draw_links'],
                                      cfg['draw_nodes'],
                                      cfg['draw_nodes_as_rectangles'],
                                      cfg['n_circle_segments'],
                                      )
    lines = network_batch['lines']
    disks = network_batch['disks']
    circles = network_batch['circles']
    node_to_lines = network_batch['node_to_lines']
    batch = network_batch['batch']

    window = App(*size,resizable=True)
    bgcolor = [ _/255 for _ in list(bytes.fromhex(cfg['bgcolor'][1:])) ] + [1.0]
    glClearColor(*bgcolor)

    if 'nodeStrokeWidth' in network:
        node_stroke_width = network['nodeStrokeWidth'] 
    else:
        node_stroke_width = cfg['node_stroke_width']

    def _set_linewidth_nodes():
        glLineWidth(node_stroke_width)

    def _set_linewidth_curves():
        glLineWidth(cfg['curve_stroke_width'])

    window.add_batch(batch,prefunc=_set_linewidth_nodes)

    discrete_plot = cfg['plot_sampled_curve']
    quarantined = set(model.get_compartment_id(C) for C in quarantine_compartments)
    #not_quarantined = set(model.compartments) - quarantined

    maxy = max([ model.y0[model.get_compartment_id(C) ]for C in (set(model.compartments) - set(ignore_plot_compartments))])

    saver = StatusSaver(len(network['nodes']))
    t = 0
    discrete_time = [t]

    if with_plot:
        scl = Scale(bound_increase_factor=cfg['bound_increase_factor'])\
                .extent(0,window.get_size()[0],cfg['plot_height']-10,10)\
                .domain(0,20*sampling_dt,0,maxy)
        curves = {}
        for iC, C in enumerate(model.compartments):
            if C in ignore_plot_compartments:
                continue
            _batch = pyglet.graphics.Batch()
            window.add_batch(_batch,prefunc=_set_linewidth_curves)
            y = [np.count_nonzero(model.node_status==model.get_compartment_id(C))]
            curve = Curve(discrete_time,y,_colors[iC],scl,_batch)
            curves[C] = curve

    def update(dt):
        
        sim_time, sim_result = model.simulate(sampling_dt)

        ndx = np.where(model.node_status != saver.old_node_status)[0]

        if len(ndx) == 0 and model.get_true_total_event_rate() == 0.0:
            return

        this_time = (discrete_time[-1] + sim_time).tolist() + [discrete_time[-1] + sampling_dt]
        discrete_time.append(discrete_time[-1] + sampling_dt)

        if with_plot:
            for k, v in sim_result.items():
                if k in ignore_plot_compartments:
                    continue
                val = np.count_nonzero(model.node_status==model.get_compartment_id(k))
                if discrete_plot:
                    curves[k].append_single_value(discrete_time[-1], v[-1])
                else:
                    val = (v.tolist() + [v[-1]])
                    curves[k].append_list(this_time, val)
                


        for node in ndx:
            status = model.node_status[node]
            disks[node].color = _colors[status]
            if status in quarantined:
                for neigh, linkid in node_to_lines[node]:
                    lines[linkid].visible = False
            elif saver.old_node_status[node] in quarantined:
                for neigh, linkid in node_to_lines[node]:
                    if model.node_status[neigh] not in quarantined:
                        lines[linkid].visible = True

        saver.update(model.node_status)

    pyglet.clock.schedule_interval(update, cfg['update_dt'])
    pyglet.app.run()


if __name__=="__main__":

    import netwulf as nw
    from epipack import StochasticEpiModel

    network, config, _ = nw.load('/Users/bfmaier/pythonlib/facebook/FB.json')

    N = len(network['nodes'])

    edge_list = [ ( link['source'], link['target'], 1.0 ) for link in network['links'] ]
    k0 = 2*len(edge_list)/len(network['nodes'])

    
    model = StochasticEpiModel(list("SIRXTQ"),N=len(network['nodes']),
                               edge_weight_tuples=edge_list
                               )
    Reff = 3
    R0 = 3
    recovery_rate = 1/8
    quarantine_rate = 1/32
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
                 ("X","I",0.4,"X","T"),
                 ("X","S",0.4,"X","Q"),
                 ],
        })
    model.set_random_initial_conditions({'I':20,'S':N-20})

    sampling_dt = 0.08

    visualize(model,network,sampling_dt,ignore_plot_compartments=['S'],quarantine_compartments=['X', 'T', 'Q'])

