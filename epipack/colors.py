"""
Functions and objects dealing with color.
"""

import numpy as np

def h2r(_hex):
    """
    Convert a hex string to an RGB-tuple.
    """
    if _hex.startswith('#'):
        l = _hex[1:]
    else:
        l = _hex
    return list(bytes.fromhex(l))

def r2h(rgb):
    """
    Convert an RGB-tuple to a hex string.
    """
    return '#%02x%02x%02x' % tuple(rgb)

def tohex(color):
    """
    Convert any color to its hex string.
    """

    if type(color) == str:
        if len(color) in (6, 7):
            try:
                h2r(color)
                return color
            except:
                pass
        try:
            return hex_colors[color]
        except KeyError as e:
            raise ValueError("unknown color: '" + color +"'")
    elif type(color) in (list, tuple, np.ndarray) and len(color) == 3:
        return r2h(color)
    else:
        raise ValueError("Don't know how to interpret color " + str(color))

def torgb(color):
    """
    Convert any color to an rgb tuple.
    """

    if type(color) == str:
        if len(color) in (6, 7):
            try:
                return h2r(color)
            except:
                pass
        try:
            return colors[color]
        except KeyError as e:
            raise ValueError("unknown color: '" + str(color) +"'")
    elif type(color) in (list, tuple, np.ndarray) and len(color) == 3:
        return h2r(color)
    else:
        raise ValueError("Don't know how to interpret color " + str(color))


def brighter(rgb):
    """
    Make the color (rgb-tuple) a tad brighter.
    """
    _rgb = tuple([ int(np.sqrt(a/255) * 255) for a in rgb ])
    return _rgb

def darker(rgb):
    """
    Make the color (rgb-tuple) a tad darker.
    """
    _rgb = tuple([ int((a/255)**2 * 255) for a in rgb ])
    return _rgb

hex_colors = {
    # default
    'marine' : '#365663',
    'red' : '#ee6f51',
    'teal' : '#2a9d8f',
    'green' : '#83e377',
    'yellow' : '#e9c46a',
    'grey' : '#646464',
    'pink' : '#ff9de4',
    'blue' : '#52b7dc',
    'grape' : '#b75b9e',
    'light grape' : '#ff9de4',
    'light marine' : '#47829a',
    'light red' : '#ff9f92',
    'light teal' : '#45dbc8',
    'light green': '#a4ffbb',
    'light yellow': '#ffee9d',
    'light grey' : '#a1a1a1',
    'light blue' : '#b0dff0',
    'light pink' : '#ffcaf0',

    # french79
    'french79 marine' : '#555587',
    'french79 red' : '#e64b3f',
    'french79 blue' : '#94d9ce',
    'french79 yellow' : '#efc94b',
    'french79 green' : '#71b081',
    'french79 light marine' : '#47829a',
    'french79 light red' : '#ec9f80',
    'french79 light blue' : '#d6e4db',
    'french79 light yellow' : '#edd88d',
    'french79 light green' : '#a4ffbb',

    # brewer
    'brewer yellow': '#ffcf46',
    'brewer pink': '#ff62b2',
    'brewer teal': '#1fb78a',
    'brewer grape': '#8e88da',
    'brewer orange': '#ff9849',
    'brewer green': '#66a61e',
    'brewer grey': '#909090',
    'brewer brown': '#a6761d',
    'brewer marine': '#555587',
    'brewer light yellow': '#ffecb8',
    'brewer light pink': '#efadcf',
    'brewer light grape': '#c4c1ef',
    'brewer light teal': '#97ddc8',
    'brewer light orange': '#ffd2af',
    'brewer light green': '#ceeaaf',
    'brewer light grey': '#cecece',
    'brewer light brown': '#d3bb90',
    'brewer light marine' : '#47829a',
}

hex_bg_colors = {
    'dark' : '#253237',
    'light' : '#fff9e8',
    'french79': '#26263d',
    'brewer dark': '#2f2940',
    'brewer light': '#fff9e8',
    'dark pastel' : '#253237',
    'light pastel' : '#fff9e8',
    'french79 pastel': '#26263d',
    'brewer dark pastel': '#2f2940',
    'brewer light pastel': '#fff9e8',
}

hex_link_colors = {
    'dark': '#4b5a62', 
    'light': '#a1a1a1',
    'french79': '#4b5a62', 
    'brewer dark': '#4b5a62', 
    'brewer light': '#a1a1a1', 
    'dark pastel': '#4b5a62', 
    'light pastel': '#a1a1a1', 
    'french79 pastel': '#4b5a62', 
    'brewer dark pastel': '#4b5a62', 
    'brewer light pastel': '#a1a1a1', 
}

colors = { name: h2r(col) for name, col in hex_colors.items() }
bg_colors = { name: h2r(col) for name, col in hex_bg_colors.items() }
link_colors = { name: h2r(col) for name, col in hex_link_colors.items() }


accompanying_color = {}

# map accompanying color, i.e. if a color is light map to its "normal"
# color, if it is normal, map to its light variant
for color, _hex in colors.items():
    name = color.split(" ")
    colname = name[-1]
    if len(name) == 1:
        accompanying_color[color] = 'light '+colname
    elif len(name) == 2:
        if name[0] == 'light':
            accompanying_color[color] = colname
        else:
            accompanying_color[color] = name[0] + ' light ' + colname
    elif len(name) == 3:
        accompanying_color[color] = name[0] + ' ' + colname

#for a, b in accompanying_color.items():
#    print(a,"  ----  ",b)

dark = [
            'marine',
            'red',
            'teal',
            'green',
            'yellow',
            'grey',
            'grape',
            'light marine',
            'light red',
            'light teal',
            'light green',
            'light yellow',
            'light grey',
            'light grape',
    ]

dark_pastel = [
            'light marine',
            'light red',
            'light teal',
            'light green',
            'light yellow',
            'light grey',
            'light grape',
            'marine',
            'red',
            'teal',
            'green',
            'yellow',
            'grey',
            'grape',
        ]

light = [
            'light yellow',
            'red',
            'teal',
            'green',
            'blue',
            'grey',
            'pink',
            'yellow',
            'light red',
            'light teal',
            'light green',
            'light blue',
            'light grey',
            'light pink',
        ]

light_pastel = [
            'yellow',
            'light red',
            'light teal',
            'light green',
            'light blue',
            'light grey',
            'light pink',
            'light yellow',
            'red',
            'teal',
            'green',
            'blue',
            'grey',
            'pink',
        ]

french79 = [
            'french79 marine',
            'french79 red',
            'french79 blue',
            'french79 yellow',
            'french79 green',
            'grey',
            'grape',
            'french79 light marine',
            'french79 light red',
            'french79 light blue',
            'french79 light yellow',
            'french79 light green',
            'light grey',
            'light grape',
        ]


french79_pastel = [
            'french79 light marine',
            'french79 light red',
            'french79 light blue',
            'french79 light yellow',
            'french79 light green',
            'light grey',
            'light grape',
            'french79 marine',
            'french79 red',
            'french79 blue',
            'french79 yellow',
            'french79 green',
            'grey',
            'grape',
        ]

brewer_light = [
            'brewer light yellow',
            'brewer pink',
            'brewer teal',
            'brewer grape',
            'brewer orange',
            'brewer green',
            'brewer grey',
            'brewer brown',

            'brewer yellow',
            'brewer light pink',
            'brewer light teal',
            'brewer light grape',
            'brewer light orange',
            'brewer light green',
            'brewer light grey',
            'brewer light brown',
        ]

brewer_light_pastel = [
            'brewer yellow',
            'brewer light pink',
            'brewer light teal',
            'brewer light grape',
            'brewer light orange',
            'brewer light green',
            'brewer light grey',
            'brewer light brown',

            'brewer light yellow',
            'brewer pink',
            'brewer teal',
            'brewer grape',
            'brewer orange',
            'brewer green',
            'brewer grey',
            'brewer brown',
        ]

brewer_dark = [
            'brewer marine',
            'brewer pink',
            'brewer teal',
            'brewer yellow',
            'brewer green',
            'brewer orange',
            'brewer grey',
            'brewer brown',

            'brewer light marine',
            'brewer light pink',
            'brewer light teal',
            'brewer light yellow',
            'brewer light green',
            'brewer light orange',
            'brewer light grey',
            'brewer light brown',
        ]

brewer_dark_pastel = [
            'brewer light marine',
            'brewer light pink',
            'brewer light teal',
            'brewer light yellow',
            'brewer light green',
            'brewer light orange',
            'brewer light grey',
            'brewer light brown',

            'brewer marine',
            'brewer pink',
            'brewer teal',
            'brewer yellow',
            'brewer green',
            'brewer orange',
            'brewer grey',
            'brewer brown',
        ]

palettes = {
       'dark':                  dark,
       'dark pastel':           dark_pastel,
       'light':                 light,
       'light pastel':          light_pastel,
       'french79':              french79,
       'french79 pastel':       french79_pastel,
       'brewer light':          brewer_light,
       'brewer light pastel':   brewer_light_pastel,
       'brewer dark':           brewer_dark,
       'brewer dark pastel':    brewer_dark_pastel,
   }

if __name__ == "__main__": # pragma : no cover

    from matplotlib.collections import LineCollection, EllipseCollection
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['Helvetica','Arial']
    import matplotlib.pyplot as pl

    for palette, this_palette in palettes.items():
        #if palette.endswith(' pastel'):
        #    continue

        fig, ax = pl.subplots(1,1)
        N = len(palette)
        firstcol = this_palette[:len(this_palette)//2]
        secondcol = this_palette[len(this_palette)//2:]
        islight = 'light' in palette
        bgcolor = hex_bg_colors[palette]
        print(firstcol, secondcol)
        if islight:
            fontcolor = 'k'
        else:
            fontcolor = 'w'

        r = 0.125
        dy = 0.5
        dx = 0.2
        XY = []
        size = []
        node_colors = []
        
        for i in range(len(firstcol)):
            firstname = firstcol[i] + ' // ' + hex_colors[firstcol[i]][1:].upper()
            secondname = hex_colors[secondcol[i]][1:].upper() + ' // ' + secondcol[i]
            XY.append([-dx, -i*dy-3])
            XY.append([+dx, -i*dy-3])
            size.extend([2*r, 2*r])
            node_colors.append(hex_colors[firstcol[i]])
            node_colors.append(hex_colors[secondcol[i]])

            ax.text(-dx-2*r, -i*dy-3, firstname, transform=ax.transData, ha='right',va='center', color=fontcolor)
            ax.text(+dx+2*r, -i*dy-3, secondname, transform=ax.transData, ha='left',va='center', color=fontcolor)

        circles = EllipseCollection(
            size, size, np.zeros_like(size),
            offsets=np.array(XY),
            units='x',
            transOffset=ax.transData,
            facecolors=node_colors,
            linewidths=0.5,
            edgecolors='k',
        )
        
        ax.add_collection(circles)
        ax.set_xlim([-3,3])
        ax.set_ylim([-8*dy-3,-2])
        ax.axis('off')
        ax.set_facecolor(bgcolor)
        fig.patch.set_facecolor(bgcolor)
        ax.set_title(palette.upper(),color=fontcolor)
        print(palette)

    pl.show()

