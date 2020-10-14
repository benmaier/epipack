from epipack.colors import * 

if __name__ == "__main__": # pragma : no cover

    from matplotlib.collections import LineCollection, EllipseCollection
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['Helvetica','Arial']
    import matplotlib.pyplot as pl

    for ipal, (palette, this_palette) in enumerate(palettes.items()):
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

        ax.text(0, -2.2, bgcolor.upper()[1:], transform=ax.transData, ha='center',va='center', color=fontcolor)

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

        ax.plot([-dx-2*r, -dx/3],2*[-(i+1)*dy-3],color=hex_link_colors[palette],lw=3) 
        ax.text(dx/3, -(i+1)*dy-3, hex_link_colors[palette][1:].upper(), transform=ax.transData, ha='left',va='center', color=fontcolor)

        ax.set_xlim([-3,3])
        ax.set_ylim([-8.5*dy-3,-2])
        ax.axis('off')
        ax.set_facecolor(bgcolor)
        fig.patch.set_facecolor(bgcolor)
        ax.set_title(palette.upper(),color=fontcolor)
        print(palette)

        fig.savefig('{0:02d}.png'.format(ipal),dpi=300)

    pl.show()

