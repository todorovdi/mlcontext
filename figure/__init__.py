
def renameTickLabels(ax, d, hor=False, rotation=0):
    assert isinstance(d,dict)
    if hor:
        old = ax.get_yticklabels()
    else:
        old = ax.get_xticklabels()

    new = [ d[tl.get_text()] for tl in old ]

    if hor:
        ax.set_yticklabels(new, rotation=rotation)
    else:
        ax.set_xticklabels(new, rotation=rotation)

    return new

palette_stabrand = ['tab:orange', 'tab:grey']
