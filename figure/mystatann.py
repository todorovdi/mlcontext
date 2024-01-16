from behav_proc import compare0,comparePairs
import numpy as np
def plotSig0(ax,x,y,txt='*',hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided', verbose=0):
        #TODO:::
    ttrs1 = compare0(df.query(f'{colpair} == @x'), coln, alt=[alt] )
    qs = 'alternative == @alt'
    ttrssig = ttrs1.query(qs + ' and pval <= 0.05')
    if verbose:
        display(ttrs1)

    lab2tick = getLab2Tick(ax, hor)
    #print(lab2tick)
    if not isinstance(x,str):
        x = str(x)

    if x not in lab2tick:
        print(lab2tick)
    xt = lab2tick[x]

    if len(ttrssig):
        #print(f'xt = {xt} y ={y}')
        if hor:
            ax.text(y,xt,txt)
        else:
            ax.text(xt,y,txt)
    return ttrssig

def plotSig0All(ax,y,txt='*',hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided'):
    for x in df[colpair].unique():
        try:
            ttrssig = plotSig0(ax,x,y,txt=txt,hor=hor, df=df, coln=coln, colpair=colpair, paired=paired,
            pooled=pooled, alt=alt)
            ttrssig['varval'] = x
            #if verbose:
            #    display(ttrssig)
        except KeyError as e:
            print('KeyErrro ',str(e))
            raise e

def getLab2Tick(ax ,hor=False):

    if hor:
        tick_labels = ax.get_yticklabels()
        tick_locations = ax.get_yticks()
    else:
        tick_labels = ax.get_xticklabels()
        tick_locations = ax.get_xticks()

    tick_labels = [lab.get_text() for lab in tick_labels]
    lab2tick = dict(zip(tick_labels,tick_locations))

    #lab2tick = {}
    #if hor:
    #    labs = ax.get_yticklabels()
    #else:
    #    labs =ax.get_xticklabels()
    #for lab in labs: #, ax.get_xticks()
    #    if hor:
    #        lab2tick[lab.get_text()] = lab._y
    #    else:
    #        lab2tick[lab.get_text()] = lab._x
    return lab2tick

def plotSig(ax,x1,x2,y,ticklen=2,txt='*',hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided', verbose=0, meanloc_voffset = 0, graded_signif = True,
           fontsize = None ):
    # x1 and x2 are tick labels

    lab2tick = getLab2Tick(ax, hor)
    #print(lab2tick)

    df_ = df.query(f'{colpair} in [@x1,@x2]')
    assert len(df_)
    ttrssig,ttrs = comparePairs(df_, coln, colpair, paired=paired, alt=alt)
    if (ttrssig is None):
        if verbose:
            display(ttrs)
        #print('no sig')
        return []
    pooled = bool(pooled)
    ttrssig = ttrssig.query('pooled == @pooled and alternative == @alt')
    if len(ttrssig) == 0:
        return []
    assert len(ttrssig) <= 1

    if verbose:
        display(ttrssig)

    if not isinstance(x1,str):
        x1 = str(x1)
    if not isinstance(x2,str):
        x2 = str(x2)
    x1t,x2t = lab2tick[x1],lab2tick[x2]
    meanloc = np.mean([x1t,x2t]) + meanloc_voffset
    if hor:
        ax.plot([y-ticklen,y,y,y-ticklen], [x1t,x1t,x2t,x2t], c='k')
    else:
        ax.plot([x1t,x1t,x2t,x2t],
                [y-ticklen,y,y,y-ticklen], c='k')

    if graded_signif:
        txt = ttrssig.iloc[0]['starcode']
        #print(ttrssig)
    #print(x1,x2, y, len(ttrssig))
    #print(meanloc)
    #meanloc = 0
    if hor:
        ax.text(y,meanloc,txt)
    else:
        ax.text(meanloc,y,txt, ha='center', fontsize = fontsize)
    return ttrssig

def plotSigAll(ax, yst, yinc, ticklen=2,txt='*',hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided', verbose=0, pairs = None, meanloc_voffset = 0,
               graded_signif = True, fontsize = None):
    vals = list(sorted(df[colpair].unique()))
    ycur = yst

    if pairs is None:
        pairs = square_updiag(vals)
    pairs = list(pairs)
    print('pairs = ',pairs)
    for x1,x2  in pairs:
    #for i,x1 in enumerate(vals):
    #    for x2 in vals[i+1:]:
        try:
            r = plotSig(ax,x1,x2,ycur,ticklen=ticklen,txt=txt,
                        hor=hor, df=df, coln=coln, colpair=colpair,
                        paired=paired,pooled=pooled, verbose=verbose,
                        alt=alt, meanloc_voffset = meanloc_voffset, graded_signif = graded_signif,
                        fontsize = fontsize)
            if len(r):
                ycur += yinc
        except KeyError as e:
            print(str(e))

    return ycur

def square_updiag(iterables):
    # prouct without duplicates and diag
    import itertools
    seen = set()
    for item in itertools.product(iterables,iterables):
        key = tuple(sorted(item))
        if key[0] == key[1]:
            continue
        if key not in seen:
              yield item
              seen.add(key)

