import matplotlib.pyplot as plt
from os.path import join as pjoin
import numpy as np
import pandas as pd
from collections.abc import Iterable
import seaborn as sns
import warnings
from config2 import path_fig

def plotScoresPerSubj(df, subjects, envs, kte = 'err_sens',
                      ww =4 ,hh = 2, ylim=( -0.3,0.3) ):
    if isinstance(envs,str):
        envs = [envs]
    nsubj = len(subjects)
    #nsubj = 2
    nr,nc = nsubj, len(envs)

    fig,axs = plt.subplots(nr,nc, figsize=(ww*nc, hh*nr))
    axs = axs.reshape((nr,nc))
    tmins_all  = set( df['tmin'] )
    #for env in envs:

    df[f'{kte}_scores_mean'] =  df[f'{kte}_scores'].apply(lambda x: np.mean(x))
    df[f'{kte}_scores_std']  =  df[f'{kte}_scores'].apply(lambda x: np.std(x))

    for envi,env in enumerate( envs ):
        grandmean_per_tmin = {}
        grandstd_per_tmin  = {}
        for tmin in tmins_all:
            # across subject and folds
            sca = np.array( list(df.loc[df['env'] == env, f'{kte}_scores'] ) )
            #sca = np.array( [ np.array(a) for a in sca ] )
            grandmean_per_tmin[tmin] = sca.mean()
            grandstd_per_tmin [tmin] = sca.std()

        for subji, subject in enumerate( subjects[:nsubj] ):
            dfc = df[(df['subject'] == subject) & (df['env'] == env)]
            dfc = dfc.sort_values(by=['tmin'], key=lambda x: list(map(float,x) ) )
            tmins = dfc['tmin']
            #scores  = dfc[f'{kte}_scores']
            ax = axs[subji,envi]
            mn = dfc[f'{kte}_scores_mean']

            ax.plot(tmins, mn, label=env)
            #std = dfc[f'{kte}_scores_std']  # std fold-wise
            std = [ grandstd_per_tmin[tm] for tm in tmins]
            ax.plot(tmins, mn - std, label=env, ls = ':')
            ax.plot(tmins, mn + std, label=env, ls = ':')

            ax.set_title(f'env = {env}')
            ax.set_ylim(ylim)
            ax.axhline(0,ls=':')
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.tight_layout()

    return axs

def genDefPairs(vns_short, env_names = ['stable','random']):
    from figure.mystatann import square_updiag
    pairs = [(env_names[0] + '_' +  vn, env_names[1] + '_' +  vn) for vn in  vns_short]
    pairs += [(env_names[0] + '_' +  vn1, env_names[0] + '_' +  vn2) for vn1,vn2 in square_updiag(vns_short) ]
    pairs += [(env_names[1] + '_' +  vn1, env_names[1] + '_' +  vn2) for vn1,vn2 in square_updiag(vns_short) ]
    return pairs

def plotOneTminDec(df_b2b_plot, roword, onlyone, vns_short, vsfigns, ypairs=None, varn2pub={},  aspect=1.6,
                   ts=None,subdir = '', save_svg=1, show_fliers = 1, xlim=None, x_ann_offset = None,
                    pos_offset_coef =  1.01,
                    compare_shift_coef = 0.08,
                    compare_offset_coef = 1.22):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re
    import warnings
    from figure.mystatann import plotSigAll, plotSig0All
    from config2 import path_fig

    import matplotlib.ticker as mticker
    locator =mticker.MaxNLocator(3)

    if ypairs is None:
        ypairs = genDefPairs(vns_short)

    #hue_order = ['stable','random']
    #tmin0 = '-0.64'  # -0.50
    tmin0 = '0.00'
    #tmin0 = '0.11'
    #tmin0 = None
    save_png = 0
    dec_type = 'b2b'
    replace_title = 1
    print('tmin0 = ', tmin0)
    pargs = dict(kind='box',  row='suff_short',
                   y = 'varname', hue=None, palette=['orange', 'grey'],
                hue_order= None, row_order=roword, showfliers=show_fliers)
    pargs['y'] = 'env_varname'
    pargs['hue'] = None # 'env'
    pargs['hue_order'] = pargs['palette'] * len(vns_short)
    if tmin0 is None:
        pargs['col'] = 'tmin'
        pargs['col_order'] = ts
    else:
        pargs['col'] = 'varseti'

    ctr = 0
    for xcol in ['vals_to_plot_nb2b']: #, 'vals_to_plot']:
        for tl in ["target","feedback"]:
            for varset, (fign, (ymin,ymax)) in vsfigns.items():
                if onlyone is not None:
                    if (onlyone[0] is None) and ( (dec_type,fign) != onlyone[1:]  ):
                        continue
                    elif (onlyone[0] is not None) and ( (tl,dec_type,fign) != onlyone ):
                        #print(f'Skip {(tl,dec_type,fign)} due to onlyone {onlyone}')
                        continue
                figfnb =  f'b2b_home_{tmin0}_{tl}_{fign}_fl{int(show_fliers)}'
            #for vns_,aspect in [(vns[:4], 1.6), (vns[4:], 1.6)]:
                pargs['x'] = xcol
                qs = ('time_locked == @tl and dec_type == @dec_type'
                                       ' and varset == @varset and varname in @vns_short')
                if tmin0 is not None:
                    qs += ' and tmin == @tmin0'
                df_ = df_b2b_plot.query(qs)
                if len(df_) == 0:
                    print('Empty df_')
                    continue
                pargs['order'] =  sum([ ['stable_' + vn, 'random_' + vn] for vn in  vns_short] ,[])
                ################ PLOT #####################
                with warnings.catch_warnings(record=True):
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    fg = sns.catplot(data=df_, **pargs, aspect=aspect)
                ###########################################
                if xlim is not None:
                    for ax in fg.axes.flatten():
                        ax.set_xlim(xlim)
                if x_ann_offset is None:
                    xlim_ao = df_[pargs['x']].min(), df_[pargs['x']].max() #ax.get_xlim()
                else:
                    xlim_ao = x_ann_offset

                for ax in fg.axes.flatten():

                    ttl = ax.get_title()
                    mr = re.match(f'suff_short = (\w*) \| {pargs["col"]} = (.*)', ttl)
                    if mr is None:
                        continue
                    mrg = mr.groups()
                    suff = mrg[0]
                    coln2 = mrg[1]; coln2e = eval(coln2)

                    if replace_title:
                        ax.set_title(f'Frequency band = {suff}')

                    df__ = df_.query(f'suff_short == @suff and {pargs["col"]} == @coln2e')
                    assert len(df__)
                    #colpair = 'varname'
                    colpair = 'env_varname'
                    coln = xcol
                    rng = xlim_ao[1] - xlim_ao[0]
                    x_single = xlim_ao[1]   * pos_offset_coef
                    x_inc = xlim_ao[1]      * compare_shift_coef
                    x_double1 = xlim_ao[1]  * compare_offset_coef
                    print(f'Starting sig for {figfnb}')
                    plotSigAll(ax, x_double1, x_inc, rng/50, df=df__,
                              coln=coln, colpair=colpair, pooled=False, hor=True, pairs=ypairs)
                    plotSig0All(ax,x_single,'*', df=df__, coln=coln,
                                colpair=colpair, pooled=False, hor=True);

                    ax.axvline(x=0, c='r', ls=':', lw=4)
                    ax.grid(True)

                    tmin0_ = tmin0
                    if tmin0 is None:
                        tmin0_ = coln2
                    if xcol == 'vals_to_plot_nb2b':
                        ax.set_xlabel(f'Normalized\ndecoding accuracy at t={tmin0_}\nrelative to {tl} onset')
                    else:
                        ax.set_xlabel(f'Decoding accuracy at t={tmin0_}\nrelative to {tl} onset')

                    if suff != roword[-1]:
                        ax.set_xlabel('')

                for ax in fg.axes.flatten():
                    xlabs = ax.get_yticklabels()
                    xlabs2 = []
                    for xl in xlabs:
                        tt = xl.get_text()
                        ttcand = '_'.join( tt.split('_')[1:] )
                        xlabs2.append( varn2pub.get(ttcand,tt) )
                    ax.set_yticklabels(xlabs2)
                    ax.set_ylabel('')
                    ax.xaxis.set_major_locator(locator)
                #sns.move_legend(fg, loc=(0.75,0.864) ) #(0,-0.005) )#'lower left')
                with warnings.catch_warnings(record=True):
                    warnings.filterwarnings("ignore", category=UserWarning)
                    plt.tight_layout()
                #plt.grid(True)
                if xcol == 'vals_to_plot_nb2b':
                    figfnb += '_norm'
                fnfig = pjoin(path_fig,subdir,figfnb + '.pdf')
                plt.savefig( fnfig)
                if save_png:
                    plt.savefig( fnfig.replace('.pdf','.png'))
                if save_svg:
                    plt.savefig( fnfig.replace('.pdf','.svg'))
                #     sns.move_legend(fg, loc='upper right')
                #     plt.suptitle(suff)
                #     plt.tight_layout()
                ctr += 1
                print('finished ',figfnb)

                plt.close()
    #        break
    #    break
    print(f'Plotted {ctr} figs')


def plotSlide(df, dfconddif0_pvd, dfconddif0r_pvd, dfconddif_pvd, vsfigns, cols0, onlyone,
        ycols = ['vals_to_plot_nb2b', 'vals_to_plot'], varn2pub={}, vns_short=[], fign_suff='',
        save_png=0, save_svg=0,
        color_stneqr = 'green',alpha_stneqr = 0.2, test1=False, subdir = ''  ):

    trial_dur = 3.9145
    mvt_dur = 0.7558
    fb_dur = 0.25
    ITI_dur = 1.5
    home_dur = 0.5
    wnd_SPoC_dur = 0.464
    xvl = 0 # where to put red vert line

    import datetime; now = datetime.datetime.now()
    import re
    from matplotlib.patches import Patch
    timestr = now.strftime("%d/%m/%y %H:%M")

    PerformanceWarning = pd.errors.PerformanceWarning

    sns.set(font_scale=2.5)
    sns.set_style('whitegrid', {'legend.frameon':True})
    pargs = dict(kind='line',   row='varname',
               x = 'tmin', hue='env',
                 hue_order = ['stable','random'], errorbar='sd',
                 palette=['orange', 'grey'], aspect=3.5, legend=False,
                 facet_kws={'sharey': False, 'sharex': True} )

    control_type_present = df['control_type'].unique()
    time_lockeds_present = df['time_locked'].unique()
    suffixes_present     = df['custom_suffix'].unique()
    dec_types_found      = df['dec_type'].unique()

    if len(control_type_present) > 1:
        pargs['col'] ='control_type'
    ctr = 0

    from itertools import product
    tpls = product(ycols, suffixes_present, time_lockeds_present, dec_types_found)
    for y,suff,tl,dec_type in tpls:
        dfconddif0_pv    = dfconddif0_pvd[y].reset_index().set_index(cols0[:-1])
        dfconddif0r_pv   = dfconddif0r_pvd[y].reset_index().set_index(cols0[:-1])
        dfconddif_pv     = dfconddif_pvd[y].reset_index().set_index(cols0[:-1])
        pargs['y'] = y
    #for y in ycols: #, 'vals_to_plot']:
    #    for suff in  suffixes_present:
    #    #for suff in ['CNslideModelES_trim0_dhittw1_s_scX0Y0c1_drall_dtall_broad_sh0.250']:
    #        for tl in time_lockeds_present:
    #        #for tl in ['feedback']:
    #            for dec_type in dec_types_found:
                    #qs = 'control_type == @ct and custom_suffix == @suff'
        qs = ('time_locked == @tl and custom_suffix == @suff '
            'and dec_type == @dec_type')
        suff_short = suff[14:].replace('_drall_dtall','').\
            replace('h0.232','').replace('_s','').replace('dhittw0cX0Y2c1','').replace('trim0','')

        #tick_skip = 2
        #for varseti, fign, (ymin,ymax) in vsifigns:
        for varset, (fign, (ymin,ymax)) in vsfigns.items():
            if onlyone is not None:
                if (tl,dec_type,fign) != onlyone:
                    #print(f'skip {(tl,dec_type,fign)} due to onlyone {onlyone}')
                    continue
            if y == 'vals_to_plot_nb2b':
                fign += '_norm'
            fign_full = f'{dec_type}_{fign_suff}' +\
                f'{suff_short}_{tl}_{fign}.pdf'
            qs_ = qs + ' and varset == @varset'
            df_ = df.query(qs_)
            #print('len df len(df_)',len(df_))
            #if varseti == -1:
            pargs['row_order'] = list(sorted(df_['varname'].unique()))
            if onlyone is not None:
                pargs['row_order'] = vns_short
                fign += '_!'

            df_g = df_[~df_[y].isna()]
            if not len(df_g):
                print('Get empty for ',(suff,tl,dec_type,fign))
                continue
            print(f'Starting plotting len(df_g)={len(df_g)}  ',(y,suff,tl,dec_type,fign))
            #if varseti >= 0:
            if dec_type != 'classic':
                pargs['facet_kws']={'sharey': True, 'sharex': True}
            else:
                pargs['facet_kws']={'sharey': False, 'sharex': True}
            try:
                fg = sns.relplot(data=df_g, **pargs)
            except KeyError as e:
                print('Err during relplot ',suff,tl,
                      dec_type,varset,e)
                plt.close()
                continue
            axsf = fg.axes.flatten()
            for axi,ax in enumerate(axsf):
                vlinelw = 5
                if dec_type == 'classic':
                    ax.set_ylabel('Decoding accuracy')
                else:
                    if y == 'vals_to_plot':
                        ax.set_ylabel('Causal factor')
                    elif y == 'vals_to_plot_nb2b':
                        ax.set_ylabel('Normalized\ndecoding\nperformance')
                    else:
                        raise ValueError('wrong y')

                ttl = ax.title.get_text()
                if len(control_type_present) > 1:
                    mr = re.match('varname = (\w*) \| control_type = (\w*)', ttl)
                else:
                    mr = re.match('varname = (\w*)', ttl)
                if mr is None:
                    continue
                else:
                    mrg = mr.groups()
                    varname = mrg[0]
                    if len(control_type_present) > 1:
                        ct = mrg[1]
                    else:
                        ct = control_type_present[0]


                if dec_type == 'classic':
                    if varname.find('err_sens') >= 0:
                        ax.set_ylim(-0.05,0.2)
                    elif varname.find('state_') >= 0:
                        ax.set_ylim(-0.15,0.65)
#                         elif varname in ['target','next_error','movement']:
#                             ax.set_ylim(-0.15,0.95)
                    else:
                        #ax.set_ylim(-0.05,0.45)
                        ax.set_ylim(-0.15,0.85)
                else:
                    print(ymin,ymax)
                    if y == 'vals_to_plot_nb2b' and dec_type != "classic":
                        if varname.find('err_sens') >= 0:
                            coef = 0.18
                        else:
                            coef = 0.5
                        ax.set_ylim(ymin*coef,ymax*coef)
                    else:
                        ax.set_ylim(ymin,ymax)

                if onlyone is not None:
                    ax.set_title(varn2pub.get(varname,varname))

                ax.axhline(y=0,   c='r', ls=':', lw=vlinelw)
                ax.axvline(x=xvl, c='r', ls=':', lw=vlinelw)
                if tl == 'feedback':
                    ax.axvline(x=xvl - mvt_dur - home_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl - mvt_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl + fb_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl + fb_dur + ITI_dur, c='b', ls=':', lw=vlinelw)
                elif tl == 'target':
                    ax.axvline(x=xvl - home_dur - ITI_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl - home_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl + mvt_dur, c='b', ls=':', lw=vlinelw)
                    ax.axvline(x=xvl + mvt_dur + fb_dur, c='b', ls=':', lw=vlinelw)

                yminc_,ymaxc_ = ax.get_ylim()
                ymaxc = ymaxc_ * 0.73

                #tpl = ('Ridge',suff,tl,ct,dec_type,varseti, varname)
                tpl = ('Ridge',suff,tl,ct,dec_type,varset, varname)

                # P,D markers are individ ttest > 0 and stable > random
                try:
                    with warnings.catch_warnings():
                        #warnings.filterwarnings('ignore', category=UserWarning)
                        warnings.filterwarnings('ignore', category=PerformanceWarning)

                        tmins_stable_nonz = dfconddif0_pv.loc[tpl].query('signif == True').reset_index(drop=False)
                    tmins_stable_nonz = tmins_stable_nonz['tmin']
                    ax.plot( tmins_stable_nonz.astype(float),
                        len(tmins_stable_nonz) * [ymaxc * 1.3],
                        lw=0, marker='P', label='stable > 0', color='r',markersize=13)

                    with warnings.catch_warnings():
                        #warnings.filterwarnings('ignore', category=UserWarning)
                        warnings.filterwarnings('ignore', category=PerformanceWarning)
                        tmins_random_nonz = dfconddif0r_pv.loc[tpl].query('signif == True').reset_index(drop=False)
                    tmins_random_nonz = tmins_random_nonz['tmin']
                    ax.plot( tmins_random_nonz.astype(float),
                        len(tmins_random_nonz) * [ymaxc * 1.2],
                        lw=0, marker='D', label='random > 0', color='k',markersize=10)
                except KeyError as e  :
                    print('problem with stable or random tmins',type(e),str(e))
                    tmins_random_nonz = []
                    tmins_stable_nonz = []

                try:
                    with warnings.catch_warnings():
                        #warnings.filterwarnings('ignore', category=UserWarning)
                        warnings.filterwarnings('ignore', category=PerformanceWarning)
                        dfsub = dfconddif_pv.loc[tpl].reset_index(drop=False)
                    dfsub['tmin'] = dfsub['tmin'].astype(float)
                    dfsub = dfsub.sort_values('tmin')
                    tmins_stable_g_random,sig = dfsub[['tmin','signif']].to_numpy().T
                    # plus markers are same as shading
                    sig = sig.astype(bool)
                    tmins_stable_g_random = tmins_stable_g_random.astype(float)
                    #import matplotlib.patches as mpatches
                    fill= ax.fill_between(tmins_stable_g_random,
                        ymaxc, where = sig, color=color_stneqr, alpha=alpha_stneqr)
#                         ax.plot( tmins_stable_g_random[sig],
#                             np.sum(sig) * [ymax * 1.2],
#                             lw=0, marker='+', label='stable != random')
                    ##poly = mcollections.PolyCollection(fill.get_paths(),
                    #        facecolor='red', alpha=0.1,
                    #        label='stable != random')

                except KeyError as e:
                    print('proble with tmins_stable_g_random tmines',str(e))
                    print(tpl, dfconddif_pv.index)
                    tmins_stable_g_random,sig = [],[]
                print(varname, 'st>tnd {}  st>0 {} rnd >0 {}'.format(len(tmins_stable_g_random), len(tmins_stable_nonz), len(tmins_random_nonz)))

                #xticks = ax.get_xticks()[::tick_skip]
                #ax.set_xticks(xticks)
                ax.grid(True)

                if axi == len(axsf) - 1:
                    ax.legend(loc='lower right',markerscale=2.5,
                              framealpha=0.55, fontsize='medium')
                    ax.set_xlabel('Window start [s]')
                if axi == len(axsf) - 2:
                    llw = 10
                    from matplotlib.lines import Line2D
                    from matplotlib.text import Text
#                         text = Text(0.5, 0.5, 'eeddddddddddde',
#                                     visible=True, label='environment')
#                         ax.add_artist(text)
                    l1 = Line2D([0], [0], color='orange', label='stable', lw=llw)
                    l2 = Line2D([0], [0], color='grey', label='random', lw=llw)
                    poly = Patch(facecolor=color_stneqr, alpha=alpha_stneqr,
                            label='stable !=\n random')
                    ax.legend(handles=[ l1, l2, poly], facecolor='white',
                              #labels=['1','2','3'],
                        loc='lower right', framealpha=0.75, fontsize='medium')
                if axi == len(axsf) - 3:
                    llw = 8
                    from matplotlib.lines import Line2D
                    from matplotlib.text import Text
#                         text = Text(0.5, 0.5, 'eeddddddddddde',
#                                     visible=True, label='environment')
#                         ax.add_artist(text)
                    l1 = Line2D([0], [0], color='red', label=tl + ' onset', lw=llw, ls=':')
                    l2 = Line2D([0], [0], color='blue', label='task stages', lw=llw, ls=':')
                    ax.legend(handles=[ l1, l2], facecolor='white',
                              #labels=['1','2','3'],
                        loc='lower right', framealpha=0.75, fontsize='small')
            try:  # save figures
                #sns.move_legend(fg, loc=(0.75,0.),
                #               facecolor='white', edgecolor='k')
                #sns.move_legend(fg, loc=(0.85,0.22))
                #sns.move_legend(fg, loc='lower right')
                #    edgecolor='black', markerscale=5., framealpha=1.)
                if onlyone is None:
                    plt.suptitle(f'{fign}; tl={tl} dec_type={dec_type} {timestr}')#, x=0.2)
                plt.tight_layout()
                plt.savefig( pjoin(path_fig, subdir, fign_full),
                            metadata = {'Keywords':f'{suff},{tl}'})
                if save_png:
                    plt.savefig( pjoin(path_fig, subdir, fign_full.replace('.pdf','.png')),
                            metadata = {'Keywords':f'{suff},{tl}'})
                if save_svg:
                    #plt.savefig( pjoin(path_fig, subdir, fign_full.replace('.pdf','.eps')))
                    plt.savefig( pjoin(path_fig, subdir, fign_full.replace('.pdf','.svg')))
                plt.close()
            except ValueError as e:
                print('!!! EXC',e)
            print(f'{dec_type} varset={varset} fig done')
            ctr += 1
            if ctr > 0 and test1:
                break

    #                if ctr > 0 and test1:
    #                    break
    #            if ctr > 0 and test1:
    #                break
    #        if ctr > 0 and test1:
    #                break
    print(f'Finished, plotted {ctr} figures')
    return fg,ctr

def genStRandLegendHandles(rect=True, include_labels = False):
    '''
    ret stable,random
    '''
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    if include_labels:
        ils = dict(label='stable')
        ilr = dict(label='random')
    else:
        ils,ilr = {},{}
    if rect:
        rect1 = mpatches.Rectangle((0, 0), 1, 1, color='tab:orange', **ils)
        rect2 = mpatches.Rectangle((0, 0), 1, 1, color='tab:grey', **ilr)
    else:
        ls = '-'
        llw = 2
        rect1 = Line2D([0], [0], color='tab:orange', lw=llw, ls=ls, **ils)
        rect2 = Line2D([0], [0], color='tab:grey', lw=llw, ls=ls, **ilr)
    return list([rect1,rect2])

def plotPatterns(X,y,epinfo,precalc_patterns, precalc_filters):
    from mne.decoding import SPoC
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict

    spoc0 = SPoC(n_components=2, log=True, reg="oas", rank="full")
    clf = make_pipeline(spoc0, Ridge())
    # Define a two fold cross-validation
    #cv = KFold(n_splits=2, shuffle=False)
    #train,_ = cv.split(X,y)
    train = np.arange(4)
    clf.fit( X[train],y[train] )
    spoc = clf.named_steps['spoc']
    spoc.filters_  = precalc_filters
    spoc.patterns_ = precalc_patterns
    # epinfo = meg_epochs.info
    spoc.plot_patterns(epinfo)

    ## GeneralizingEstimator -- to make temporal get
    #time_gen = GeneralizingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
    ## again, cv=3 just for speed
    #scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=None)
    ## Mean scores across cross-validation splits
    #scores = np.mean(scores, axis=0)

    #fig, ax = plt.subplots(1, 1)
    #im = ax.imshow(
    #scores,
    #interpolation="lanczos",
    #origin="lower",
    #cmap="RdBu_r",
    #extent=epochs.times[[0, -1, 0, -1]],
    #vmin=0.0,
    #vmax=1.0,
    #)
    #ax.set_xlabel("Testing Time (s)")
    #ax.set_ylabel("Training Time (s)")
    #ax.set_title("Temporal generalization")
    #ax.axvline(0, color="k")
    #ax.axhline(0, color="k")
    #cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label("AUC")

def relplot_multi(sep_ys_by = 'hue', **kwargs):
    '''like relplot but for multiple ys (they go get separated by hue)
    sep_ys_by is WITHIN row separation
    '''
    assert 'y' not in kwargs
    assert sep_ys_by not in kwargs
    assert 'data' in kwargs
    assert 'x' in kwargs
    assert '__varname' not in kwargs['data']
    assert '__varval' not in kwargs['data']
    assert '__varrow' not in kwargs['data']
    kind = kwargs.get('kind','line')

    if sep_ys_by == 'col' and kind == 'density':
        raise ValueError('not implemented, use row or hue. Col is supposed to be used for condition variable')

    if 'facet_kws' not in kwargs:
        kwargs['facet_kws'] = {'sharex':True, 'sharey':False}

    df = kwargs['data'].copy()    
    assert len(df)
    ys = kwargs['ys']

    szstr = ''
    tic = 'trial_index'
    if tic not in df:
        tic = 'trials'
    if tic in df:
        dfsz = df.groupby([tic]).size()
        szmin,szmax = dfsz.min(),dfsz.max()
        if szmin != szmax:
            szstr = f'N={szmin}-{szmax}'
        else:
            szstr = f'N={szmin}'


    def density_plot(x,y, **kwargs):
        #ax = sns.kdeplot(data=dfcs_fixhistlen,
        #           x='err_sens',y=vn, fill=False, hue=coln_col)
        df__ = pd.DataFrame( np.array(list(zip(x,y))))
        ax = sns.kdeplot(data=df__,
                x=x,y=y, fill=True, **kwargs)
        ax.axhline(0,ls=':', c='r')
        ax.axvline(0,ls=':', c='r')

    dfs = []

    df['__varname'] = pd.Series(['']*len(df), dtype=str)
    df['__varrow'] = pd.Series([-1]*len(df), dtype=int)
    df['__varval'] = pd.Series([0.]*len(df), dtype=float)
    if isinstance(ys[0], str):
        for i,yn in enumerate(ys):
            df['__varval' ] = df[yn]
            df['__varname'] = yn + f' {szstr}'
            dfs += [df.copy()]
        df = pd.concat(dfs, ignore_index = True)
        del kwargs['data']
        del kwargs['ys']
        #for i,yn in enumerate(kwargs['ys']):
        kwargs['data'] = df
        if kind == 'density':
            raise ValueError('not implemented, use version with list of lists')
        fg = sns.relplot(**kwargs,y='__varval',
                         **{sep_ys_by:'__varname'} )

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            del kwargs['ylabel'] 
            for ax in fg.axes.flatten():
            #fg.axes[0,0].set_ylabel(ylabel)
                ax.set_ylabel(ylabel)
        else:        
            for ax in fg.axes.flatten():
                ax.set_ylabel(ys[0])
            #fg.axes[0,0].set_ylabel(ys[0])
    else:
        assert 'row' not in kwargs
        for i,yns in enumerate(ys):
            for j,yn in enumerate(yns):
                df['__varval' ] = df[yn]
                ynext = yn + f' {szstr}'  
                if kind == 'line':
                    df['__varname'] = ynext
                else:
                    df['__varname'] = yn
                df['__varrow'] = i
                dfs += [df.copy()]
        df = pd.concat(dfs, ignore_index = True)

        del kwargs['data']
        del kwargs['ys']
        kwargs['data'] = df

        if 'ylabel' in kwargs:
            ylabels = kwargs['ylabel']
            assert isinstance(ylabels, Iterable) and not isinstance(ylabels, str)
            del kwargs['ylabel'] 
        else:
            ylabels = None

        if 'ylim' in kwargs:
            ylims = kwargs['ylim']
            del kwargs['ylim'] 
        else:
            ylims = None


        if kind == 'density':
            x = kwargs['x']
            del kwargs['x']
            subkws = kwargs['facet_kws']
            del kwargs['facet_kws']
            fg = sns.FacetGrid(**{sep_ys_by:'__varname'}, **kwargs, **subkws) #col='__varrow', data=kwargs['data'])
            fg.map(density_plot,  x, "__varval", label='')          
        else:
            fg = sns.relplot(**kwargs,y='__varval',
                         **{sep_ys_by:'__varname'}, row='__varrow')

        for i,yns in enumerate(ys):
            if ylabels is not None:
                fg.axes[i,0].set_ylabel(ylabels[i])
            else:
                fg.axes[i,0].set_ylabel(yns[0])

        #print(fg.axes.shape)
        if ylabels is not None:
            for i,yns in enumerate(ys):
                if ylims is not None:
                    if ylims[i] is not None:
                        print(i,ylims[i] )
                        fg.axes[i,0].set_ylim(ylims[i])


        for i,yns in enumerate(ys):
            fg.axes[i,0].set_title('')
            #else:
            #    fg.axes[i][0].set_ylabel(yns[0])

    return fg, df


def make_fig3(df_, palette, hue_order, col_order, ps_2nice, hues, pswb2r, pswb2pr, coord_let, coord_let_shift, show_plots=0):
    from config2 import path_fig
    #palette=['blue', 'orange', 'green', 'olive','cyan','brown']

    #vn = 'err_sens_abserrcorr'
    #'err_sens_prevabserrcorr'
    #'err_sens_prev_error_abs_resid'
    for lablet, varn_y, pswb2, rtype, ylab in zip(['A','B'],
        ['err_sens', 'err_sens_prevabserrcorr' ],[pswb2r, pswb2pr],
        ['mean r','mean partial r'],
        ['Error sensitivy','Error sensitivy conditioned\non previous absolute error']):

        fg = sns.relplot(data=df_, kind='line',
                    x='trialwpertstage_wb', col='ps2_',
                   y=varn_y, hue='pert_stage_wb',
                         errorbar = 'sd', palette = palette,
                   facet_kws={'sharex':False},
                         hue_order=hue_order,
                    col_order = col_order, legend=None)
        # for ax in fg.axes.flatten():
        #     ax.axhline(0,ls=':',c='red', alpha=0.7)
            
        for i, ax in enumerate(fg.axes.flat):
            col_ = fg.col_names[i]
            ax.set_title(ps_2nice[ax.get_title()[7:]] )
            if col_ != 'rnd':        
                sp = np.array(palette)[hues[i]]
                sp = list(sp) + list(sp)
                sns.lineplot(data=df_[df_['ps2_'] == col_], 
                    x='trialwpertstage_wb', y='pred', 
                    hue='pert_stage_wb', ax=ax, legend=None,
                            palette = sp, dashes=[4,2])

            r_value = pswb2['r'][col_]
            if r_value is not None:
                ax.text(0.05, 0.95, f'{rtype} = {r_value:.2f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left', fontsize=12)
            else:
                print(col_, r_value)
            
        #addTitleInfo(fg.axes.flat[0])
            
        print(fg.hue_kws, fg.hue_names)
        fg.refline(y=0, color='red')
        #fg.map(plt.hist, 'tip').refline(0.15)
        #fg.set_titles('{col_name}')
        fg.set_xlabels('Trial number')
        fg.set_ylabels(ylab)
            
        ax = fg.axes.flat[0]
        ax.annotate(lablet, xy=coord_let, xytext=coord_let_shift, 
          fontsize=19, fontweight='bold', va='top', ha='left',
          xycoords='axes fraction', textcoords='offset points')
        fnfig = pjoin(path_fig, 'behav', f'Fig3_{lablet}_dynESpert_stage')
        plt.savefig(fnfig + '.png')
        plt.savefig(fnfig + '.pdf')
        plt.savefig(fnfig + '.svg')
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close()

        ###############################


        #fg = sns.relplot(data=df_, kind='line',
        #            x='trialwpertstage_wb', col='ps2_',
        #           y=vn, hue='pert_stage_wb',
        #                 errorbar = 'sd', palette = palette,
        #           facet_kws={'sharex':False},
        #                 hue_order=hue_order,
        #            col_order = col_order, legend=None)
        ## for ax in fg.axes.flatten():
        ##     ax.axhline(0,ls=':',c='red', alpha=0.7)
        #    
        #for i, ax in enumerate(fg.axes.flat):
        #    col_ = fg.col_names[i]
        #    ax.set_title(ps_2nice[ax.get_title()[7:]] )
        #    if col_ != 'rnd':        
        #        sp = np.array(palette)[hues[i]]
        #        sp = list(sp) + list(sp)
        #        sns.lineplot(data=df_[df_['ps2_'] == col_], 
        #            x='trialwpertstage_wb', y='ppred', 
        #            hue='pert_stage_wb', ax=ax, legend=None,
        #                    palette = sp, dashes=[4,2])      
        #    
        #    r_value = pswb2pr['r'][col_]
        #    if r_value is not None:
        #        ax.text(0.05, 0.95, f'mean partial r = {r_value:.2f}', transform=ax.transAxes,
        #            verticalalignment='top', horizontalalignment='left', fontsize=12)
        #    else:
        #        print(col_, r_value)
        #    
        ##addTitleInfo(fg.axes.flat[0])
        #    
        #print(fg.hue_kws, fg.hue_names)
        #fg.refline(y=0, color='red')

        ##fg.set_titles('{col_name}')
        #fg.set_xlabels('Trial number')
        #fg.set_ylabels('Error sensitivy conditioned\non previous absolute error')


        #lablet = 'B'
        #ax = fg.axes.flat[0]
        #ax.annotate(lablet, xy=coord_let, xytext=coord_let_shift, 
        #  fontsize=19, fontweight='bold', va='top', ha='left',
        #  xycoords='axes fraction', textcoords='offset points')
        #fnfig = pjoin(path_fig, 'behav', f'Fig3_{lablet}_dynESpert_stage')
        #plt.savefig(fnfig + '.png')
        #plt.savefig(fnfig + '.svg')
        #plt.savefig(fnfig + '.pdf')
        #plt.tight_layout()
        ##plt.show()
        #plt.close()

