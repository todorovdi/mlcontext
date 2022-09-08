import matplotlib.pyplot as plt
import numpy as np

def plotScoresPerSubj(df, subjects, envs, kte = 'err_sens',
                      ww =4 ,hh = 2, ylim=( -0.3,0.3) ):
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
