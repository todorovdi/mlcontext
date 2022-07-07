import os
import multiprocessing

# path_data = '/Volumes/Samsung_T1/MEGdata/MemErrors/'
# path_data = '/Volumes/data/MemErrors/data/'
# path_data = '/data/quentinra/MemErrors/data2/'
path_data = os.path.expandvars("$DATA_MEMORY_ERRORS_STAB_AND_STOCH")
path_fig = os.path.expandvars("$FIG_MEMORY_ERRORS_STAB_AND_STOCH")
                                #FIG_MEMORY_ERRORS_STAB_AND_STOCH

subjects_predef = ['sub01_WGPOZPEE', 'sub02_CLTLNQWL', 'sub03_GPVDQMWB',
            'sub04_XNDMUSRS', 'sub05_ZGPBOAQU', 'sub06_DLLYEPVA',
            'sub07_MJWXBESS', 'sub08_TXVPROYY', 'sub09_VFDOXEVC',
            'sub10_BJJWDKEK', 'sub11_ERHGZFPL', 'sub12_ZWFBQSXR',
            'sub13_EALZKBNL', 'sub14_RPEADEJG', 'sub15_TAMMXQQS',
            'sub16_SJILLGUV', 'sub17_SUMYMRAR', 'sub18_BBPOBFOQ',
            'sub19_MVAQVMEL', 'sub20_YOGCJKKB']

# they are same
subjects = [f for f in os.listdir(path_data) if f.startswith('sub') ]
subjects = list(sorted(subjects))

n_jobs = multiprocessing.cpu_count() - 2
XGB_tree_method_def = 'gpu_hist'


freq_names = ['broad', 'theta', 'alpha', 'beta', 'gamma']
freqs = [(4, 60), (4, 7), (8, 12), (13, 30), (31, 60)]
freq_name2freq = dict( list(zip(freq_names,freqs) ) )
