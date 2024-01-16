import pandas as pd
from os.path import join as pjoin

d = pjoin(path_code, 'exper_protocol')
fnf = pjoin(d,'dataMEG', 'Dmitrii_test_context_change_20230426_092338_trigger.log')
with open(fnf, 'r') as f:
    lines = f.readlines()

df = pd.read_csv(fnf, delimiter=';', names = ['trigger', 'time','addinfo'])

def f(row):
    r = row['addinfo']
    tind = -100
    print(r, type(r))
    if (r is not None) and (not isinstance(r,float)):
        r = eval(r)
        tind = r.get('trial_index')
    return tind
df['trial_index'] = df.apply(f,1)
print(df)
