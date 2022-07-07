#!/bin/python3
with open("exec_output.log") as f:
    lines = f.readlines()

#ended on sub11 hpass0.1 rt = Ridge  freq broad

import re

for li,line in enumerate(lines):
    #s = 'Opening raw data file'
    s = '__START'
    s2 = '----- starting env = '
    s3 = '---------- Starting freq ='
    if line.startswith(s):
        #r = re.match('.*(sub[0-9]+)_.*/.*', line)
        r = re.match('.*subj=(sub[0-9]+).*, hpass=([\w.]+), regression_type=(\w+) at (.*)', line)
        if r is None:
            print(line)
        else:
            subj = r.groups()[0]
            hpass = r.groups()[1]
            rt = r.groups()[2]
            dt = r.groups()[3]
            print(f'{subj:6}, hpass={hpass:10}, rt={rt:8}, Started at {dt[:19]}')

    elif line.startswith(s2):
        print(line[:-1])
    elif line.startswith(s3):
        print(line[:-1])
