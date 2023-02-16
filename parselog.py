#!/bin/python3
with open("exec_output.log") as f:
    lines = f.readlines()

# ended on sub11 hpass0.1 rt = Ridge  freq broad

import re

for li, line in enumerate(lines):
    # s = 'Opening raw data file'
    s = '__START'
    s2 = '----- starting env = '
    s3 = '---------- Starting freq ='
    if line.startswith(s):
        # r = re.match('.*(sub[0-9]+)_.*/.*', line)
        r = re.match(r'.*subj=(sub[0-9]+).*, hpass=([\w.]+), regression_type=(\w+)'
                     r', freq_name=(\w+), env=(\w+) at (.*)', line)
        # print(line[:-1])
        if r is None:
            print(line[:-1])
        else:
            subj = r.groups()[0]
            hpass = r.groups()[1]
            rt = r.groups()[2]
            freq_name = r.groups()[3]
            env = r.groups()[4]
            dt = r.groups()[-1]
            print(f'{subj:5}, hpass={hpass:9}, rt={rt:7}, freq={freq_name:5}, env={env}, '
                  f'start={dt[:19]}')

    # elif line.startswith(s2):
    #     print(line[:-1])
    # elif line.startswith(s3):
    #     print(line[:-1])
