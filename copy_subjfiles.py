from config2 import subjects
import os.path as op
import shutil
import os

path_data = '/Volumes/data/MemErrors/data2/'
dst_path = '/Users/quentinra/Desktop/results'

do_copy = False
if do_copy:
    for subject in subjects:
        print(subject)
        src = op.join(path_data, subject, 'results', 'td_long2_0.1_short_with_ICA')
        files = os.listdir(src)
        dst = op.join(dst_path, subject, 'td_long2_0.1_short_with_ICA')
        if not op.exists(dst):
            os.makedirs(dst)
        for file in files:
            dst_file = op.join(dst, file)
            src_file = op.join(src, file)
            shutil.copy(src_file, dst_file)
