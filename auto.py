import os
import subprocess

exp_set = '/zsl'
# exp_set = '/gzsl'
exp_type = '/r_net'
data_sets = ['/cub', '/flo', '/sun', '/awa']

print(os.curdir)
# cuda_device = 'CUDA_VISIBLE_DEVICES=1'
for data_set in data_sets:
    exp_file_path = './exp' + exp_set + exp_type + data_set
    shell_files = os.listdir(exp_file_path)

    if len(shell_files) > 0:
        for file in shell_files:
            out_file = open(exp_file_path + '/' + os.path.splitext(file)[0] + '.txt', 'w')
            subprocess.run(['sh', file], stdout=out_file)
            out_file.close()