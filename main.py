#!/usr/bin/python3.6

import os
import shlex
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

exp_setting = '/zsl'
# exp_setting = '/gzsl'

# model_type = '/bcclswgan'
# model_type = '/bcwgan'
# model_type = '/bcrwgan'
# model_type = '/clswgan'
# model_type = '/wgan'
# model_type = '/rwgan
model_type = '/dwgan'

exp_type = '/base'
# exp_type = '/new_datasets'

data_sets = ['/awa', '/cub', '/flo', '/sun', '/awa2', '/apy']

for data_set in data_sets:
    exp_file_path = './exp' + exp_setting + model_type + exp_type + data_set

    shell_files = []
    for (dirpath, dirnames, filenames) in os.walk(exp_file_path):
        shell_files.extend(filenames)
        break

    if len(shell_files) > 0:
        for file in shell_files:
            # 把当前shell文件里的参数保存为列表
            file_path = exp_file_path + '/' + file
            with open(file_path,'r') as f:
                args_in_file = f.readline()
                args_in_file = shlex.split(args_in_file)

            # 用子进程跑对应的代码，并把输出写入到指定文件
            with open(exp_file_path + '/result/' + os.path.splitext(file)[0] + '.txt', 'w') as f:
                print('----------')
                print('%s #Begin!#' % file_path)
                print('----------')
                subprocess.run(args_in_file, stdout=f)
                # subprocess.run(args_in_file)
                print('----------')
                print('%s #Done!#' % file_path)
                print('----------')