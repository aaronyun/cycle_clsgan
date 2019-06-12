import os
import shlex
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

exp_set = '/zsl'
# exp_set = '/gzsl'

exp_type = '/bc'

data_sets = ['/awa', '/cub', '/flo', '/sun', '/awa2', '/apy']

for data_set in data_sets:
    exp_file_path = './exp' + exp_set + exp_type + data_set
    # 取当前目录里的所有文件的名字（没有子文件夹的名字）
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