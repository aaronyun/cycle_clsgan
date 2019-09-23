import os
import shlex
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# exp_set = '/zsl'
exp_set = '/gzsl'

# model_type = '/clswgan'
# model_type = '/wgan'
# model_type = '/dwgan'
# model_type = '/bcclswgan'
# model_type = '/bcwgan'
# model_type = '/bcrwgan'
# model_type = '/rjwgan'
# model_type = '/mmcrwgan'
# model_type = '/rawgan'
# model_type = '/mmc_rclswgan'
# model_type = '/mmcfrwgan'
model_type = '/rwgan'
# model_type = '/frwgan'
# model_type = '/rrgan'
# model_type = '/robgan'

exp_type = '/e8_v1.1'
# exp_type = '/base'
# exp_type = '/for_test'

# datasets = ['/APY', '/AWA1', '/AWA2', '/CUB', '/FLO', '/SUN']
datasets = ['/CUB']

print('\n')
print("########## EXPERIMENT SETTING: %s ##########" % exp_set)
print('\n')

for dataset in datasets:
    exp_file_path = './exp' + exp_set + model_type + exp_type + dataset

    shell_files = []
    for (dirpath, dirnames, filenames) in os.walk(exp_file_path):
        shell_files.extend(filenames)
        break

    # if len(shell_files) > 0:
    for file in shell_files:
        # 把当前shell文件里的参数保存为列表
        file_path = exp_file_path + '/' + file
        with open(file_path, 'r') as exp_args:
            args = exp_args.readline()
            args = shlex.split(args)

        with open(exp_file_path + '/result/' + os.path.splitext(file)[0] + '.txt', 'w') as result_f:
            print('----------')
            print('%s #Begin!#' % file_path)
            print('----------')
            # subprocess.run(args, stdout=result_f)
            subprocess.run(args)
            print('----------')
            print('%s #Done!#' % file_path)
            print('----------')