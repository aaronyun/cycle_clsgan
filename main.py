import os
import shlex
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# exp_sets = ['/gzsl', '/zsl']
# exp_sets = ['/zsl']
exp_sets = ['/gzsl']

# model_type = '/clswgan'
# model_type = '/wgan'
# model_type = '/rwgan'
# model_type = '/dwgan'
# model_type = '/bcclswgan'
# model_type = '/bcwgan'
# model_type = '/bcrwgan'
# model_type = '/rjwgan'
# model_type = '/mmcrwgan'
# model_type = '/rawgan'
model_type = '/mmc_rclswgan'

exp_type = '/base'
# exp_type = '/best'
# exp_type = '/e2_batchTune'
# exp_type = '/e4_r'
# exp_type = '/e3_synNum'

datasets = ['/apy', '/awa', '/awa2', '/cub', '/flo', '/sun']

for exp_set in exp_sets:

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
                subprocess.run(args, stdout=result_f)
                # subprocess.run(args)
                print('----------')
                print('%s #Done!#' % file_path)
                print('----------')