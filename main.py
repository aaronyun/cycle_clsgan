import os
import shlex
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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
# model_type = '/rwgan'
model_type = '/frwgan'
# model_type = '/rrgan'
# model_type = '/robgan'

exp_type = '/e6_v2'
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





zsl_accuracy = compute_accuracy(test_features, test_label, test_id, test_attributes)
gzsl_unseen_accuracy = compute_accuracy(test_features, test_label, np.arange(50), attributes)
gzsl_seen_accuracy = compute_accuracy(test_seen_features, test_seen_label, np.arange(50), attributes)

def compute_accuracy(test_features, test_label, test_id, test_attributes):
    test_data = TensorDataset(test_features, test_label)
    test_batch = 32
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    total_rewards = 0
    # fetch attributes
    # pdb.set_trace()

    sample_labels = test_id
    sample_attributes = test_attributes
    class_num = sample_attributes.shape[0]
    test_size = test_features.shape[0]

    print("class num:", class_num)
    predict_labels_total = []
    re_batch_labels_total = []

    for batch_features, batch_labels in test_loader:

        batch_size = batch_labels.shape[0]

        batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
        sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())

        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 4096)
        relations = relation_network(relation_pairs).view(-1, class_num)

        # re-build batch_labels according to sample_labels

        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(int(index[0][0]))
        re_batch_labels = torch.LongTensor(re_batch_labels)
        # pdb.set_trace()

        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu().numpy()
        re_batch_labels = re_batch_labels.cpu().numpy()

        predict_labels_total = np.append(predict_labels_total, predict_labels)
        re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

    # compute averaged per class accuracy
    predict_labels_total = np.array(predict_labels_total, dtype='int')
    re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
    unique_labels = np.unique(re_batch_labels_total)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(re_batch_labels_total == l)[0]
        acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]
    return acc
