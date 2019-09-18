# -*- coding: utf-8 -*-
from torch.utils.data import TensorDataset, DataLoader

def compute_accuracy(test_features, test_label, test_id, test_attributes):
    test_batch = 32
    total_rewards = 0

    test_data = TensorDataset(test_features, test_label)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

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

        # expanding
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