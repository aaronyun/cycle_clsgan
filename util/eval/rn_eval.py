# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.metrics import accuracy_score

from util.mlp import AttributeNet, RelationNet

def compute_accuracy(netAtt, netRN, test_features, test_labels, attributes):
    """

    Args:
        test_features: test visual features
        test_labels: label of test features
        attributes: all the attributes of dataset

    Returns:
        acc: mean accuracy of all test classes
    """
    # Data Preperation
    test_data = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Unique Info
    unique_test_labels = np.unique(test_labels.numpy())
    unique_attributes = torch.index_select(attributes, 0, torch.LongTensor(unique_test_labels))
    unique_cls_num = unique_test_labels.shape[0]

    predict_labels_total = []
    re_batch_labels_total = []

    for batch_features, batch_labels in test_loader:
        current_batch_size = batch_features.size(0)

        # Relation Pair Preperation
        ## attributes prepare
        ### up-sampling attributes
        up_attributes_v = netAtt(Variable(unique_attributes).cuda())
        ### extend up-sampled attributes
        up_attributes_ext = (up_attributes_v.data.cpu()).unsqueeze(0).repeat(current_batch_size, 1, 1)
        ## visual features prepare
        ### extend batch features
        batch_features_ext = batch_features.unsqueeze(0).repeat(unique_cls_num, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        ## relation pairs
        relation_pairs = torch.cat((up_attributes_ext, batch_features_ext), 2).view(-1, 4096)

        # Relaton Score
        relations = netRN(Variable(relation_pairs).cuda()).view(-1, unique_cls_num) # (batch_size, cls_num)

        # Get Predicted Label and Re-projected Label
        ## predicted labels
        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu().numpy()
        ## re-build batch_labels according to unique_test_labels
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(unique_test_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = np.array(re_batch_labels)

        print(predict_labels)
        print(re_batch_labels)
        # Store Final Label Result
        predict_labels_total = np.append(predict_labels_total, predict_labels)
        re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

    # Compute Averaged Per Class Accuracy
    predict_labels_total = np.array(predict_labels_total, dtype='int')
    re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
    unique_labels = np.unique(re_batch_labels_total)

    acc = 0
    for l in unique_labels:
        idx = np.nonzero(re_batch_labels_total == l)[0]
        acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]

    return acc