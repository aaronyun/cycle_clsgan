# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.metrics import accuracy_score

from util.mlp import AttributeNet, RelationNet

def compute_accuracy(opt, test_features, test_labels, attributes):
    """

    Args:
        test_features: 
        test_labels: 
        attributes: 

    Returns:
    
    """
    # Get Unique Form of Data
    ## unique test labels select
    test_labels_np = test_labels.numpy()
    unique_test_labels_np = np.unique(test_labels_np)
    unique_test_labels = torch.LongTensor(unique_test_labels_np)
    if opt.cuda:
        unique_test_labels = unique_test_labels.cuda()
    ## unique test attributes select
    unique_test_attributes = torch.index_select(attributes, 0, unique_test_labels)

    class_num = unique_test_labels.size(0)
    batch_size = 32 

    # Data Preperation
    ## data iterator
    test_data = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Net Initialize
    netAtt = AttributeNet(opt)
    netRN = RelationNet(opt)
    if opt.cuda:
        netAtt = netAtt.cuda()
        netRN = netRN.cuda()

    predict_labels_total = []
    re_batch_labels_total = []

    for batch_features, batch_labels in test_loader:

        # Up-sampling Attributes
        up_attributes_v = netAtt(Variable(unique_test_attributes))

        # Relation Pair Preperation
        ## extend middle attributes
        up_attributes_ext = up_attributes_v.data.unsqueeze(0).repeat(batch_size, 1, 1)
        ## extend batch features
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        ## relation pairs
        relation_pairs = torch.cat((up_attributes_ext, batch_features_ext), 2).view(-1, 4096)

        # Relaton Score
        relations = netRN(Variable(relation_pairs)).view(-1, class_num)

        # Get Predicted Label and Re-projected Label
        ## predicted labels
        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.numpy()
        ## re-build batch_labels according to unique_test_labels
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(unique_test_labels == label)
            re_batch_labels.append(int(index[0][0]))
        re_batch_labels = np.array(re_batch_labels)

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