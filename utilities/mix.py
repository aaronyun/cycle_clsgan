#!/usr/bin/python3.6

"""
Mixing DATASETs based on existing functions.
"""

import random
import numpy as np
import torch

from utilities import util

mix_rseed = random.randint(1, 10000)
random.seed(mix_rseed)

class DataMixer(object):
    def __init__(self, data, opt):
        self.opt = opt

        self.train_feature = (data.train_feature).numpy()
        self.train_label = (data.train_label).numpy()
        self.att = (data.attribute).numpy()

        # we dont need test data to be mixed
        # self.test_unseen_feature = (data.test_unseen_feature).numpy()
        # self.test_unseen_att = (data.test_unseen_att).numpy()
        # self.test_unseen_label = (data.test_unseen_label).numpy()
        # self.test_unseen_feature = (data.test_unseen_feature).numpy()
        # self.test_unseen_att = (data.test_unseen_att).numpy()
        # self.test_unseen_label = (data.test_unseen_label).numpy()

    def mix(self):
        while True:
            loc1 = random.randint(0, self.train_label.shape[0]-1)
            feature1 = self.train_feature[loc1]
            label1 = self.train_label[loc1]
            att1 = self.att[label1]

            loc2 = random.randint(0, self.train_label.shape[0]-1)
            feature2 = self.train_feature[loc2]
            label2 = self.train_label[loc2]
            att2 = self.att[label2]

            if label1 != label2:
                break

        mean1 = np.mean(feature1)
        mean2 = np.mean(feature2)

        # mix images
        r = np.array(random.random()) # r~U(0,1)
        g1 = np.std(feature1)
        g2 = np.std(feature2)
        p = 1.0 / (1 + g1 / g2 * (1 - r) / r)

        # （公式中）feature1 = （论文中）image - mean( of image)
        image = (((feature1 - mean1) * p + (feature2 - mean2) * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2)).astype(np.float32)

        # mix attributes and labels
        att = att1*r + att2*(1-r)
        label = label1*r + label2*(1-r)

        return image, label, att

    def get_mixing_batch(self):
        mixing_feature_batch = []
        mixing_label_batch = []
        mixing_att_batch = []

        #TODO bc是能实现data augmentation的，用起来
        for i in range(self.opt.batch_size):
            feature, label, att = self.mix()
            mixing_feature_batch.append(feature)
            mixing_label_batch.append(label)
            mixing_att_batch.append(att)

        mixing_feature_batch = torch.from_numpy(np.asarray(mixing_feature_batch))
        mixing_label_batch = torch.from_numpy(np.asarray(mixing_label_batch))
        mixing_att_batch = torch.from_numpy(np.asarray(mixing_att_batch))

        return mixing_feature_batch, mixing_label_batch, mixing_att_batch