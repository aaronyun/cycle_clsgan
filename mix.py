#!/usr/bin/python3.6

"""
Mixing DATASETs based on existing functions.
"""

import random
import numpy as np
import torch

import util

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
            image1 = self.train_feature[loc1]
            label1 = self.train_label[loc1]
            att1 = self.att[label1]

            loc2 = random.randint(0, self.train_label.shape[0]-1)
            image2 = self.train_feature[loc2]
            label2 = self.train_label[loc2]
            att2 = self.att[label2]

            if label1 != label2:
                break
        #! 没有像BC论文中那样进行图像的处理

        # 混合图片
        r = random.random() # r直接设为一个0~1之间的float
        if self.opt.bc:
            g1 = np.std(image1)
            g2 = np.std(image2)
            p = 1.0 / (1 + g1 / g2 * (1 - r) / r)
            image = ((image1 * p + image2 * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2)).astype(np.float32)
        else:
            image = (image1 * r + image2 * (1 - r)).astype(np.float32)

        # 混合属性和标签
        att = att1*r + att2*(1-r)
        label = label1*r + label2*(1-r)

        return image, label, att

    def get_mixing_batch(self):
        mixing_img_batch = []
        mixing_label_batch = []
        mixing_att_batch = []

        for i in range(self.opt.batch_size):
            img, label, att = self.mix()
            mixing_img_batch.append(img)
            mixing_label_batch.append(label)
            mixing_att_batch.append(att)

        mixing_img_batch = torch.from_numpy(np.asarray(mixing_img_batch))
        mixing_label_batch = torch.from_numpy(np.asarray(mixing_label_batch))
        mixing_att_batch = torch.from_numpy(np.asarray(mixing_att_batch))

        return mixing_img_batch, mixing_label_batch, mixing_att_batch