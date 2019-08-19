#!/usr/bin/python3.7

import sys
import random

import torch
import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

def weights_init(m):
    """Initialize weight for neural network.

    Args:

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    """
    """
    #? 为什么需要进行map_label()
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        # (label==classes[i]) 返回一个tensor，label里凡是与classes[i]相等的元素所在位置为1，其他的为0
        mapped_label[label==classes[i]] = i

    return mapped_label

class Img_Preprocess(object):
    def __init__(self):
        pass

    def padding(self, pad):
        """
        """
        def f(image):
            return np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant')

        return f

    def random_crop(self, size):
        """
        """
        def f(image):
            _, h, w = image.shape
            p = random.randint(0, h-size)
            q = random.randint(0, w-size)

            return image[:, p:p+size, q:q+size]

        return f

    def horizontal_flip(self):
        """
        """
        def f(image):
            if random.randint(0, 1): # 对大部分image进行翻转
                image = image[:, :, ::-1]
            return image

        return f

    def zero_mean(self, mean, std):
        """
        """
        def f(image):
            image_mean = np.mean(image, keepdims=True)
            return (image - image_mean - mean[:, None, None]) / std[:, None, None]

        return f

class Triple_Selector(object):
    """
    """
    def __init__(self, dataset, anchor_index):
        """
        """
        self.base = dataset.train_feature

        self.anchor = self.base.index_select(self.base, 0, anchor_index)
        self.anchor_labels = torch.index_select(dataset.train_labels, 1, anchor_index)

        # exclude anchor from base data
        all_idx = [x for x in range(base.size(0))]
        excluded_anchor_idx = [x for x in index if x not in anchor_index]
        self.base_exclude_anchor = torch.index_select(self.base, 0, excluded_anchor_idx)
        self.base_exclude_anchor_labels = torch.index_select(dataset.train_labels, 1, excluded_anchor_idx)

        self.hardest_triple = construct_hardest_triple(self.anchor, self.base_exclude_anchor)


    def construct_hardest_triple(self, anchor, base_exclude_anchor):
        """Construct hardest triple data based on given anchor and base dataset.

        Anchors are selected from base dataset.

        Args:
            anchor: anchor visual feature to construct hardest triple.
            base_data: data base used to select positive and negative visual features.

        Returns:
            hardest_triple_data: triples constructed from anchor.

        Raises:

        """
        # Firstly get distance between visual features
        pairwise_dist = pairwise_distances(anchor, base_exclude_anchor)

        # Secondly get a mask which can hide values have different classes
        pos_mask = get_positive_mask()
        anchor_pos_dist = pos_mask * pairwise_dist # distance between anchors and positives
        _, max_idx = torch.max(anchor_pos_dist, 1)

        anchor_hardest_pos = torch.index_select(self.base, 0, max_idx)

        # Same as positives
        neg_mask = get_negative_mask()
        anchor_neg_dist = neg_mask * pairwise_dist
        _, min_idx = torch.min(anchor_neg_dist, 1)

        anchor_hardest_neg = torch.index_select(self.base, 0, min_idx)

        # concatenate anchors, positives and negatives
        hardest_triple_data = (anchor, anchor_hardest_pos, anchor_hardest_neg)

        return hardest_triple_data

    def pairwise_distances(self):
        """Compute distance between anchor and base.

            1. Anchors are selected from base, and anchors must be excluded from base before computation. 
            2. The result distance matrix contains elements, all of which are distances between different visual features.

        Args:
            anchor: anchor data
            base_exclude_anchor: base data which does not contain anchor data.

        Returns:
            distances: squared Euclidian distance between anchor and base

        Raises:

        """
        # 1. 对anchor和base各自做自身的点积，然后取出各自的对角线，得到的就是各自向量的平方，然后anchor按列扩展，base按行扩展，得到形状和第2步一样的矩阵
        # 2. anchor和base做点积
        # 3. 将扩展后的矩阵减去2倍的第2步矩阵，即为anchor和base中除自身以外的向量的欧式距离

        # Firstly get dot products
        anchor_dot = torch.matmul(anchor, anchor.t())
        base_exclude_anchor_dot = torch.matmul(base_exclude_anchor, base_exclude_anchor.t())
        dot_product = torch.matmul(anchor, base_exclude_anchor.t())

        anchor_diag = torch.diag(anchor_dot)
        base_exclude_anchor_diag = torch.diag(base_exclude_anchor_dot)

        distances_matrix = anchor_diag.unsqueeze(1) - 2.0 * dot_product + base_exclude_anchor_diag.unsqueeze(0)

        return distances_matrix

    def get_positive_mask(self):
        """
        """
        positive_mask = self.anchor_labels.unsqueeze(1) == self.base_exclude_anchor_labels.unsqueeze(0)

        return positive_mask

    def get_negative_mask():
        """
        """
        negtive_mask = ~(self.anchor_labels.unsqueeze(1) == self.base_exclude_anchor_labels.unsqueeze(0))

        return negative_mask

    def next_batch(self, triple_type='hardest'):
        """Get a batch of mixing triples.

        Args:
            triple_batch_size: size of batch.

        Returns:
            triple_batch: a batch of triples.
        """
        
        if triple_type == 'hardest':
            anchor, pos, neg = self.hardest_triple
            triple_data = torch.cat((anchor, pos, neg), 0)
            # random permutation
            rand_idx = torch.randperm(triple_data.size(0))[0:triple_batch_size]

            triple_batch = self.triple_data[rand_idx]
        else:
            raise('Triple data type should be hardest')

        return triple_batch

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()] 
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        """Read data of .mat suffix.

        Args:

        Returns:

        """
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    
        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        if not opt.validation: # no validation
            if opt.preprocessing: # preprocessing
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                # TRAIN
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.train_att = torch.index_select(self.attribute, 0, self.train_label)

                # TEST unseen
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_unseen_att = torch.index_select(self.attribute, 0, self.test_unseen_label)

                # TEST seen
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                self.test_seen_att = torch.index_select(self.attribute, 0, self.test_seen_label)

            else: # no preprocessing
                # TRAIN
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.train_att = torch.index_select(self.attribute, 0, self.train_label)
                # TEST unseen
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_unseen_att = torch.index_select(self.attribute, 0, self.test_unseen_label)
                # TEST seen
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                self.test_seen_att = torch.index_select(self.attribute, 0, self.test_seen_label)
        else: # validation
            # TRAIN
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.train_att = torch.index_select(self.attribute, 0, self.train_label)
            # TEST unseen
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
            self.test_unseen_att = torch.index_select(self.attribute, 0, self.test_unseen_label)

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = self.seenclasses.clone()

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_batch_one_class(self, batch_size):
        """
        """
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        """Get a batch of training data randomly.

        Args:
            batch_size: size of batch

        Returns:
            A tuple contains a batch of features, labels and corresponding attributes.
        """
        idx = torch.randperm(self.ntrain)[0:batch_size] # get a permutation

        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        return batch_feature, batch_label, batch_att, idx

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        """
        """
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att