import random

import torch
from torch.autograd import Variable
import torch.autograd as autograd
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

def sample(opt, data):
    batch_feature, batch_label, batch_att, batch_index = data.next_batch(opt.batch_size)

    batch_vf = torch.FloatTensor(batch_feature)
    batch_att = torch.FloatTensor(batch_att)
    batch_label = torch.LongTensor(batch_label)
    batch_index = torch.LongTensor(batch_index)

    if opt.cuda:
        batch_vf = batch_vf.cuda()
        batch_label = batch_label.cuda()
        batch_att = batch_att.cuda()
        batch_index = batch_index.cuda()

    return batch_vf, batch_label, batch_att, batch_index

def calc_gradient_penalty(opt, netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1

    return gradient_penalty

def generate_syn_feature(opt, netG, classes, all_attribute, num_perclass):
    """
    """
    nclass = classes.size(0)

    syn_feature = torch.FloatTensor(nclass*num_perclass, opt.res_size)
    syn_label = torch.LongTensor(nclass*num_perclass) 
    syn_att = torch.FloatTensor(num_perclass, opt.att_size)
    syn_noise = torch.FloatTensor(num_perclass, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = all_attribute[iclass]

        syn_att.copy_(iclass_att.repeat(num_perclass, 1))
        syn_noise.normal_(0, 1)

        output = netG(Variable(syn_noise, requires_grad=False), Variable(syn_att, requires_grad=False))
        syn_feature.narrow(0, i*num_perclass, num_perclass).copy_(output.data.cpu())
        syn_label.narrow(0, i*num_perclass, num_perclass).fill_(iclass)

    return syn_feature, syn_label

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

class Triplet_Selector(object):
    """Construct various triplet data based on given anchor and train data.

    Class information...

    Attributes:

    """
    def __init__(self, opt, dataset, anchor, anchor_label, anchor_index, triplet_type='hard'):
        """
        """
        self.cuda = opt.cuda
        # Initialize
        self.triplet_type = triplet_type

        self.base = dataset.train_feature
        self.base_label = dataset.train_label
        self.base_index = torch.LongTensor([x for x in range(self.base.size(0))])# list

        self.anchor = anchor
        self.anchor_label = anchor_label
        self.anchor_index = anchor_index

        # Exclude Anchor from Train Data
        ## get index of remainder data
        self.remainder_idx = torch.LongTensor([x for x in self.base_index if x not in self.anchor_index])
        ## select features and lables by indexing
        self.remainder = torch.index_select(self.base, 0, self.remainder_idx)
        self.remainder_label = torch.index_select(self.base_label, 0, self.remainder_idx)

        if self.cuda:
            self.base, self.base_label, self.base_index = self.base.cuda(), self.base_label.cuda(), self.base_index.cuda()

            self.remainder, self.remainder_label, self.remainder_idx = self.remainder.cuda(), self.remainder_label.cuda(), self.remainder_idx.cuda()

        # Get Triplet Data Based on Triplet Type
        if self.triplet_type == 'hard_both':
            self.triplet_data = self.construct_hard_both_triplet(anchor, self.remainder)
        elif self.triplet_type == 'semi_hard':
            self.triplet_data = self.construct_semi_hard_triplet(anchor, self.remainder)
        elif self.triplet_type == 'hard_pos':
            self.triplet_data = self.construct_hard_pos_triplet(anchor, self.remainder)
        elif self.triplet_type == 'hard_neg':
            self.triplet_data = self.construct_hard_neg_triplet(anchor, self.remainder)
        elif self.triplet_type == 'easy':
            self.triplet_data = self.construct_easy_triplet(anchor, self.remainder)
        else:
            raise('Invalid triplet data type!')

    def construct_hard_both_triplet(self, anchor, remainder):
        """Construct hard triplet based on given anchors and remainder.

        Anchors are selected from base data.

        Args:
            anchor: anchor visual feature to construct hard triplet
            remainder: used to select positive and negative visual features

        Returns:
            hard_triplet: triplets constructed from anchor.

        Raises:

        """
        # Distance Between Anchor and Remainder
        pairwise_dist = self.pairwise_distances(anchor, remainder)

        # Positive Data
        ## positive mask to hide values which blong to different class
        pos_mask = self.get_positive_mask()
        ## distance between anchors and positives
        anchor_pos_dist = torch.mul(pos_mask, pairwise_dist)
        ## index of maximum
        _, max_idx = torch.max(anchor_pos_dist, 1)
        ## final hardest positive element
        hard_pos = torch.index_select(remainder, 0, max_idx)

        # Negative Data
        ## negative mask
        neg_mask = self.get_negative_mask()
        ## distance
        anchor_neg_dist = torch.mul(neg_mask, pairwise_dist)
        ## index of minimum
        _, min_idx = torch.min(anchor_neg_dist, 1)
        ## hardest negative element
        hard_neg = torch.index_select(remainder, 0, min_idx)

        hard_triplet = (anchor, hard_pos, hard_neg)

        return hard_triplet

    def construct_hard_pos_triplet(self, anchor, remainder):
        """Construct hard triplet (only hard positive, negative are selected randomly) based on given anchor and remainder.

        Anchors are selected from base data. Only hardest positives are calculated, corresponding negatives are randomly select (three times) based on harsedt positves.

        Args:
            anchor: anchor visual feature to construct hard triplet
            remainder: used to select positive and negative visual features

        Returns:
            hard_pos_triple: triples constructed from anchor.

        Raises:

        """
        # Distance Between Anchor and Remainder
        pairwise_dist = self.pairwise_distances(anchor, remainder)

        # Positive Data
        ## positive mask to hide values that blong to different class
        pos_mask = self.get_positive_mask()
        ## distance between anchors and positives
        anchor_pos_dist = torch.mul(pos_mask, pairwise_dist)
        ## index of maximum
        _, max_idx = torch.max(anchor_pos_dist, 1)
        ## final hardest positive element
        hard_pos = torch.index_select(remainder, 0, max_idx)

        # Negative Data
        ## negative mask
        neg_mask = self.get_negative_mask()

        ## get index of negatives in remainder
        rand_idx = self.get_rand_index(neg_mask)
        ## negative select
        batch_neg_1 = torch.index_select(remainder, 0, rand_idx)
        ## get index of negatives in remainder
        rand_idx = self.get_rand_index(neg_mask)
        ## negative select
        batch_neg_2 = torch.index_select(remainder, 0, rand_idx)
        ## get index of negatives in remainder
        rand_idx = self.get_rand_index(neg_mask)
        ## negative select
        batch_neg_3 = torch.index_select(remainder, 0, rand_idx)

        neg = torch.cat((batch_neg_1, batch_neg_2, batch_neg_3), 0)

        hard_pos_triplet = (anchor.repeat(3,1), hard_pos.repeat(3,1), neg)

        return hard_pos_triplet

    def construct_hard_neg_triplet(self, anchor, remainder):
        """Construct hard triplet (only hard negative, positive are selected randomly) based on given anchor and remainder.

        Anchors are selected from base data. Only hardest negatives are calculated, corresponding postives are randomly select (three times) based on harsedt negatives.

        Args:
            anchor: anchor visual feature to construct hard triplet
            remainder: used to select positive and negative visual features

        Returns:
            hard_pos_triple: triples constructed from anchor.

        Raises:

        """
        # Distance Between Anchor and Remainder
        pairwise_dist = self.pairwise_distances(anchor, remainder)

        # Negative Data
        ## negative mask
        neg_mask = self.get_negative_mask()
        ## distance
        anchor_neg_dist = torch.mul(neg_mask, pairwise_dist)
        ## index of minimum
        _, min_idx = torch.min(anchor_neg_dist, 1)
        ## hardest negative element
        hard_neg = torch.index_select(remainder, 0, min_idx)

        # Positive Data
        ## positive mask to hide values that blong to different class
        pos_mask = self.get_positive_mask()
        
        ## get index of positives in remainder
        rand_idx = self.get_rand_index(pos_mask)
        ## positives select
        batch_pos_1 = torch.index_select(remainder, 0, rand_idx)
        ## get index of positives in remainder
        rand_idx = self.get_rand_index(pos_mask)
        ## positives select
        batch_pos_2 = torch.index_select(remainder, 0, rand_idx)
        ## get index of positives in remainder
        rand_idx = self.get_rand_index(pos_mask)
        ## positives select
        batch_pos_3 = torch.index_select(remainder, 0, rand_idx)

        pos = torch.cat((batch_pos_1, batch_pos_2, batch_pos_3), 0)

        hard_neg_triplet = (anchor.repeat(3,1), pos, hard_neg.repeat(3,1))

        return hard_neg_triplet

    def construct_semi_hard_triplet(self, anchor, remainder):
        """TBD
        """
        # Distance Between Anchor and Remainder
        pairwise_dist = self.pairwise_distances(anchor, remainder)

        return anchor, anchor_semi_hard_pos, anchor_semi_hard_neg

    def construct_easy_triplet(self, anchor, remainder):
        """TBD
        """
        # Distance Between Anchor and Remainder
        pairwise_dist = self.pairwise_distances(anchor, remainder)

        return anchor, anchor_easy_pos, anchor_easy_neg

    def next_batch(self, batch_size):
        """Get a batch of triplets.

        Args:
            batch_size: number of triplets in a batch

        Returns:
            triplet_batch: a batch of triplets.
        """
        anchor, pos, neg = self.triplet_data

        # Randomly Select a Batch of Triplet Data
        rand_idx = (torch.randperm(anchor.size(0))[0:batch_size])
        if self.cuda:
            rand_idx = rand_idx.cuda()
        batch_anchor = anchor[rand_idx]
        batch_pos = pos[rand_idx]
        batch_neg = neg[rand_idx]

        triple_batch = torch.cat((batch_anchor, batch_pos, batch_neg), 0)

        return triple_batch

    def pairwise_distances(self, anchor, remainder):
        """Compute distance between anchor and train data.

            1. Anchors are selected from train data, and anchors must be excluded from train data before computation. 
            2. The result distance matrix contains elements all of which are distances between different visual features.

        Args:
            anchor: anchor data
            remainder: train data which does not contain anchor data.

        Returns:
            distances: squared Euclidian distance between anchor and base

        Raises:

        """
        # Dot Products
        anchor_dot = torch.matmul(anchor, anchor.t())
        remainder_dot = torch.matmul(remainder, remainder.t())
        dot_product = torch.matmul(anchor, remainder.t())

        # Diagnal Elements
        anchor_diag = torch.diag(anchor_dot)
        remainder_diag = torch.diag(remainder_dot)

        # Distance Between Different Elements
        distances_matrix = anchor_diag.unsqueeze(1) - 2.0 * dot_product + remainder_diag.unsqueeze(0)

        return distances_matrix

    def get_positive_mask(self):
        """
        """
        positive_mask = self.anchor_label.unsqueeze(1) == self.remainder_label.unsqueeze(0)
        positive_mask = positive_mask.type(torch.FloatTensor)

        if self.cuda:
            positive_mask = positive_mask.cuda()

        return positive_mask

    def get_negative_mask(self):
        """
        """
        negative_mask = ~(self.anchor_label.unsqueeze(1) == self.remainder_label.unsqueeze(0))
        negative_mask = negative_mask.type(torch.FloatTensor)

        if self.cuda:
            negative_mask = negative_mask.cuda()

        return negative_mask

    def get_rand_index(self, mask):
        """
        """
        idx = []
        for i in range(mask.size(0)):
            breaker = True
            while breaker:
                rand = random.randint(0, mask.size(1)-1)
                if self.triplet_type == 'hard_pos':
                    if mask[i][rand] == 1:
                        idx.append(rand)
                        breaker = False
                elif self.triplet_type == 'hard_neg':
                    if mask[i][rand] == 1:
                        idx.append(rand)
                        breaker = False
                else:
                    raise('Call Error!')

        idx = torch.LongTensor(idx)
        if self.cuda:
            idx = torch.LongTensor(idx).cuda()

        return idx

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
        """Read data in matlab file.

        Args:
            opt:

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