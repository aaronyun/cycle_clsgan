# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# 用relation net实现端到端分类
#------------------------------------------------------------------------------#

from __future__ import print_function

import os
import sys
import random

import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np
from sklearn.manifold import TSNE

sys.path.append('/data0/docker/xingyun/projects/mmcgan')

from util import opts
from util import tools
from util import mlp
from util.eval import classifier
from util.eval import classifier2
from util.eval import rn_eval

#------------------------------------------------------------------------------#

opt = opts.parse()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#------------------------------------------------------------------------------#

# Initialize Internal State
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)

# Sets the Seed for Generating Random Numbers
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

#------------------------------------------------------------------------------#

# Running Environment Setting
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#------------------------------------------------------------------------------#

# Load Datasets
data = tools.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

#------------------------------------------------------------------------------#

# Generator Initialize
netG = mlp.Gen(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Discriminator Initialize
netD = mlp.Dis(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Reverse Net Initialize
#! more delicate design needed
if opt.r_hl == 1:
    netR = mlp.MLP_1HL_Dropout_FR(opt)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_FR(opt)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_FR(opt)
elif opt.r_hl == 4:
    netR = mlp.MLP_3HL_Dropout_FR(opt)
else:
    raise('Initialize Error of Reverse Net')
print(netR)

# Fusion Net Initialize
netF = mlp.FusionNet(opt)
print(netF)

# Relation Net Initialize
netAtt = mlp.AttributeNet(opt)
netRN = mlp.RelationNet(opt)

#------------------------------------------------------------------------------#

# Setup Optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizerAtt = optim.Adam(netAtt.parameters(), lr=1e-5, weight_decay=1e-5)
netAtt_scheduler = StepLR(optimizerAtt, step_size=30000, gamma=0.5)
optimizerRN = optim.Adam(netRN.parameters(), lr=1e-5, weight_decay=1e-5)
netRN_scheduler = StepLR(optimizerRN, step_size=30000, gamma=0.5)

# Loss Function
cos_criterion = nn.CosineSimilarity()
euc_criterion = nn.PairwiseDistance(p=2)
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
mse_criterion = nn.MSELoss()

#------------------------------------------------------------------------------#

noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()
    netF.cuda()
    netAtt.cuda()
    netRN.cuda()

    noise = noise.cuda()

    cos_criterion = cos_criterion.cuda()
    euc_criterion = euc_criterion.cuda()
    triplet_criterion = triplet_criterion.cuda()
    mse_criterion = mse_criterion.cuda()

    one = one.cuda()
    mone = mone.cuda()

#------------------------------------------------------------------------------#

# store best result and corresponding epoch
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  F_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |  ACC_seen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  F_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |')

#------------------------------------------------------------------------------#

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):

#------------------------------------------------------------------------------#

        ################################
        # DISCRIMINATOR TRAINING
        ################################
        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(opt.critic_iter):
            netD.zero_grad()

            # Data Sampling
            input_vf, input_label, input_att, input_index = tools.sample(opt, data)
            input_vf_v = Variable(input_vf)
            input_att_v = Variable(input_att)
            input_label_v = Variable(input_label)

            # Train D with Real Data
            d_real = netD(input_vf_v, input_att_v)
            d_real = d_real.mean()
            d_real.backward(mone)

            # Train D with Fake Data
            ## generate fake data
            noise.normal_(0, 1)
            noise_v = Variable(noise)
            d_gen_vf_v = netG(noise_v, input_att_v)
            ## training
            d_fake = netD(d_gen_vf_v.detach(), input_att_v)
            d_fake = d_fake.mean()
            d_fake.backward(one)

            # Gradient Penalty 
            gradient_penalty = tools.calc_gradient_penalty(opt, netD, input_vf_v.data, d_gen_vf_v.data, input_att_v)
            gradient_penalty.backward()

            # Wasserstein Distance
            Wasserstein_D = d_real - d_fake

            # Overall Discriminator Loss
            D_cost = d_fake - d_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False

#------------------------------------------------------------------------------#

        ############################
        # GENERATOR TRAINING
        ###########################
        netG.zero_grad()

        # Data Sampling
        input_vf, input_label, input_att, input_index = tools.sample(opt, data)
        input_vf_v = Variable(input_vf)
        input_att_v = Variable(input_att)
        input_label_v = Variable(input_label)

        # Train Generator with Discriminator
        ## generate fake data
        noise.normal_(0, 1)
        noise_v = Variable(noise)
        g_gen_vf_v = netG(noise_v, input_att_v)
        ## training
        g_fake = netD(g_gen_vf_v, input_att_v)
        g_fake = g_fake.mean()

        # Generator Loss
        G_cost = -g_fake

#------------------------------------------------------------------------------#

        ################################
        # FUSION TRAINING
        ################################

        # Triplet Data
        anchor = g_gen_vf_v.data
        anchor_label = input_label_v.data
        anchor_index = input_index.squeeze()

        triplet = tools.Triplet_Selector(opt, data, anchor, anchor_label, anchor_index, triplet_type='hard_pos')

        # Fusion Net Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad()

            ## get a batch of triples
            triplet_batch = triplet.next_batch(opt.triplet_num)
            triplet_batchv = Variable(triplet_batch)

            ## train F Net with triples
            triplet_hf = netF(triplet_batchv)

            ## triplet loss
            anchor = triplet_hf[0: opt.triplet_num]
            pos = triplet_hf[opt.triplet_num : opt.triplet_num*2]
            neg = triplet_hf[opt.triplet_num*2 : opt.triplet_num*3]

            triplet_loss = triplet_criterion(anchor, pos, neg)
            triplet_loss = triplet_loss.mean()
            triplet_loss.backward(retain_graph=True)

            F_cost = triplet_loss
            optimizerF.step()

        # generate hidden feature after F training
        gen_hf_v = netF(g_gen_vf_v)

#------------------------------------------------------------------------------#

        # for visualization
        train_vf = input_vf_v.data
        gen_vf = g_gen_vf_v.data
        label = input_label_v.data
        gen_hf = gen_hf_v.data

#------------------------------------------------------------------------------#

        ################################
        # REVERSE NET TRAINING
        ################################
        netR.zero_grad()

        # Train Reverse Net with Generated Visual Feature
        ## r training
        syn_att_v = netR(gen_hf_v)
        ## attribute consistency loss
        R_cost = cos_criterion(syn_att_v, input_att_v)
        R_cost = R_cost.mean()
        # update r net
        R_cost.backward(mone, retain_graph=True)
        optimizerR.step()

#------------------------------------------------------------------------------#

        # FINAL GENERATOR LOSS
        errG = G_cost - opt.r_weight * R_cost + F_cost
        errG.backward()
        optimizerG.step()

#------------------------------------------------------------------------------#

        netG.eval()

        ################################
        # RELATION NET TRAINING
        ################################
        # Data Preperation
        ## generate data for unseen class
        syn_feature, syn_label = tools.generate_syn_feature(opt, netG, data.unseenclasses, data.attribute, opt.syn_num)
        ## overall training data
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        ## data iterator
        train_data = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True)

        for eposide in range(opt.rn_eposides):
            netAtt.zero_grad()
            netRN.zero_grad()

            netAtt_scheduler.step(eposide)
            netRN_scheduler.step(eposide)

            # Get a Batch of Data
            batch_features, batch_labels = train_loader.__iter__().next()
            ## batch_featuers: torch.cuda.FloatTensor
            ## batch_labels: torch.LongTensor

            # Find Unique Labels of Current Batch
            unique_labels = []
            for label in batch_labels:
                if label not in unique_labels:
                    unique_labels.append(label)
            unique_labels = torch.LongTensor(unique_labels)
            ## number of unique classes in this batch
            class_num = unique_labels.size(0)

            # Up-sampling Attributes
            ## find unique attributes
            unique_attributes = torch.index_select(data.attribute, 0, unique_labels)
            ## generate mid-features
            up_attributes_v = netAtt(Variable(unique_attributes).cuda()) # (cla_num, vf_size)

            # Relation Pair Preperation
            ## extend middle features
            up_attributes_ext = up_attributes_v.data.unsqueeze(0).repeat(opt.batch_size, 1, 1) # extend the first dimension
            ## extend batch features
            batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            ## relation pairs
            relation_pairs = torch.cat((up_attributes_ext.cuda(), batch_features_ext.cuda()), 2).view(-1, 4096)

            # Relation Score
            relations = netRN(Variable(relation_pairs)).view(-1, class_num) # (batch, cls_num))

            # One-hot Label
            # re-build batch_labels according to unique_labels
            unique_labels = unique_labels.numpy()
            re_batch_labels = []
            for label in batch_labels.numpy():
                index = np.argwhere(unique_labels == label)
                re_batch_labels.append(int(index[0][0]))
            re_batch_labels = torch.LongTensor(re_batch_labels)

            one_hot_labels_v = Variable(torch.zeros(opt.batch_size, class_num).scatter_(1, re_batch_labels.view(-1, 1), 1))
            if opt.cuda:
                one_hot_labels_v = one_hot_labels_v.cuda()

            # Loss
            loss = mse_criterion(relations, one_hot_labels_v)
            loss.backward()

            optimizerAtt.step()
            optimizerRN.step()

        netG.train()

#------------------------------------------------------------------------------#

    ################################
    # EVALUATION
    ################################
    print('testing begin...')
    netAtt.eval()
    netRN.eval()

    # Test Data
    test_seen_vf = data.test_seen_feature
    test_seen_label = data.test_seen_label
    test_unseen_vf = data.test_unseen_feature
    test_unseen_label = data.test_unseen_label
    all_attribute = data.attribute
    if opt.cuda:
        test_seen_vf = test_seen_vf.cuda()
        test_seen_label = test_seen_label.cuda()
        test_unseen_vf = test_unseen_vf.cuda()
        test_unseen_label = test_unseen_label.cuda()
        all_attribute = all_attribute.cuda()

    if opt.gzsl:
        acc_unseen = rn_eval.compute_accuracy(opt, test_unseen_vf, test_unseen_label, all_attribute)
        acc_seen = rn_eval.compute_accuracy(opt, test_seen_vf, test_seen_label, all_attribute)

        H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc_unseen, acc_seen, H))

        if H > max_H:
            max_H = H
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # embedding for hidden features
            # hf_train_embed = TSNE(n_components=3).fit_transform(train_hfv.data.cpu().numpy())
            hf_gen_embed = TSNE(n_components=3).fit_transform(gen_hf.cpu().numpy())

            # label of features
            tsne_label = (label.cpu()).numpy()

    else:
        acc = compute_accuracy(opt, test_unseen_vf, test_unseen_label, all_attribute)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # embedding for hidden features
            # hf_train_embed = TSNE(n_components=3).fit_transform(train_hf.data.cpu().numpy())
            hf_gen_embed = TSNE(n_components=3).fit_transform(gen_hf.cpu().numpy())

            # label of features
            tsne_label = (label.cpu()).numpy()

    netAtt.train()
    netRN.train()

#------------------------------------------------------------------------------#

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

# save visualization data
exp_set = '/gzsl'
model = '/frwgan'
exp_type = '/e7_rn/'

root = '/data0/docker/xingyun/projects/mmcgan/fig' + exp_set + model + exp_type + opt.dataset

np.save(file=root+'/label', arr=tsne_label) # 数据的标签

np.save(file=root+'/vf_train_embed', arr=vf_train_embed) # 训练集的vf
np.save(file=root+'/vf_gen_embed', arr=vf_gen_embed) # 生成的vf

# np.save(file=root+'/hf_train_embed', arr=hf_train_embed)
np.save(file=root+'/hf_gen_embed', arr=hf_gen_embed) # 生成的hf