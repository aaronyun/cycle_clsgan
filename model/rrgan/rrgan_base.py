# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# 采用rob-gan的实现
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

import numpy as np
from sklearn.manifold import TSNE

sys.path.append('/data0/docker/xingyun/projects/mmcgan')

from util import opts
from util import tools
from util import mlp
from util.eval import classifier, classifier2
from util.loss import loss_nll
from util.pgd import attack_Linf_PGD

#------------------------------------------------------------------------------#

opt = opts.parse()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#------------------------------------------------------------------------------#

# initialize internal state
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)

# sets the seed for generating random numbers
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

#------------------------------------------------------------------------------#

# running environment setting
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#------------------------------------------------------------------------------#

# load datasets
data = tools.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

#------------------------------------------------------------------------------#

# Generator Initialize
netG = mlp.Gen(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Discriminator Initialize
netD = mlp.robDis(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Reverse Net Initialize
if opt.r_hl == 1:
    netR = mlp.MLP_1HL_Dropout_R(opt)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_R(opt)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_R(opt)
elif opt.r_hl == 4:
    netR = mlp.MLP_4HL_Dropout_R(opt)
else:
    raise('Initialize Error of R')
print(netR)

#------------------------------------------------------------------------------#

# Setup Optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Loss Function
cos_criterion = nn.CosineSimilarity()
euc_criterion = nn.PairwiseDistance(p=2)

#------------------------------------------------------------------------------#

zeros_v = Variable(torch.FloatTensor(opt.batch_size).fill_(0))
ones_v = Variable(torch.FloatTensor(opt.batch_size).fill_(1))

noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()

    cos_criterion = cos_criterion.cuda()
    euc_criterion = euc_criterion.cuda()

    noise = noise.cuda()
    zeros_v = zeros_v.cuda()
    ones_v = ones_v.cuda()
    one = one.cuda()
    mone = mone.cuda()

#------------------------------------------------------------------------------#

# Store Best Results
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  ACC_unseen  |  ACC_seen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  ACC_unseen  |')

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

            # Train D with Adversarial Data
            ## get adversarial sample
            adv_vf_v = attack_Linf_PGD(input_vf_v, ones_v, input_att_v, input_label_v, netD, loss_nll, opt.adv_steps, opt.epsilon)
            ## training
            d_adv_bin_v, d_adv_multi_v = netD(adv_vf_v, input_att_v)
            ## loss for adversarial data
            loss_d_adv = loss_nll(d_adv_bin_v, ones_v, d_adv_multi_v, input_label_v, lam=0.5)

            # Train D with Fake Data
            ## generate fake data
            noise.normal_(0, 1)
            noise_v = Variable(noise)
            fake_vf_v = netG(noise_v, input_att_v)
            ## training
            d_fake_bin_v, d_fake_multi_v = netD(fake_vf_v.detach(), input_att_v)
            ## loss for fake data
            loss_d_fake = loss_nll(d_fake_bin_v, zeros_v, d_fake_multi_v, input_label_v, lam=1)

            # Overall Discriminator Loss
            D_cost = loss_d_adv + loss_d_fake
            D_cost.backward()
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
        gen_vf_v = netG(noise_v, input_att_v)
        ## training
        g_fake_bin_v, g_fake_multi_v = netD(gen_vf_v, input_att_v)
        # loss
        loss_g = loss_nll(g_fake_bin_v, ones_v, g_fake_multi_v, input_label_v, lam=0.5)

        # Generator Loss
        G_cost = loss_g

#------------------------------------------------------------------------------#

        # for visualization
        train_vf = input_vf_v.data
        gen_vf = gen_vf_v.data
        label = input_label_v.data

#------------------------------------------------------------------------------#

        ################################
        # R TRAINING
        ################################
        netR.zero_grad()

        # Train Reverse Net with Generated Visual Feature
        ## r training
        syn_att_v = netR(gen_vf_v)
        ## attribute consistency loss
        R_cost = cos_criterion(syn_att_v, input_att_v)
        R_cost = R_cost.mean() # R_cost should increase

        R_cost.backward(mone, retain_graph=True)
        optimizerR.step()

#------------------------------------------------------------------------------#

        # FINAL GENERATOR LOSS
        errG = G_cost - opt.r_weight * R_cost
        errG.backward()
        optimizerG.step()

#------------------------------------------------------------------------------#

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], R_cost.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # label of features
            tsne_label = (label.cpu()).numpy()

    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, tools.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], R_cost.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # label of features
            tsne_label = (label.cpu()).numpy()

    netG.train()

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

# save visualization data
exp_set = '/gzsl'
model = '/rrgan'
exp_type = '/base/'

root = '/data0/xingyun/docker/mmcgan_torch030/fig' + exp_set + model + exp_type + opt.dataset

np.save(file=root+'/label', arr=tsne_label) # 数据的标签

np.save(file=root+'/vf_train_embed', arr=vf_train_embed) # 训练集的vf
np.save(file=root+'/vf_gen_embed', arr=vf_gen_embed) # 生成的vf