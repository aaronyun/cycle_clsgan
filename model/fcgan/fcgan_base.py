# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# reverse + fusion + end-to-end(class with hidden features)
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

import numpy as np
from sklearn.manifold import TSNE

sys.path.append('/data0/docker/xingyun/projects/mmcgan')

from util import opts
from util import tools
from util import mlp
from util.eval import classifier, classifier2

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
if opt.r_hl == 1:
    netR = mlp.MLP_1HL_Dropout_R(opt)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_R(opt)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_R(opt)
elif opt.r_hl == 4:
    netR = mlp.MLP_4HL_Dropout_R(opt)
else:
    raise('Initialize Error of Reverse Net')
if opt.netR !='':
    netR.load_state_dict(torch.load(opt.netR))
print(netR)

# Fusion Net Initialize
netF = mlp.FusionNet(opt)
if opt.netF != '':
    netF.load_state_dict(torch.load(opt.netF))
print(netF)

#------------------------------------------------------------------------------#

# Setup Optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Loss Function
cos_criterion = nn.CosineSimilarity()
euc_criterion = nn.PairwiseDistance(p=2)

#------------------------------------------------------------------------------#

noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

# if opt.cuda:
#     netD.cuda()
#     netG.cuda()
#     netR.cuda()

#     noise = noise.cuda()

#     cos_criterion = cos_criterion.cuda()
#     euc_criterion = euc_criterion.cuda()

#     one = one.cuda()
#     mone = mone.cuda()

#------------------------------------------------------------------------------#

# Store Best Results
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  F_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |  ACC_seen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  F_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |')

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
            batch_vf, batch_label, batch_att, batch_index = tools.sample(opt, data)

            # Train D with Real Data
            d_real_v = netD(Variable(batch_vf), Variable(batch_att))
            d_real_v = d_real_v.mean()
            d_real_v.backward(mone)

            # Train D with Fake Data
            ## generate fake data
            noise.normal_(0, 1)
            gen_vf_v = netG(Variable(noise), Variable(batch_att))
            ## training
            d_fake_v = netD(gen_vf_v.detach(), Variable(batch_att))
            d_fake_v = d_fake_v.mean()
            d_fake_v.backward(one)

            # Gradient Penalty
            gradient_penalty_v = tools.calc_gradient_penalty(opt, netD, batch_vf, gen_vf_v.data, Variable(batch_att))
            gradient_penalty_v.backward()

            # Wasserstein Distance
            Wasserstein_D = d_real_v - d_fake_v

            # Overall Discriminator Loss
            D_cost = d_fake_v - d_real_v + gradient_penalty_v

            # Update Parameters
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False

#------------------------------------------------------------------------------#

        ############################
        # GENERATOR TRAINING
        ###########################
        netG.zero_grad()

        # Data Sampling
        batch_vf, batch_label, batch_att, batch_index = tools.sample(opt, data)

        # Train Generator with Discriminator
        ## generate fake data
        noise.normal_(0, 1)
        gen_train_vf_v = netG(Variable(noise), Variable(batch_att))
        ## training
        g_fake = netD(gen_train_vf_v, Variable(batch_att))
        g_fake = g_fake.mean()

        # Generator Loss
        G_cost = -g_fake # Decrease

#------------------------------------------------------------------------------#

        ################################
        # FUSION TRAINING
        ################################
        # Data Preparation
        train_vf = batch_vf
        gen_vf = gen_train_vf_v.data

        # Fusion Net Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad() 
            for batch_vf, batch_label in train_loader:
                batch_vf_v = Variable(batch_vf)
                batch_label_v = Variable(batch_label)

                ## training
                fusion_vf = netF(batch_vf_v)

                ## loss
                # 1. 分类损失
                # 2. 聚合损失

#------------------------------------------------------------------------------#

        ################################
        # REVERSE NET TRAINING
        ################################
        netR.zero_grad()

        # Train Reverse Net with Generated Visual Feature
        ## r training
        syn_att_v = netR(gen_train_vf_v)
        ## attribute consistency loss
        R_cost = cos_criterion(syn_att_v, Variable(batch_att))
        R_cost = R_cost.mean
        # update r net
        R_cost.backward(mone, retain_graph=True)
        optimizerR.step()

#------------------------------------------------------------------------------#

        # FINAL GENERATOR LOSS
        errG = G_cost - opt.r_weight * R_cost
        errG.backward()
        optimizerG.step()


#------------------------------------------------------------------------------#

        ################################
        # CLASSIFICATION
        ################################

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = tools.generate_syn_feature(opt, netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H
            corresponding_epoch = epoch

    else:
        syn_feature, syn_label = tools.generate_syn_feature(opt, netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, tools.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

    netG.train()

#------------------------------------------------------------------------------#

# Output Best Results
if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

#------------------------------------------------------------------------------#