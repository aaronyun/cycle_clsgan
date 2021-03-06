# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# 保持reverse网络不变，采用更加简单的fuse方法，并且对hidden features直接进行分类
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

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()

    noise = noise.cuda()

    cos_criterion = cos_criterion.cuda()
    euc_criterion = euc_criterion.cuda()

    one = one.cuda()
    mone = mone.cuda()

#------------------------------------------------------------------------------#

# Store Best Results
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |  ACC_seen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |')

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
        gen_train_vf_v = netG(noise_v, input_att_v)
        ## training
        g_fake = netD(gen_train_vf_v, input_att_v)
        g_fake = g_fake.mean()

        # Generator Loss
        G_cost = -g_fake # Decrease

#------------------------------------------------------------------------------#

        ################################
        # REVERSE NET TRAINING
        ################################
        netR.zero_grad()

        # Train Reverse Net with Generated Visual Feature
        ## r training
        syn_att_v = netR(gen_train_vf_v)
        ## attribute consistency loss
        R_cost = cos_criterion(syn_att_v, input_att_v)
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
        # FUSION TRAINING
        ################################

        # Data Preparation
        train_vf = torch.cat((input_vf_v.data, g_gen_vf_v.data), 0)
        train_label = input_label_v.data.repeat(1,2)

        train_data = TensorDataset(train_vf, train_label)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Fusion Net Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad()
            for batch_vf, batch_label in train_loader:
                batch_vf_v = Variable(batch_vf)
                batch_label_v = Variable(batch_label)

                ## training
                fusion_vf = netF(batch_vf_v)

                ## loss


#------------------------------------------------------------------------------#

    netG.eval()

    # Classification
    # 端到端的意思是，不单独用生成的数据再去训练一个分类器，而是在每一轮训练过程中，将特征喂进分类网络直接得到分类结果，而没有训练的过程

    netG.train()

#------------------------------------------------------------------------------#

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))