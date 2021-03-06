# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
# 采用同时使用train_vf和gen_vf来训练Fusion网络，相应的训练R时也采用两种hf训练
#------------------------------------------------------------------------------#

from __future__ import print_function

import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

sys.path.append('/data0/docker/xingyun/projects/mmcgan')

from util import opts
from util import tools
from util import mlp
from util.eval import classifier
from util.eval import classifier2

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

# running environment setting
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#------------------------------------------------------------------------------#

# load datasets and datamixer
data = tools.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

#------------------------------------------------------------------------------#

# Generator initialize
netG = mlp.Gen(opt.att_size + opt.nz, opt.ngh, opt.res_size)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Discriminator initialize
netD = mlp.Dis(opt.res_size + opt.att_size, opt.ndh)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Reverse net initialize
if opt.r_hl == 1:
    netR = mlp.MLP_1HL_Dropout_FR(opt.hfSize, opt.nrh1, opt.att_size)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_FR(opt.hfSize, opt.nrh1, opt.nrh2, opt.att_size)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_FR(opt.hfSize, opt.nrh1, opt.nrh2, opt.nrh3, opt.att_size)
elif opt.r_hl == 4:
    netR = mlp.MLP_3HL_Dropout_FR(opt.hfSize, opt.nrh1, opt.nrh2, opt.nrh3, opt.nrh4, opt.att_size)
else:
    raise('Initialize Error of R')
print(netR)

# Fusion net initialize
netF = mlp.FusionNet(opt.res_size, 1024, opt.hfSize)
print(netF)

#------------------------------------------------------------------------------#

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# loss function
cos_criterion = nn.CosineSimilarity()
euc_criterion = nn.PairwiseDistance(p=2)
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

#------------------------------------------------------------------------------#

# create input tensor
input_res = torch.FloatTensor(opt.batch_size, opt.res_size)
input_att = torch.FloatTensor(opt.batch_size, opt.att_size)
input_label = torch.LongTensor(opt.batch_size)
input_index = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()
    netF.cuda()

    noise, input_res, input_att, input_label, input_index  = noise.cuda(), input_res.cuda(), input_att.cuda(), input_label.cuda(), input_index.cuda()

    cos_criterion = cos_criterion.cuda()
    euc_criterion = euc_criterion.cuda()
    triplet_criterion = triplet_criterion.cuda()

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

            sample()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            # train D with real data
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # generate fake visual feature
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)

            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_attv)
            gradient_penalty.backward()

            # wasserstein distance
            Wasserstein_D = criticD_real - criticD_fake

            # discriminator loss
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False

#------------------------------------------------------------------------------#

        ############################
        # GENERATOR TRAINING
        ###########################
        netG.zero_grad()

        sample()
        input_vfv = Variable(input_res)
        input_attv = Variable(input_att)

        # generate fake data
        noise.normal_(0, 1)
        gen_vfv = netG(noisev, input_attv)

        # train G with Discriminator
        criticG_fake = netD(gen_vfv, input_attv)
        criticG_fake = criticG_fake.mean()

        G_cost = -criticG_fake

#------------------------------------------------------------------------------#
        ################################
        # FUSION TRAINING
        ################################
        train_vf = input_res
        gen_vf = gen_vfv.data

        # get triple data
        anchor = torch.cat((train_vf, gen_vf), 0)
        anchor_index = (input_index.repeat(1,2)).squeeze()

        triple_data = tools.Triple_Selector(data, anchor, anchor_index)

        # Fusion Net Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad()

            # get a batch of triples
            triple_batch = triple_data.next_batch(opt.batch_size, triple_type='hardest')
            triple_batchv = Variable(triple_batch)

            # train F Net with triples
            triple_hf = netF(triple_batchv)

            # triplet loss
            anchor = triple_hf[0: opt.batch_size]
            pos = triple_hf[opt.batch_size : opt.batch_size*2]
            neg = triple_hf[opt.batch_size*2 : opt.batch_size*3]

            triplet_loss = triplet_criterion(anchor, pos, neg)
            triplet_loss = triplet_loss.mean()
            triplet_loss.backward(retain_graph=True)

            #? 是否需要作一个hf的loss（因为train_vf和gen_vf所对应的类别是相同的，所以在得到hf后，我们也应该期望相同类别的hf相似）
            train_hfv = netF(Variable(train_vf))
            gen_hfv = netF(Variable(gen_vf))

            F_cost = triplet_loss
            optimizerF.step()

#------------------------------------------------------------------------------#

        ################################
        # REVERSE NET TRAINING
        ################################
        netR.zero_grad()

        # R training
        r_train = torch.cat((train_hfv, gen_hfv), 0)
        attv = netR(r_train)

        # attribute consistency loss
        R_cost = cos_criterion(attv, input_attv.repeat(2,1))
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

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # embedding for hidden features
            hf_train_embed = TSNE(n_components=3).fit_transform(train_hfv.data.cpu().numpy())
            hf_gen_embed = TSNE(n_components=3).fit_transform(gen_hfv.data.cpu().numpy())

            # label of features
            tsne_label = (input_label.cpu()).numpy()

    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, tools.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

            # embedding for visual features
            vf_train_embed = TSNE(n_components=3).fit_transform(train_vf.cpu().numpy())
            vf_gen_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())

            # embedding for hidden features
            hf_train_embed = TSNE(n_components=3).fit_transform(train_hf.data.cpu().numpy())
            hf_gen_embed = TSNE(n_components=3).fit_transform(gen_hfv.data.cpu().numpy())

            # label of features
            tsne_label = (input_label.cpu()).numpy()

    netG.train()

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

# save visualization data
exp_set = '/gzsl'
model = '/frwgan'
# exp_type = '/base/'
exp_type = '/e5_v1/'

root = '/data0/docker/xingyun/projects/mmcgan/fig' + exp_set + model + exp_type + opt.dataset

np.save(file=root+'/label', arr=tsne_label) # 数据的标签

np.save(file=root+'/vf_train_embed', arr=vf_train_embed) # 训练集的vf
np.save(file=root+'/vf_gen_embed', arr=vf_gen_embed) # 生成的vf

np.save(file=root+'/hf_train_embed', arr=hf_train_embed)
np.save(file=root+'/hf_gen_embed', arr=hf_gen_embed) # 生成的hf