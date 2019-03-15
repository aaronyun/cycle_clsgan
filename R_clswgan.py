#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import math
import argparse
import random

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import util
import classifier
import classifier2
import model

################################################################################

parser = argparse.ArgumentParser()

# data specification
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--dataroot', default='/data0/docker/xingyun/f_xGAN/data', help='path to dataset')
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic feqatures')

# dada processing
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)

# model specification
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')

parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nrh', type=int, default=4096, help='size of the hidden units in R network')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--cycle_weight', type=float, default=0.01, help='weight of the att generate loss')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# experiment setting
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')


parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--outname', help='folder to output data and model checkpoints')

# local using
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')


# unknown
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

opt = parser.parse_args()
print(opt)

################################################################################


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed) # initialize internal state

torch.manual_seed(opt.manualSeed) # sets the seed for generating random numbers
if opt.cuda:
    # sets the seed for generating random numbers on all GPUs
    torch.cuda.manual_seed_all(opt.manualSeed)


cudnn.benchmark = True # it enables benchmark mode in cudnn benchmark mode is good whenever your input sizes for your network do not vary This way, cudnn will look for the optimal set of algorithms for that particular configuration this usually leads to faster runtime

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

################################################################################

# 按dataset参数值加载数据，默认为FLO
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain) # ntrain是特征向量个数

################################################################################

netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netR = model.MLP_R(opt)
# if opt.netR != '':
#     netR.load_state_dict(torch.load(opt.netR))
print(netR)

################################################################################

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss() # Negative Log Likelihood loss
R_criterion = nn.CosineSimilarity(dim=1)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

################################################################################

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    # self.copy_(src)将src复制到self里
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, requires_grad=False), Variable(syn_att, requires_grad=False))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    
    # torch.rand()默认从正太分布取值
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    #!用插值作为输入计算判别网络的梯度
    #!grad()函数？具体的梯度数据？
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1

    return gradient_penalty

################################################################################

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):

    # FP = 0 #?干什么用的

    # mean_lossD = 0
    # mean_lossG = 0

    for i in range(0, data.ntrain, opt.batch_size):

        #!###########################
        #! (1) Update D network: optimize WGAN-GP objective, Equation (2)
        #!##########################p
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):

            sample()

            netD.zero_grad() # 设置梯度为0，重新开始一次计算

            # 用真实数据训练D
            # sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # 用生成的数据训练D
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            # fake_norm = fake.data[0].norm() #? fake_norm没有使用过
            # sparse_fake = fake.data[0].eq(0).sum() #? sparse_fake没有使用过

            criticD_fake = netD(fake.detach(), input_attv) # detach(), detached from the current graph
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # 梯度惩罚
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty #?为什么取反？
            optimizerD.step() # 进行一次参数更新

        #!###########################
        #! (2) Update G network: optimize WGAN-GP objective, Equation (2)
        #!##########################
        for p in netD.parameters():
            p.requires_grad = False # 避免D网络更新

        netG.zero_grad()

        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)

        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        #!###########################
        #! 每进行一个batch的训练，就对R网络进行一次更新
        #!##########################

        netR.zero_grad()

        # syn_features = Variable(fake)
        syn_att = netR(fake)

        R_cost = R_criterion(syn_att, input_attv)

        errR = R_cost.mean()
        errR.backward(retain_graph=True)
        optimizerR.step()

        # 因为R网络存在的目的是提升G网络的能力，所以把损失加到G的zong损失上
        errG = G_cost + opt.cls_weight * c_errG + opt.cycle_weight * errR
        errG.backward()
        optimizerG.step()

    # mean_lossG和mean_lossD计算了，但没有输出过
    # mean_lossD /=  data.ntrain / opt.batch_size
    # mean_lossG /=  data.ntrain / opt.batch_size

    print('[%d/%d] Loss_D: %.4f Loss_R: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f' % (epoch, opt.nepoch, D_cost.data[0], R_cost.data[0], G_cost.data[0], Wasserstein_D.data[0], c_errG.data[0]))

    # evaluate the model, set G to evaluation mode
    netG.eval() #? 具体的作用是什么？

    if opt.gzsl: # GZSL，使用简单的多分类网络来做最后的类别判断
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls_.acc_unseen, cls_.acc_seen, cls_.H))
    else: # ZSL，使用简单的多分类网络来做最后的类别判断
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc
        print('unseen class accuracy= ', acc)

    # reset G to training mode
    netG.train()