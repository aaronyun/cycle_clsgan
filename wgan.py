#!/usr/bin/python3.6

from __future__ import print_function
import sys
import os
import math
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as tfunc
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
parser.add_argument('--dataroot', default='/data0/docker/xingyun/f_xGAN/data', help='path to dataset')
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic feqatures')

# dada processing
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)

# network specification
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--r_path', default='/home/xingyun/docker/cycle_clsgan/r_param/', help='path to load parameters of R')
# parser.add_argument('--r_hl', type=int, default=1, help="the number of hidden layers in R net")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--nrh', type=int, default=1024, help='size of the hidden units in R network')
parser.add_argument('--nrh1', type=int, default=1024, help='size of the first hidden units in R network')
parser.add_argument('--nrh2', type=int, default=521, help='size of the second hidden units in R network')
parser.add_argument('--nrh3', type=int, default=312, help='size of the third hidden units in R network')
parser.add_argument('--nrh4', type=int, default=156, help='size of the fourth hidden units in R network')
parser.add_argument('--drop_rate', type=float, default=1, help='the rate of hidden unit to dropout')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--r_weight', type=float, default=1, help='weight of the att generate loss')


# experiment setting
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--r_iteration', type=int, default=3, help='the pretraining time of R net')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--val_every', type=int, default=10) # 只在这里出现了
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

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
    torch.cuda.manual_seed_all(opt.manualSeed)

################################################################################

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

################################################################################

data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

################################################################################

netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

################################################################################

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    noise = noise.cuda()
    input_res, input_att, input_label = input_res.cuda(), input_att.cuda(), input_label.cuda()
    one = one.cuda()
    mone = mone.cuda()

################################################################################

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
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

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1

    return gradient_penalty

################################################################################

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('epoch lossG lossD wDistance acc')
for epoch in range(opt.nepoch):
    max_acc = 0
    max_H = 0
    for i in range(0, data.ntrain, opt.batch_size):
        sample()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        # --------------------------------------------
        # 训练D网络，等式（2）
        # --------------------------------------------
        for p in netD.parameters():
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):

            netD.zero_grad()

            # 用真实数据训练D
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # 用生成的数据训练D
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)

            criticD_fake = netD(fake.detach(), input_attv) # detach(), detached from the current graph
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # 梯度惩罚项
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake

            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        # -------------------------------------------
        # 训练G网络，等式（2）
        # -------------------------------------------
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)

        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)

        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        errG = G_cost
        errG.backward()
        optimizerG.step()

    print('%d %.4f %.4f %.4f' % (epoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0]))

    # evaluate the model, set G to evaluation mode
    netG.eval()

    if opt.gzsl: # GZSL，使用简单的多分类网络来做最后的类别判断
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen_class_acc=%.4f, seen_class_acc=%.4f, h=%.4f' % (cls_.acc_unseen, cls_.acc_seen, cls_.H))
        if cls_.H > max_H:
            max_H = cls_H
    else: # ZSL，使用简单的多分类网络来做最后的类别判断
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc
        print('unseen_class_acc= ', acc)
        if acc > max_acc:
            max_acc = acc

    # reset G to training mode
    netG.train()

if opt.gzsl:
    print('max_H:', max_H)
else:
    print('max_acc:', max_acc)
