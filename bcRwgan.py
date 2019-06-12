#!/usr/bin/python3.6

from __future__ import print_function
import sys
import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import util
import opts
import classifier
import classifier2
import model
import mix

opt = opts.parse()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# setting random state
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    print('cuda is on, sets the seed for generating random numbers on all GPU')
    torch.cuda.manual_seed_all(opt.manualSeed)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

# load data and datamixer
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

data_mixer = mix.DataMixer(data, opt)

# network initialize
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

if opt.dataset == 'FLO':
    netR = model.MLP_1HL_Dropout_R(opt)
    # netR.load_state_dict(torch.load(opt.r_path + 'FLO.pth'))
elif opt.dataset == 'CUB1':
    netR = model.MLP_2HL_Dropout_R(opt)
    # netR.load_state_dict(torch.load(opt.r_path + 'CUB1.pth'))
elif opt.dataset == 'SUN1':
    netR = model.MLP_3HL_Dropout_R(opt)
    # netR.load_state_dict(torch.load(opt.r_path + 'SUN1.pth'))
elif opt.dataset == 'AWA1':
    netR = model.MLP_4HL_Dropout_R(opt)
    # netR.load_state_dict(torch.load(opt.r_path + 'AWA1.pth'))
elif opt.dataset == 'APY':
    netR = model.MLP_2HL_Dropout_R(opt)
elif opt.dataset == 'AWA2':
    netR = model.MLP_2HL_Dropout_R(opt)
else:
    print('There is no dataset called %s' % opt.dataset)
print(netR)

# setting R loss
r_criterion = nn.CosineSimilarity()
# r_criterion = nn.PairwiseDistance()

# initialize data container
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

# gpu setting
if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    r_criterion = r_criterion.cuda()
    input_label = input_label.cuda()

# miscellaneous functions
def sample():
    if not opt.bc:
        batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    else:
        batch_feature, batch_label, batch_att = data_mixer.get_mixing_batch()
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

    #!用插值作为输入，计算判别网络的梯度
    #!grad()函数？具体的梯度数据？
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1

    return gradient_penalty

################################################################################

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('epoch lossG lossD lossR wDistance acc')
for epoch in range(opt.nepoch):
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
            p.requires_grad = False # 避免D网络更新

        netG.zero_grad()

        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)

        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)

        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        # -----------------------
        # 对R网络进行与G同步的训练
        # -----------------------
        netR.zero_grad()

        syn_att = netR(fake)

        errR = r_criterion(syn_att, input_attv)
        errR = errR.mean()

        errR.backward(mone, retain_graph=True)
        optimizerR.step()

        errG = G_cost  - opt.r_weight * errR
        errG.backward()
        optimizerG.step()

    print('%d %.4f %.4f %.4f %.4f' % (epoch, D_cost.data[0], G_cost.data[0], errR.data[0], Wasserstein_D.data[0]))

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen_class_acc=%.4f, seen_class_acc=%.4f, h=%.4f' % (cls_.acc_unseen, cls_.acc_seen, cls_.H))
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc
        print('unseen_class_acc= ', acc)

    netG.train()