#!/usr/bin/python3.7

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

from utilities import mix, opts, util
from utilities import classifier, classifier2, mlp

opt = opts.parse()
print(opt)

# initialize internal state
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)

# sets the seed for generating randon numbers
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# running environment setting
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load datasets and datamixer
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
data_mixer = mix.DataMixer(data, opt)

# initialize neural nets
netG = mlp.MLP_G(opt)
print(netG)

netD = mlp.MLP_CRITIC(opt)
print(netD)

reverseG = mlp.MLP_reverse_G(opt)
print(reverseG)

reverseD = mlp.MLP_reverse_D(opt)
print(reverseD)

# create input tensor
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    reverseG.cuda()
    reverseD.cuda()
    input_res, input_att, input_label = input_res.cuda(), input_att.cuda(), input_label.cuda()
    noise = noise.cuda()

    one = one.cuda()
    mone = mone.cuda()

# auxiliary functions
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

        temp = iclass_att.clone()
        temp = temp.repeat(num, 1)
        syn_att.copy_(temp)
        # syn_att.copy_(iclass_att.repeat(num, 1))

        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
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
    interpolates = interpolates

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1

    return gradient_penalty

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRD = optim.Adam(reverseD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRG = optim.Adam(reverseG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# store best result and corresponding epoch
max_H = 0
max_acc = 0
corresponding_epoch = 0

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        sample()

        # ORIGINAL DISCRIMINATOR TRAINING
        for p in netD.parameters():
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            netD.zero_grad()

            # train D with real data
            criticD_real = netD(input_res, input_att) # output Tensor
            criticD_real = criticD_real.mean()
            # we need criticD_real to be larger, so use mone
            criticD_real.backward(mone)

            # train D with generated data
            noise.normal_(0, 1)
            fake = netG(noise, input_att)

            criticD_fake = netD(fake.detach(), input_att) # detach(), detached from the current graph
            criticD_fake = criticD_fake.mean()
            # we need criticD_fake to be smaller, so use one
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data,  input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake

            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        # GENERATOR TRAINING
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        noise.normal_(0, 1)

        fake = netG(noise, input_att)
        criticG_fake = netD(fake, input_att)

        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        ################################
        # REVERSE DISCRIMINATOR TRAINING
        ################################
        reverse_feature = fake
        for p in reverseD.parameters():
            p.requires_grad = True # set to False when train reversG

        for iter_reverse_d in range(opt.reverse_iter):
            reverseD.zero_grad()

            # train reverseD with real data
            reverseD_real = reverseD(reverse_feature, input_att)
            reverseD_real = reverseD_real.mean()
            reverseD_real.backward(mone, retain_graph=True)

            # train reverseD with fake data
            fake_att = reverseG(reverse_feature) # generate fake data first

            reverseD_fake = reverseD(reverse_feature, fake_att.detach())
            reverseD_fake = reverseD_fake.mean()
            reverseD_fake.backward(one, retain_graph=True)

            # gradient penalty
            reverse_gradient_penalty = calc_gradient_penalty(reverseD, input_att, fake_att.data, reverse_feature)
            gradient_penalty.backward(retain_graph=True)

            Wasserstein_reverse_D = reverseD_real - reverseD_fake

            reverseD_cost = reverseD_fake - reverseD_real + reverse_gradient_penalty
            optimizerRD.step()

        ############################
        # REVERSE GENERATOR TRAINING
        ############################ c
        for p in reverseD.parameters():
            p.requires_grad = False

        reverseG.zero_grad()

        fake_att = reverseG(reverse_feature)
        reverseG_fake = reverseD(reverse_feature, fake_att)
        reverseG_fake = reverseG_fake.mean()
        reverseG_cost = -reverseG_fake

        reverseG_cost.backward(retain_graph=True)
        optimizerRG.step()

        #! here is Generator update
        errG = G_cost + opt.reverse_weight * reverseG_cost
        errG.backward()
        optimizerG.step()

    print('==================================================')
    print('RESULT OF EPOCH: %d' % epoch)
    print('==================================================')

    print('|    D_cost    |    G_cost    |     Wasserstein_D     |')
    print('|{:^14.4f}|{:^14.4f}|{:^23.4f}|'.format(D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0]))

    print('| reverseD_cost| reverseG_cost| Wasserstein_reverse_D |')
    print('|{:^14.4f}|{:^14.4f}|{:^23.4f}|'.format(reverseD_cost.data[0], reverseG_cost.data[0], Wasserstein_reverse_D.data[0]))

    print('What we really cared loss: G loss=%.4f' % (errG.data[0]))

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen_class_acc=%.4f, seen_class_acc=%.4f, h=%.4f' % (cls_.acc_unseen, cls_.acc_seen, cls_.H))
        print('\n')

        if cls_.H > max_H:
            mac_H = cls_.H
            corresponding_epoch = epoch
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc
        print('unseen_class_acc= ', acc)
        print('\n')

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

    netG.train()

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch))