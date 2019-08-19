from __future__ import print_function

import os
import sys
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

from utilities import opts, util
from utilities import classifier, classifier2, mlp

opt = opts.parse()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

# load datasets and datamixer
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
# data_mixer = mix.DataMixer(data, opt)

# initialize neural nets
netG = mlp.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mlp.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

if opt.rg_hl == 1:
    netRG = mlp.MLP_1HL_reverseG(opt)
elif opt.rg_hl == 2:
    netRG = mlp.MLP_2HL_reverseG(opt)
elif opt.rg_hl == 3:
    netRG = mlp.MLP_3HL_reverseG(opt)
elif opt.rg_hl == 4:
    netRG = mlp.MLP_4HL_reverseG(opt)
else:
    raise('wrong value of r_hl')
print(netRG)

netRD = mlp.MLP_reverse_D(opt)
print(netRD)

# semantic feature consistency loss
r_criterion = nn.CosineSimilarity()
# r_criterion = nn.PairwiseDistance()

# create input tensor
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
r_noise = torch.FloatTensor(opt.batch_size, opt.r_nz)


one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    # netR.cuda()
    netRG = netRG.cuda()
    netRD = netRD.cuda()

    noise, input_res, input_att, input_label  = noise.cuda(), input_res.cuda(), input_att.cuda(), input_label.cuda()
    r_noise = r_noise.cuda()

    r_criterion = r_criterion.cuda()

    one = one.cuda()
    mone = mone.cuda()

# auxiliary functions
def sample():
    if opt.bc:
        batch_feature, batch_label, batch_att = data_mixer.get_mixing_batch()
    else:
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

        # temp = iclass_att.clone()
        # temp = temp.repeat(num, 1)
        # syn_att.copy_(temp)
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
# optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRG = optim.Adam(netRG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRD = optim.Adam(netRD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# store best result and corresponding epoch
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  Wasserstein_D  |  RD_cost  |  RG_cost  |  Wasserstein_RD  |  att_consistency  |  ACC_seen  |  ACC_unseen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  Wasserstein_D  |  RD_cost  |  RG_cost  |  Wasserstein_RD  |  att_consistency  |  ACC_unseen  |')

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        ################################
        # DISCRIMINATOR TRAINING
        ################################
        for p in netD.parameters():
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            # train D with real data
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train D with generated data
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake_res = netG(noisev, input_attv)

            criticD_fake = netD(fake_res.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake_res.data, input_attv)
            gradient_penalty.backward()

            # wasserstein distance
            Wasserstein_D = criticD_real - criticD_fake

            # Discriminator Loss
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # GENERATOR TRAINING
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # avoid computation

        sample()
        netG.zero_grad()

        # generate fake data
        noise.normal_(0, 1)
        noisev = Variable(noise)
        input_attv = Variable(input_att)
        fake_res = netG(noisev, input_attv)

        # train G with D
        criticG_fake = netD(fake_res, input_attv)
        criticG_fake = criticG_fake.mean()

        # Generator Loss
        G_cost = -criticG_fake

        ################################
        # REVERSE DISCRIMINATOR TRAINING
        ################################
        for p in netRD.parameters():
            p.requires_grad = True

        for iter_reverseD in range(opt.reverse_iter):
            sample()
            netRD.zero_grad()

            # the groundtruth of reverse training are generated
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            reverse_res = netG(noisev, input_attv)

            # train netRD with real data
            reverseD_real = netRD(input_attv, reverse_res)
            reverseD_real = reverseD_real.mean()
            reverseD_real.backward(mone, retain_graph=True)

            # generate fake attribute
            r_noise.normal_(0, 1)
            r_noisev = Variable(r_noise)
            fake_att = netRG(reverse_res, r_noisev)

            # train netRD with fake data
            reverseD_fake = netRD(fake_att, reverse_res)
            reverseD_fake = reverseD_fake.mean()
            reverseD_fake.backward(one, retain_graph=True)

            # gradient penalty
            reverse_gradient_penalty = calc_gradient_penalty(netRD, input_att, fake_att.data, reverse_res)
            reverse_gradient_penalty.backward()

            # wasserstein distance
            Wasserstein_RD = reverseD_real - reverseD_fake

            reverseD_cost = reverseD_fake - reverseD_real + reverse_gradient_penalty
            optimizerRD.step()

        ################################
        # REVERSE GENERATOR TRAINING
        ################################
        for p in netRD.parameters():
            p.requires_grad = False

        sample()
        netRG.zero_grad()

        # we postulate reverse_vf, generated by netG, are groundtruth for reverse training
        noise.normal_(0, 1)
        noisev = Variable(noise)
        input_attv = Variable(input_att)
        reverse_vf = netG(noisev, input_attv)

        # generate fake_att with netRG
        r_noise.normal_(0, 1)
        r_noisev = Variable(r_noise)
        fake_att = netRG(reverse_vf, r_noisev)

        # train netRG with netRD
        reverseG_fake = netRD(fake_att, reverse_vf)
        reverseG_fake = reverseG_fake.mean()
        reverseG_cost = -reverseG_fake

        # update netRG directly
        reverseG_cost.backward(retain_graph=True)
        optimizerRG.step()

        # ATT CONSISTENCY
        att_consistency = r_criterion(fake_att, input_attv)
        att_consistency = att_consistency.mean()

        # ################################
        # # R TRAINING
        # ################################
        # netR.zero_grad()

        # syn_att = netR(fake_vf)

        # # caculate r loss with predefined loss function
        # errR = r_criterion(syn_att, input_attv)
        # R_cost = errR.mean()

        # # backward R and J simultaneously
        # R_cost.backward(mone, retain_graph=True)
        # optimizerR.step()

        # Final Generator Loss
        errG = G_cost - opt.consistency_weight * att_consistency
        errG.backward()
        optimizerG.step()

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^11.4f}|{:^11.4f}|{:^18.4f}|{:^19.4f}|{:^12.4f}|{:^14.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0], reverseD_cost.data[0], reverseG_cost.data[0], Wasserstein_RD.data[0], att_consistency.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H
            corresponding_epoch = epoch
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^11.4f}|{:^11.4f}|{:^18.4f}|{:^19.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0], reverseD_cost.data[0], reverseG_cost.data[0], Wasserstein_RD.data[0],  att_consistency, acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

    netG.train()

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))