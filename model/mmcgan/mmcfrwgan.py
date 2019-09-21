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
import numpy as np

sys.path.append('/data0/docker/xingyun/projects/mmcgan')

from util import opts, tools, mlp
from util.eval import classifier, mm_classifier

# parameters
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

# load datasets
data = tools.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# Generator initialize
netG = mlp.G(opt.att_size + opt.nz, opt.ngh, opt.res_size)
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
    netR = mlp.MLP_1HL_Dropout_R(opt.res_size, 4096, opt.att_size)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_R(opt.res_size, opt.nrh1, opt.nrh2, opt.att_size)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_R(opt.res_size, opt.nrh1, opt.nrh2, opt.nrh3, opt.att_size)
elif opt.r_hl == 4:
    netR = mlp.MLP_4HL_Dropout_R(opt.res_size, opt.nrh1, opt.nrh2, opt.nrh3, opt.nrh4, opt.att_size)
else:
    raise('wrong value of r_hl')
print(netR)

# Fusion net initialize
netF = mlp.FusionNet(opt.res_size, 1024, opt.hfSize)

# semantic consistency loss
cos_criterion = nn.CosineSimilarity()
# fusion loss
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# create input tensor
input_res = torch.FloatTensor(opt.batch_size, opt.res_size)
input_att = torch.FloatTensor(opt.batch_size, opt.att_size)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netR.cuda()
    netF.cuda()

    noise, input_res, input_att, input_label  = noise.cuda(), input_res.cuda(), input_att.cuda(), input_label.cuda()

    cos_criterion = cos_criterion.cuda()
    triplet_criterion = triplet_criterion.cuda()

    one = one.cuda()
    mone = mone.cuda()

# auxiliary functions
def sample():
    batch_feature, batch_label, batch_att, batch_index = data.next_batch(opt.batch_size)

    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(tools.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)

    syn_feature = torch.FloatTensor(nclass*num, opt.res_size)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.att_size)
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

        # generate visual feature
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
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# store best result and corresponding epoch
max_H = 0.0
max_acc = 0.0
corresponding_epoch = 0

# preperation for visualization
data_to_plot = []

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |  ACC_seen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  R_cost  |  Wasserstein_D  |  ACC_unseen  |')

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        ################################
        # DISCRIMINATOR TRAINING
        ################################
        for p in netD.parameters():
            p.requires_grad = True # set to False after netD training

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

            # train D with generated data
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
            p.requires_grad = False # avoid computation

        ############################
        # GENERATOR TRAINING
        ###########################
        netG.zero_grad()

        sample()
        input_vfv = Variable(input_res)
        input_attv = Variable(input_att)

        # generate fake data
        noise.normal_(0, 1)
        noisev = Variable(noise)
        gen_vfv = netG(noisev, input_attv)

        criticG_fake = netD(gen_vfv, input_attv)
        criticG_fake = criticG_fake.mean()

        G_cost = -criticG_fake

        ################################
        # FEATURE FUSION TRAINING
        ################################
        # Data Preperation
        train_vf = input_vfv.data # anchor from training data
        gen_vf = gen_vfv.data # anchor generated from attribute
        anchor_vf = torch.cat((train_vf, gen_vf), 0)
        anchor_index = batch_index.repeat(1,2)

        triple_data = tools.Triple_Selector(data, anchor_vf, anchor_index) # triple selector

        # Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad()

            # train F Net with triple of train and generated visual feature
            mixing_triple_batch = triple_data.next_batch(opt.triple_batch_size, triple_type='hardest') # get a batch of mixing triples
            mixing_triple_batchv = Variable(mixing_triple_batch)
            mixing_hf = netF(mixing_triple_batchv)

            # triplet loss
            anchor = mixing_hf[0 : opt.triple_batch_size]
            pos = mixing_hf[opt.triple_batch_size : opt.triple_batch_size*2]
            neg = mixing_hf[opt.triple_batch_size*2 : opt.triple_batch_size*3]

            triplet_loss = triplet_criterion(anchor, pos, neg, margin=1.0)
            triplet_loss = triplet_loss.mean()
            triplet_loss.backward(retain_graph=True)

            # consistency loss
            train_hf = netF(train_vf)
            gen_hf = netF(gen_vf)
            consistency_loss = cos_criterion(train_hf, gen_hf)
            consistency_loss = consistency_loss.mean()
            consistency_loss.backward(retain_graph=True)

            # update parameters
            F_cost = triplet_loss + consistency_loss
            optimizerF.step()

        ################################
        # R TRAINING
        ################################
        netR.zero_grad()

        # data to train R
        train_hfv = netF(input_vfv)
        gen_hfv = netF(gen_vfv)

        # R training
        syn_gen_attv = netR(gen_hfv)
        syn_train_attv = netR(train_hfv)

        # loss of R
        errR_train = cos_criterion(syn_train_attv, input_attv)
        errR_gen = cos_criterion(syn_gen_attv, input_attv)
        R_cost = err_train + err_gen
        R_cost = R_cost.mean()

        R_cost.backward(mone, retain_graph=True)
        optimizerR.step()

        ################################
        # FINAL GENERATOR LOSS
        ################################
        errG = G_cost - opt.r_weight * R_cost
        errG.backward()
        optimizerG.step()

    netG.eval()

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        syn_att = netR(Variable(syn_feature.cuda(), volatile=True))
        train_feature, train_label = data.train_feature, data.train_label
        train_att = netR(Variable(train_feature.cuda(), volatile=True))

        vf = torch.cat((data.train_feature, syn_feature), 0)
        att = torch.cat((train_att.data.cpu(), syn_att.data.cpu()), 0)

        # concatenate visual feature and attribute
        train_X = torch.cat((vf, att), 1)
        train_Y = torch.cat((train_label, syn_label), 0)
        nclass = opt.nclass_all

        # train final classifier and evaluate model
        cls_ = mm_classifier.CLASSIFIER(netR, train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        data_to_plot.append([D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H])

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H 
            corresponding_epoch = epoch
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        syn_att = torch.index_select(data.attribute, 0, syn_label)

        data_to_plot.append([D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc])

        cls_ = mm_classifier.CLASSIFIER(syn_feature, tools.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

    netG.train()

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

# visualization
x = np.arange(1, opt.nepoch+1)
data_to_plot = np.array(data_to_plot)

# plt.subplot(311)
# plt.plot(x, data_to_plot[:,0], label='Discriminator')
# plt.plot(x, data_to_plot[:,1], label='Generator')
# plt.plot(x, data_to_plot[:,2], label='Rverse Net')

# plt.subplot(312)
# plt.plot(x, data_to_plot[:,4], label='unseen class acc')
# plt.plot(x, data_to_plot[:,5], label='seen class acc')
# plt.plot(x, data_to_plot[:,6], label='h')

# plt.subplot(313)
# plt.plot(x, data_to_plot[:,3], label='wasserstein distance')

# only plot unseen acc and seen acc
# plt.plot(x, data_to_plot[:,4], 'r', label='unseen class acc')
# plt.plot(x, data_to_plot[:,5], 'k', label='seen class acc')

# save figure
# plt.savefig('/data0/xingyun/docker/mmcgan_torch030/figure/' + opt.dataset + 'cost_fig.pdf')
