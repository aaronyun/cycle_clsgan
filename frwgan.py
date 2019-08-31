from __future__ import print_function

import os
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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from utilities import opts, util
from utilities import classifier, classifier2, mlp, f_classifier

#------------------------------------------------------------------------------#

# parameters
opt = opts.parse()
print(opt)

# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass

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

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#------------------------------------------------------------------------------#

# load datasets
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

#------------------------------------------------------------------------------#

# Generator initialize
netG = mlp.MLP_G(opt)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Discriminator initialize
netD = mlp.MLP_CRITIC(opt)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Reverse net initialize
if opt.r_hl == 1:
    netR = mlp.MLP_1HL_Dropout_R(opt)
elif opt.r_hl == 2:
    netR = mlp.MLP_2HL_Dropout_R(opt)
elif opt.r_hl == 3:
    netR = mlp.MLP_3HL_Dropout_R(opt)
elif opt.r_hl == 4:
    netR = mlp.MLP_4HL_Dropout_R(opt)
else:
    raise('wrong value of r_hl')
# if opt.r_hl == 1:
#     netR = mlp.AD_1HL_Dropout_R(opt)
# elif opt.r_hl == 2:
#     netR = mlp.AD_2HL_Dropout_R(opt)
# elif opt.r_hl == 3:
#     netR = mlp.AD_3HL_Dropout_R(opt)
# elif opt.r_hl == 4:
#     netR = mlp.AD_4HL_Dropout_R(opt)
# else:
#     raise('wrong value of r_hl')
# print(netR)
# 只有CUB(312)和FLO(1024)数据集的属性维度变大

# Fusion net initialize
netF = mlp.MLP_DropoutAdapt(opt)
print(netF)

#------------------------------------------------------------------------------#

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# loss function
cos_criterion = nn.CosineSimilarity()
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
euc_criterion = nn.PairwiseDistance(p=2)

#------------------------------------------------------------------------------#

# create input tensor
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
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
    triplet_criterion = triplet_criterion.cuda()
    euc_criterion = euc_criterion.cuda()

    one = one.cuda()
    mone = mone.cuda()

#------------------------------------------------------------------------------#

# auxiliary functions
def sample():
    batch_feature, batch_label, batch_att, batch_index = data.next_batch(opt.batch_size)

    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
    input_index.copy_(batch_index)

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
        noisev = Variable(noise)
        gen_vfv = netG(noisev, input_attv)

        # train G with Discriminator
        criticG_fake = netD(gen_vfv, input_attv)
        criticG_fake = criticG_fake.mean()

        G_cost = -criticG_fake

#------------------------------------------------------------------------------#

        ################################
        # FUSION TRAINING
        ################################
        # Data Preperation
        input_vf = input_res # anchor from training data
        gen_vf = gen_vfv.data # anchor generated from attribute

        anchor_vf = torch.cat((input_vf, gen_vf), 0)
        # index of anchor in all training visual features
        anchor_index = (input_index.repeat(1,2)).squeeze()

        # triple constructor
        triple_data = util.Triple_Selector(data, anchor_vf, anchor_index)

        # Training
        for iter_f in range(opt.fusion_iter):
            netF.zero_grad()

            # get a batch of triples
            triple_batch = triple_data.next_batch(opt.batch_size, triple_type='hardest')

            # train F Net with triples
            triple_batchv = Variable(triple_batch)
            triple_hf = netF(triple_batchv)

            # triplet loss
            anchor = triple_hf[0 : opt.batch_size]
            pos = triple_hf[opt.batch_size : opt.batch_size*2]
            neg = triple_hf[opt.batch_size*2 : opt.batch_size*3] 

            triplet_loss = triplet_criterion(anchor, pos, neg)
            triplet_loss = triplet_loss.mean()
            triplet_loss.backward(retain_graph=True)

            # consistency loss
            train_hf = netF(Variable(input_vf))
            gen_hf = netF(Variable(gen_vf))
            #? 应该采用类似球形的损失
            consistency_loss = euc_criterion(train_hf, gen_hf)
            consistency_loss = consistency_loss.mean()
            consistency_loss.backward(retain_graph=True)

            # update parameters
            F_cost = triplet_loss + consistency_loss
            optimizerF.step()

#------------------------------------------------------------------------------#

        ################################
        # R TRAINING
        ################################
        netR.zero_grad()

        #! VER: train R with generated visual features
        #R training
        syn_attv = netR(gen_vfv)

        # attribute consistency loss
        R_cost = cos_criterion(syn_attv, input_attv)
        R_cost = R_cost.mean()

        #! VER: train R with generated vf and train vf
        # gen_syn_attv = netR(gen_vfv) 
        # train_syn_attv = netR(input_vfv)
        # gen_errR = cos_criterion(gen_syn_attv, input_attv)
        # train_errR = cos_criterion(train_syn_attv, input_attv)

        # R_cost = gen_errR + train_errR
        # R_cost = R_cost.mean()

        #! VER: train R with hidden features generated by F
        # # R training
        # syn_train_attv = netR(train_hf)
        # syn_gen_attv = netR(gen_hf)

        # # attribute consistency loss
        # err_syn_train = cos_criterion(syn_train_attv, input_attv)
        # err_syn_gen = cos_criterion(syn_gen_attv, input_attv)
        # R_cost = err_syn_train + err_syn_gen
        # R_cost = R_cost.mean()

        #!!! Update R net
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

        # train final classifier and evaluate model
        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        # data_to_plot.append([D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H])

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|{:^12.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H 
            corresponding_epoch = epoch

            # tsne when get best result
            tsne_train_vf_embed = TSNE(n_components=3).fit_transform(input_vf.cpu().numpy())
            tsne_gen_vf_embed = TSNE(n_components=3).fit_transform(gen_vf.cpu().numpy())
            tsne_train_hf_embed = TSNE(n_components=3).fit_transform(train_hf.data.cpu().numpy())
            tsne_gen_hf_embed = TSNE(n_components=3).fit_transform(gen_hf.data.cpu().numpy())
            tsne_label = (input_label.cpu()).numpy()

    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        # data_to_plot.append([D_cost.data[0], G_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc])

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], F_cost.data[0], R_cost.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))

# save visualization data
exp_set = '/gzsl'

model = '/frwgan'

exp_type = '/e4_rwgan_with_F/'
# exp_type = '/base/'

root = '/home/xingyun/docker/mmcgan_torch030/fig' + exp_set + model + exp_type + opt.dataset

np.save(file=root+'/label', arr=tsne_label)
np.save(file=root+'/train_vf_embed', arr=tsne_train_vf_embed)
np.save(file=root+'/gen_vf_embed', arr=tsne_gen_vf_embed)
np.save(file=root+'/train_hf_embed', arr=tsne_train_hf_embed)
np.save(file=root+'/gen_hf_embed', arr=tsne_gen_hf_embed)