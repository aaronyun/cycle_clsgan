from __future__ import print_function

import os
import random
import argparse

import torch
import torch.nn as nn 
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utilities import util, mlp, opts
from utilities import classifier, classifier2

opt = opts.parse()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# intialize internal state
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

# load dataset
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = mlp.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mlp.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

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

    noise = noise.cuda()
    input_res, input_att, input_label = input_res.cuda(), input_att.cuda(), input_label.cuda()

    cls_criterion.cuda()

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
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
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
    
# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# store best result and corresponding epoch
max_H = 0
max_acc = 0
corresponding_epoch = 0

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():
    p.requires_grad = False

if opt.gzsl:
    print('EPOCH          |  D_cost  |  G_cost  |  CLS_cost  |  Wasserstein_D  |  ACC_seen  |  ACC_unseen  |    H    |')
else:
    print('EPOCH          |  D_cost  |  G_cost  |  CLS_cost  |  Wasserstein_D  |  ACC_unseen  |')

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # DISCRIMINATOR TRAINING
        ###########################
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
            fake = netG(noisev, input_attv)

            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_attv)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # GENERATOR TRAINING
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # avoid computation

        netG.zero_grad()

        # generate fake data
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)

        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()

        G_cost = -criticG_fake

        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        errG = G_cost + opt.cls_weight * c_errG
        errG.backward()
        optimizerG.step()

    netG.eval() # evaluate the model, set G to evaluation mode

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        cls_ = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^12.4f}|{:^17.4f}|{:^12.4f}|{:^14.4f}|{:^9.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], c_errG.data[0], Wasserstein_D.data[0], cls_.acc_unseen, cls_.acc_seen, cls_.H))

        if cls_.H > max_H:
            max_H = cls_.H
            corresponding_epoch = epoch
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num) 

        cls_ = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls_.acc

        print('[{:^4d}/{:^4d}]    |{:^10.4f}|{:^10.4f}|{:^12.4f}|{:^17.4f}|{:^14.4f}|'.format(epoch+1, opt.nepoch, D_cost.data[0], G_cost.data[0], c_errG.data[0], Wasserstein_D.data[0], acc))

        if acc > max_acc:
            max_acc = acc
            corresponding_epoch = epoch

    netG.train() # reset G to training mode

if opt.gzsl:
    print('max H: %f in epoch: %d' % (max_H, corresponding_epoch+1))
else:
    print('max unseen class acc: %f in epoch: %d' % (max_acc, corresponding_epoch+1))