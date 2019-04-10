#!/usr/bin/python3.5

from __future__ import print_function
import argparse

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import util
import classifier
import classifier2
import model

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
parser.add_argument('--drop_rate', type=float, default=0.2, help='the rate of hidden unit to dropout')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--r_weight', type=float, default=1, help='weight of the att generate loss')


# experiment setting
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--r_iteration', type=int, default=10, help='the pretraining time of R net')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--val_every', type=int, default=10)
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
# 针对数据集来进行R网络的预训练（对应着不同层数的R网络）
################################################################################

cudnn.benchmark = True

data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    # self.copy_(src)将src复制到self里
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)

if opt.dataset == 'FLO':
    netR = model.MLP_1HL_Dropout_R(opt)
elif opt.dataset == 'CUB1':
    netR = model.MLP_2HL_Dropout_R(opt)
elif opt.dataset == 'SUN1':
    netR = model.MLP_3HL_Dropout_R(opt)
elif opt.dataset == 'AWA1':
    netR = model.MLP_4HL_Dropout_R(opt)
else:
    print('There is no dataset called %s' % dataset)
print(netR)

r_criterion = nn.PairwiseDistance()
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.cuda:
    netR.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_label = input_label.cuda()
    r_criterion = r_criterion.cuda()

for param in netR.parameters():
    param.requires_grad = True

for iter_r in range(opt.r_iteration):
    batch_num = 0
    for i in range(0, data.ntrain, opt.batch_size):
        batch_num += 1
        sample()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        pretrain_att = netR(input_resv)

        pretrain_r_loss = r_criterion(pretrain_att, input_attv)
        pretrain_r_loss = pretrain_r_loss.mean()
        pretrain_r_loss.backward()

        optimizerR.step()

        print("iter:[%d/%d] batch: %d r_loss:%.4f" % (opt.r_iteration, iter_r, batch_num, pretrain_r_loss.data[0]))

# 训练完成后保存网络参数
path = './r_param/'
file_name = opt.dataset + '.pth'
torch.save(netR.state_dict(), path + file_name)