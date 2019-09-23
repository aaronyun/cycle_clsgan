import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Read arguments from commandline')

#------------------------------------------------------------------------------#

    # Exp Env Setting
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

#------------------------------------------------------------------------------#

    # Data Specification
    parser.add_argument('--dataroot', default='/data0/docker/xingyun/datasets')
    parser.add_argument('--dataset', default='AWA2', help='which dataset, AWA2 as the default dataset')
    parser.add_argument('--matdataset', default=True, help='whether dataset in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--res_size', type=int, default=2048)
    parser.add_argument('--att_size', type=int, default=85)
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
    parser.add_argument('--ntrain_class', type=int, default=40, help='number of train classes') #! NEW

    parser.add_argument('--preprocessing', action='store_true', default=False, help='enable MinMaxScaler on visual features')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--val_every', type=int, default=10)

#------------------------------------------------------------------------------#

    # Adversarial Sample
    parser.add_argument('--adv_steps', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.0625)

#------------------------------------------------------------------------------#

    # Generator Setting
    parser.add_argument('--netG', default='', help='path to netG (to continue training')
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--gen_hu', type=int, default=4096, help='size of the hidden layer in generator')
    parser.add_argument('--nz', type=int, default=85, help='size of the noise z vector')

    # Discriminator Setting
    parser.add_argument('--netD', default='', help='path to netD (to continue training')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--dis_hu', type=int, default=1024, help='size of the hidden layer in discriminator')

    # Reverse Net Setting
    # parser.add_argument('--r_iteration', type=int, default=3, help='the pretraining time of R net')
    parser.add_argument('--r_hl', type=int, default=1, help='how many hidden layers in R net')
    parser.add_argument('--re_hl1', type=int, default=1024, help='size of the first layer in R net')
    parser.add_argument('--re_hl2', type=int, default=512, help='size of the second layer in R net')
    parser.add_argument('--re_hl3', type=int, default=256, help='size of the third layer in R net')
    parser.add_argument('--re_hl4', type=int, default=128, help='size of the fourth layer in R net')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='the rate of unit to drop in R net')
    parser.add_argument('--r_weight', type=float, default=1, help='weight of att generate loss')

    # Fusion Net Setting
    parser.add_argument('--hf_size', type=int, default=512, help='hidden feature size(final layer size of fusion net)')
    parser.add_argument('--fusion_iter', type=int, default=2, help='how many times training fusion net in a epoch')
    parser.add_argument('--triplet_num', type=int, default=128, help='batch size of Fusion Net training')
    parser.add_argument('--fusion_hu', type=int, default=1024, help='size of hidden units in fusion net')

#------------------------------------------------------------------------------#

    # Relation Net Setting
    parser.add_argument('--rn_hu', type=int, default=1200, help='hidden units of relation net, different dataset has different number of units')
    parser.add_argument('--an_hu', type=int, default=1200, help='hidden units of attribute net, different datset has different number of units')
    parser.add_argument('--rn_episodes', type=int, default=200000, help='times to train relation net in a epoch')    
    parser.add_argument('--netAtt', default='', help='path to netAtt (to continue training')
    parser.add_argument('--netRN', default='', help='path to netRN (to continue training')
    parser.add_argument('--rn_batch_size', type=int, default=1024, help='batch size when training relation net')

#------------------------------------------------------------------------------#

    # Exp Setting
    parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')

    # detail
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GAN')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--syn_num', type=int, default=100, help='number of features to generate per class')

    # final classifier
    parser.add_argument('--pretrain_classifier', default='', help='path to pretrain classifier (to continue training)')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train final softmax classifier')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')

#------------------------------------------------------------------------------#

    # Unknown
    parser.add_argument('--outname', help='folder to output data and model checkpoints')
    parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

    opt = parser.parse_args()

    return opt