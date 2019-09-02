import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Read arguments from commandline')

    #! DATA SPECIFICATION
    parser.add_argument('--dataroot', default='/data0/docker/xingyun/mmcgan/data')
    parser.add_argument('--dataset', default='AWA2', help='which dataset, AWA2 as the default dataset')
    parser.add_argument('--matdataset', default=True, help='whether dataset in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--resSize', type=int, default=2048)
    parser.add_argument('--attSize', type=int, default=85)
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

    parser.add_argument('--preprocessing', action='store_true', default=False, help='enable MinMaxScaler on visual features')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--standardization', action='store_true', default=False)

    #! G&D SPECIFICATION
    # parser.add_argument('--netG', default='', help='path to netG (to continue training')
    # parser.add_argument('--netD', default='', help='path to netD (to continue training')
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden layer in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden layer in discriminator')
    parser.add_argument('--nz', type=int, default=85, help='size of the noise z vector')

    #! REVERSE NET SPECIFICATION
    # parser.add_argument('--r_nz', type=int, default=2048, help='size of the noise in reverse net')
    # parser.add_argument('--r_path', default='/home/xingyun/docker/mmcgan/r_param', help='path to load parameters of R net')
    # parser.add_argument('--r_iteration', type=int, default=3, help='the pretraining time of R net')
    parser.add_argument('--r_hl', type=int, default=1, help='how many hidden layers in R net')
    parser.add_argument('--nrh1', type=int, default=1024, help='size of the first layer in R net')
    parser.add_argument('--nrh2', type=int, default=512, help='size of the second layer in R net')
    parser.add_argument('--nrh3', type=int, default=256, help='size of the third layer in R net')
    parser.add_argument('--nrh4', type=int, default=128, help='size of the fourth layer in R net')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='the rate of unit to drop in R net')
    parser.add_argument('--r_weight', type=float, default=1, help='weight of att generate loss')
    parser.add_argument('--consistency_weight', type=float, default=1, help='weight of semantic consistency loss add on feature generator loss')

    # parser.add_argument('--reverse_iter', type=int, default=5, help='training iteration of reverse D')
    # parser.add_argument('--rg_hl', type=int , default=1, help='layers of reverse generator')
    # parser.add_argument('--nrgh', type=int, default=4096, help='size of hidden units in R net when R has only one hidden layer')
    # parser.add_argument('--nrgh1', type=int, default=1024, help='size of the first layer in R net')
    # parser.add_argument('--nrgh2', type=int, default=512, help='size of the second layer in R net')
    # parser.add_argument('--nrgh3', type=int, default=256, help='size of the third layer in R net')
    # parser.add_argument('--nrgh4', type=int, default=128, help='size of the fourth layer in R net')

    #! FUSION NET SPECIFICATION
    #? --hfSize应该分数据集分别调参
    parser.add_argument('--hfSize', type=int, default=512, help='hidden feature size(final layer size of fusion net)')
    parser.add_argument('--fusion_iter', type=int, default=3, help='how many times training fusion net in a epoch')
    parser.add_argument('--triple_batch_size', type=int, default=128, help='batch size of Fusion Net training')

    #! EXPERIMENT SPECIFICATION
    # exp type
    parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    # detail
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GAN')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    # final classifier
    parser.add_argument('--pretrain_classifier', default='', help='path to pretrain classifier (to continue training)')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train final softmax classifier')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--syn_num', type=int, default=100, help='number of features to generate per class')
    # bc learning
    parser.add_argument('--bc', type=bool, default=False, help='whether use BC learning')

    #! EXPERIMENT ENVIRONMENT SPECIFICATION
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--val_every', type=int, default=10)

    #! UNKNOWN
    parser.add_argument('--outname', help='folder to output data and model checkpoints')
    parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

    opt = parser.parse_args()

    return opt