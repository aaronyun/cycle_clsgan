import torch
import torch.nn as nn


def weights_init(net):
    classname = net.__class__.__name__
    # find()寻找子字符串的位置，返回最小的下标，返回-1表示不存在
    if classname.find('Linear') != -1:
        net.weight.data.normal_(0.0, 0.02)
        net.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)

#------------------------------------------------------------------------------#

# DISCRIMINATOR
class Dis(nn.Module):
    def __init__(self, opt): 
        super(Dis, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.att_size, opt.dis_hu)
        self.fc2 = nn.Linear(opt.dis_hu, 1)

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, vf, att):
        in_ = torch.cat((vf, att), 1)

        h1 = self.fc1(in_)
        h1 = self.lrelu(h1)

        _out = self.fc2(h1)

        return _out

# class MLP_D(nn.Module):
class DisSigmoid(nn.Module):
    def __init__(self, opt): 
        super(DisSigmoid, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.att_size, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        _in = torch.cat((x, att), 1) 

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)

        h2 = self.fc2(h1)
        _out = self.sigmoid(h2)

        return _out

class robDis(nn.Module):
    def __init__(self, opt):
        super(robDis, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.att_size, opt.ndh)
        self.bin = nn.Linear(opt.ndh, 1)

        self.multi = nn.Linear(hidden_size, multi_size)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, x, att):
        _in = torch.cat((x, att), 1)

        h = self.fc1(_in)
        h = self.relu(h)

        _bin = self.bin(h).view(-1)
        _multi = self.multi(h)

        return _bin, _multi

# GENERATOR
class Gen(nn.Module):
    def __init__(self, opt):
        super(Gen, self).__init__()
        self.fc1 = nn.Linear(opt.att_size+opt.nz, opt.gen_hu)
        self.fc2 = nn.Linear(opt.gen_hu, opt.res_size)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        in_ = torch.cat((noise, att), 1)

        h1 = self.fc1(in_)
        h1 = self.lrelu(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

#------------------------------------------------------------------------------#
# Original R Net which has input size of 2048
class MLP_1HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, 4096)
        self.fc2 = nn.Linear(4096, opt.att_size)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

class MLP_2HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        _out = self.relu(h3)

        return _out

class MLP_3HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.re_hl3)
        self.fc4 = nn.Linear(opt.re_hl3, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        _out = self.relu(h4)

        return _out

class MLP_4HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.re_hl3)
        self.fc4 = nn.Linear(opt.re_hl3, opt.re_hl4)
        self.fc5 = nn.Linear(opt.re_hl4, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(res)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        h4 = self.lrelu(h4)
        h4 = self.dropout(h4)

        h5 = self.fc5(h4)
        _out = self.relu(h5)

        return _out

#------------------------------------------------------------------------------#
# R net to Fusion with input size of opt.hfSize
class MLP_1HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hf_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.att_size)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

class MLP_2HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hf_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        _out = self.relu(h3)

        return _out

class MLP_3HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hf_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.re_hl3)
        self.fc4 = nn.Linear(opt.re_hl3, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        _out = self.relu(h4)

        return _out

class MLP_4HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hf_size, opt.re_hl1)
        self.fc2 = nn.Linear(opt.re_hl1, opt.re_hl2)
        self.fc3 = nn.Linear(opt.re_hl2, opt.re_hl3)
        self.fc4 = nn.Linear(opt.re_hl3, opt.re_hl4)
        self.fc5 = nn.Linear(opt.re_hl4, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        h4 = self.lrelu(h4)
        h4 = self.dropout(h4)

        h5 = self.fc5(h4)
        _out = self.relu(h5)

        return _out

#------------------------------------------------------------------------------#

# REVERSE GENERATOR
class MLP_1HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.r_nz, opt.nrgh)
        self.fc2 = nn.Linear(opt.nrgh, opt.att_size)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _in = torch.cat((res, r_noise), 1)

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

class MLP_2HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.r_nz, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.att_size)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _in = torch.cat((res, r_noise), 1)

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        _out = self.relu(h3)

        return _out

class MLP_3HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.nrgh3)
        self.fc4 = nn.Linear(opt.nrgh3, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _in = torch.cat((res, r_noise), 1)

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        _out = self.relu(h4)

        return _out

class MLP_4HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.r_nz, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.nrgh3)
        self.fc4 = nn.Linear(opt.nrgh3, opt.nrgh4)
        self.fc5 = nn.Linear(opt.nrgh4, opt.att_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _in = torch.cat((res, r_noise), 1)

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.lrelu(h2)
        h2 = self.dropout(h2)

        h3 = self.fc3(h2)
        h3 = self.lrelu(h3)
        h3 = self.dropout(h3)

        h4 = self.fc4(h3)
        h4 = self.lrelu(h4)
        h4 = self.dropout(h4)

        h5 = self.fc5(h4)
        _out = self.relu(h5)

        return _out

# REVERSE DISCRIMINATOR
class MLP_reverse_D(nn.Module):
    def __init__(self, opt):
        super(MLP_reverse_D, self).__init__()
        self.fc1 = nn.Linear(opt.res_size+opt.att_size, opt.dis_hu)
        self.fc2 = nn.Linear(opt.dis_hu, 1)

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, att, x):
        _in = torch.cat((att, x), 1)

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)

        _out = self.fc2(h1)

        return _out

#------------------------------------------------------------------------------#

class FusionNet(nn.Module):
    def __init__(self, opt):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(opt.res_size, opt.fusion_hu)
        self.fc2 = nn.Linear(opt.fusion_hu, opt.hf_size)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.drop = nn.Dropout(opt.drop_rate)

    def forward(self, vf):
        _in = vf

        h1 = self.fc1(_in)
        h1 = self.lrelu(h1)
        h1 = self.drop(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

#------------------------------------------------------------------------------#

class AttributeNet(nn.Module):
    def __init__(self, opt):
        super(AttributeNet, self).__init__()
        self.fc1 = nn.Linear(opt.att_size, opt.an_hu)
        self.fc2 = nn.Linear(opt.an_hu, opt.res_size)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        
        _in = x
        h1 = self.fc1(_in)
        h1 = self.relu(h1)

        h2 = self.fc2(h1)
        _out = self.relu(h2)

        return _out

class RelationNet(nn.Module):
    def __init__(self, opt):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(opt.res_size*2, opt.rn_hu)
        self.fc2 = nn.Linear(opt.rn_hu, 1)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        _in = x

        h1 = self.fc1(_in)
        h1 = self.relu(h1)

        h2 = self.fc2(h1)
        _out = self.sigmoid(h2)

        return _out

#------------------------------------------------------------------------------#
