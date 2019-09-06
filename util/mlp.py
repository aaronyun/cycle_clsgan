import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    # find()寻找子字符串的位置，返回最小的下标，返回-1表示不存在
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# DISCRIMINATOR
class MLP_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, vf, att):
        in_ = torch.cat((vf, att), 1)
        h = self.lrelu(self.fc1(in_))
        out_ = self.fc2(h)
        return out_

class MLP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class robDis(nn.Module):
    def __init__(self, opt):
        super(robDis, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.attSize, opt.ndh)

        self.bin = nn.Linear(opt.ndh, 1)
        nn.init.xavier_uniform(self.bin.weight, gain=1.0)
        self.multi = nn.Linear(opt.ndh, opt.ntrain_class)
        nn.init.xavier_uniform(self.multi.weight, gain=1.0)

        self.relu = nn.ReLU(True)

    def forward(self, x, att):
        _in = torch.cat((x, att), 1)
        h = self.relu(self.fc1(_in))

        _bin = self.bin(h)
        _multi = self.multi(h)

        return _bin.view(-1), _multi


# GENERATOR
class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        in_ = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(in_))
        out_ = self.relu(self.fc2(h))
        return out_

#------------------------------------------------------------------------------#
# Original R Net which has input size of 2048
class MLP_1HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, 4096)
        self.fc2 = nn.Linear(4096, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))

        return h

class MLP_2HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.attSize)
        # 当使用两个隐藏层的时候，各层的激活函数用什么呢
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        h = self.relu(self.fc3(h))

        return h

class MLP_3HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.nrh3)
        self.fc4 = nn.Linear(opt.nrh3, opt.attSize)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc3(h))
        h = self.dropout(h)
        h = self.relu(self.fc4(h))

        return h

class MLP_4HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.nrh3)
        self.fc4 = nn.Linear(opt.nrh3, opt.nrh4)
        self.fc5 = nn.Linear(opt.nrh4, opt.attSize)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc3(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc4(h))
        h = self.dropout(h)
        h = self.relu(self.fc5(h))

        return h

#------------------------------------------------------------------------------#
# R net to Fusion with input size of opt.hfSize
class MLP_1HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hfSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.attSize)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        _out = self.relu(self.fc2(h))

        return _out

class MLP_2HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hfSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.attSize)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res
        h = self.lrelu(self.fc1(_in))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        _out = self.relu(self.fc3(h))

        return _out

class MLP_3HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hfSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.nrh3)
        self.fc4 = nn.Linear(opt.nrh3, opt.attSize)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res
        h = self.lrelu(self.fc1(_in))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc3(h))
        h = self.dropout(h)
        _out = self.relu(self.fc4(h))

        return _out

class MLP_4HL_Dropout_FR(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_Dropout_FR, self).__init__()
        self.fc1 = nn.Linear(opt.hfSize, opt.nrh1)
        self.fc2 = nn.Linear(opt.nrh1, opt.nrh2)
        self.fc3 = nn.Linear(opt.nrh2, opt.nrh3)
        self.fc4 = nn.Linear(opt.nrh3, opt.nrh4)
        self.fc5 = nn.Linear(opt.nrh4, opt.attSize)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        _in = res
        h = self.lrelu(self.fc1(_in))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc3(h))
        h = self.dropout(h)
        h = self.lrelu(self.fc4(h))
        h = self.dropout(h)
        _out = self.relu(self.fc5(h))

        return _out

#------------------------------------------------------------------------------#

# REVERSE GENERATOR
class MLP_1HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.r_nz, opt.nrgh)
        self.fc2 = nn.Linear(opt.nrgh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _input = torch.cat((res, r_noise), 1)

        h = self.fc1(_input)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        _out = self.relu(h)

        return _out

class MLP_2HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.r_nz, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _input = torch.cat((res, r_noise), 1)

        h = self.fc1(_input)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc3(h)
        _out = self.relu(h)

        return _out

class MLP_3HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.r_nz, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.nrgh3)
        self.fc4 = nn.Linear(opt.nrgh3, opt.attSize)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _input = torch.cat((res, r_noise), 1)

        h = self.fc1(_input)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc3(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc4(h)
        _out = self.relu(h)

        return h

class MLP_4HL_reverseG(nn.Module):
    def __init__(self, opt):
        super(MLP_4HL_reverseG, self).__init__()
        self.fc1 = nn.Linear(opt.resSize+opt.r_nz, opt.nrgh1)
        self.fc2 = nn.Linear(opt.nrgh1, opt.nrgh2)
        self.fc3 = nn.Linear(opt.nrgh2, opt.nrgh3)
        self.fc4 = nn.Linear(opt.nrgh3, opt.nrgh4)
        self.fc5 = nn.Linear(opt.nrgh4, opt.attSize)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res, r_noise):
        _input = torch.cat((res, r_noise), 1)

        h = self.fc1(_input)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc3(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc4(h)
        h = self.lrelu(h)
        h = self.dropout(h)

        h = self.fc5(h)
        h = self.relu(h)

        return h

# REVERSE DISCRIMINATOR
class MLP_reverse_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_reverse_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, att, x):
        h = torch.cat((att, x), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

#------------------------------------------------------------------------------#

# JUDGE
class TF_judge(nn.Module):
    def __init__(self, opt):
        super(TF_judge, self).__init__()
        self.fc1 = nn.Linear(opt.attSize*2, opt.attSize)
        self.fc2 = nn.Linear(opt.attSize, 1)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, fake_att, real_att):
        _in = torch.cat((fake_att, real_att), 1)
        h = self.fc1(_in)
        h = self.lrelu(h)
        h = self.fc2(h)
        # h = self.sigmoid(h)
        _out = self.relu(h)

        return _out

#------------------------------------------------------------------------------#

class MLP_Dropout_Fusion(nn.Module):
    def __init__(self, opt):
        super(MLP_Dropout_Fusion, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, 1024)
        self.fc2 = nn.Linear(1024, opt.hfSize)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.drop = nn.Dropout(p=opt.drop_rate)

    def forward(self, vf):
        _in = vf
        h = self.lrelu(self.fc1(_in))
        h = self.drop(h)
        _out = self.relu(self.fc2(h))

        return _out

#------------------------------------------------------------------------------#
#TEST#
class test_3_HL_R(nn.Module):
    def __init__(self, opt):
        super(test_3_HL_R, self).__init__()
        self.fc1 = nn.Linear(opt.hfSize, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, opt.attSize)

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.drop = nn.Dropout(p=opt.drop_rate)

    def forward(self, vf):
        _in = vf
        h = self.lrelu(self.fc1(_in))
        h = self.drop(h)
        h = self.lrelu(self.fc2(h))
        h = self.drop(h)
        h = self.lrelu(self.fc3(h))
        h = self.drop(h)
        _out = self.relu(self.fc4(h))

        return _out