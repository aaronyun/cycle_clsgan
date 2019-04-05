#!/usr/bin/python3.5

import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__ #! m究竟是什么
    # find()寻找子字符串的位置，返回最小的下标，返回-1表示不存在
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_AC_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_AC_2HL_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_2HL_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, x):
        h = self.dropout(self.lrelu(self.fc1(x)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_3HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_3HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.fc4(h)
        return h

class MLP_2HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h

class MLP_2HL_Dropout_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_Dropout_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

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

class MLP_2HL_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.relu(self.fc3(h))
        return h

class MLP_3HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return h

class MLP_2HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.relu(self.fc2(h))
        return h

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        # fc1的输入大小是属性向量的大小加上噪声的大小
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_2048_1024_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2048_1024_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc3 = nn.Linear(1024, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        #self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h

class MLP_SKIP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_SKIP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc_skip = nn.Linear(opt.attSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        #h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc2(h))
        h2 = self.fc_skip(att)
        return h+h2

class MLP_SKIP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_SKIP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc_skip = nn.Linear(opt.attSize, opt.ndh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h2 = self.lrelu(self.fc_skip(att))
        h = self.sigmoid(self.fc2(h+h2))
        return h

# 线性映射R网络
class MLP_R(nn.Module):
    def __init__(self, opt):
        super(MLP_R, self).__init__()
        self.fc = nn.Linear(opt.resSize, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc(res))
        return h

# 单隐藏层R网络
class MLP_1HL_Dropout_R(nn.Module):
    def __init__(self, opt):
        super(MLP_1HL_Dropout_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.nrh)
        self.fc2 = nn.Linear(opt.nrh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=opt.drop_rate)

        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))

        return h

# 双隐藏层R网络
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