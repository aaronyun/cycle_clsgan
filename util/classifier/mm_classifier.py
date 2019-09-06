import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Variable

import numpy as np
from sklearn.preprocessing import MinMaxScaler 

sys.path.append('/home/xingyun/docker/mmcgan_torch030')

from util import tools

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, netR,  _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        self.cuda = _cuda

        self.train_X =  _train_X
        self.train_Y = _train_Y

        # test seen data
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        # self.test_seen_att = data_loader.attribute[self.test_seen_label]
        if self.cuda:
            self.test_seen_att = netR(Variable(self.test_seen_feature.cuda(), volatile=True))
        else:
            self.test_seen_att = netR(Variable(self.test_seen_feature, volatile=True))

        # test unseen data
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        if self.cuda:
            self.test_unseen_att = netR(Variable(self.test_unseen_feature.cuda(), volatile=True))
        else:
            self.test_unseen_att = netR(Variable(self.test_unseen_feature, volatile=True))

        # DATA FOR EVALUATE MODEL
        self.test_seen = torch.cat((self.test_seen_feature, self.test_seen_att.data.cpu()), 1)
        self.test_unseen = torch.cat((self.test_unseen_feature, self.test_unseen_att.data.cpu()), 1)
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses

        self.nclass = _nclass
        self.ntrain = self.train_X.size(0)
        self.input_dim = _train_X.size(1)

        self.nepoch = _nepoch
        self.batch_size = _batch_size # equals to syn_num
        self.lr = _lr
        self.beta1 = _beta1

        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(tools.weights_init)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit_gzsl()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))

    # ZSL
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc = self.val_zsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
        return best_acc

    # test_label is integer 
    def val_zsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True)) 
            else:
                output = self.model(Variable(test_X[start:end], volatile=True)) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_zsl(tools.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc_zsl(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() 

    # GZSL
    def fit_gzsl(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                # print('Training classifier loss= ', loss.data[0])

            # evaluate model every epoch
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen, self.test_unseen_label, self.unseenclasses)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            # print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H

        return best_seen, best_unseen, best_H

    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True)) 
            else:
                output = self.model(Variable(test_X[start:end], volatile=True)) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, nclass)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x): 
        h = self.fc1(x)
        h = self.lrelu(h)

        h = self.fc2(h)
        h = self.lrelu(h)

        h = self.fc3(h)
        h = self.logic(h)

        return h