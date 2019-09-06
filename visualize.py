#------------------------------------------------------------------------------#
# Visualize data on local machine
#------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# exp_set = '\\zsl'
exp_set = '\\gzsl'

model = '\\frwgan'
# model = '\\rwgan'

# exp_type= '\\e4_rwgan_with_F'
exp_type= '\\base'

# dataset= '\\APY'
# dataset= '\\AWA1'
# dataset= '\\AWA2'
dataset= '\\CUB'
# dataset= '\\FLO'
# dataset= '\\SUN'

root = 'G:\\mmcgan_torch030\\fig' + exp_set + model + exp_type + dataset

# load data
label = np.load(file=root+'\\label.npy')

vf_train_embed = np.load(file=root+'\\vf_train_embed.npy')
vf_gen_embed = np.load(file=root+'\\vf_gen_embed.npy')

# hf_train_embed = np.load(file=root+'\\hf_train_embed.npy')
# hf_gen_embed = np.load(file=root+'\\hf_gen_embed.npy')

# att_embed = np.load(file=root+'\\att_embed.npy')
# train_gen_att_embed = np.load(file=root+'\\train_gen_att_embed.npy')
# gen_gen_att_embed = np.load(file=root+'\\gen_gen_att_embed.npy')


# vf&hf visualization
fig1 = plt.figure(num=1, tight_layout=True)

# visual feature
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title('train_vf vs gen_vf')
ax1.set_facecolor('#C0C0C0')
ax1.scatter(vf_train_embed[:,0], vf_train_embed[:,1], vf_train_embed[:,2], c='g', marker='o')
ax1.scatter(vf_gen_embed[:,0], vf_gen_embed[:,1], vf_gen_embed[:,2], c='r', marker='s')

fig2 = plt.figure(num=2, tight_layout='3d')

# hidden feature
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title('gen_hf')
ax1.set_facecolor('#C0C0C0')
# ax1.scatter(hf_train_embed[:,0], hf_train_embed[:,1], hf_train_embed[:,2], c='g', marker='o')
ax1.scatter(hf_gen_embed[:,0], hf_gen_embed[:,1], hf_gen_embed[:,2], c='r', marker='s')

plt.show()

# # attribute visualization
# fig2 = plt.figure(num=2, tight_layout=True)

# ax1 = fig2.add_subplot(131, projection='3d')
# ax1.scatter(att_embed[:,0], att_embed[:,1], att_embed[:,2], c=label, marker='o', cmap=plt.cm.Spectral)
# ax1.scatter(train_gen_att_embed[:,0], train_gen_att_embed[:,1], train_gen_att_embed[:,2], c=label, marker='s', cmap=plt.cm.Spectral)

# ax2 = fig2.add_subplot(132, projection='3d')
# ax2.scatter(att_embed[:,0], att_embed[:,1], att_embed[:,2], c=label, marker='o', cmap=plt.cm.Spectral)
# ax2.scatter(gen_gen_att_embed[:,0], gen_gen_att_embed[:,1], gen_gen_att_embed[:,2], c=label, marker='*', cmap=plt.cm.Spectral)

# ax3 = fig2.add_subplot(133, projection='3d')
# ax3.scatter(train_gen_att_embed[:,0], train_gen_att_embed[:,1], train_gen_att_embed[:,2], c=label, marker='s', cmap=plt.cm.Spectral)
# ax3.scatter(gen_gen_att_embed[:,0], gen_gen_att_embed[:,1], gen_gen_att_embed[:,2], c=label, marker='*', cmap=plt.cm.Spectral)

# plt.show()