#------------------------------------------------------------------------------#
# For visualization on local
#------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# exp_set = '\\zsl'
exp_set = '\\gzsl'

model = '\\frwgan'

exp_type= '\\e4_rwgan_with_F'
# exp_type= '\\base'

# dataset= '\\APY'
# dataset= '\\AWA1'
# dataset= '\\AWA2'
# dataset= '\\CUB'
dataset= '\\FLO'
# dataset= '\\SUN'

root = 'G:\\mmcgan_torch030\\fig' + exp_set + model + exp_type + dataset

label = np.load(file=root+'\\label.npy')
train_vf_embed = np.load(file=root+'\\train_vf_embed.npy')
gen_vf_embed = np.load(file=root+'\\gen_vf_embed.npy')
train_hf_embed = np.load(file=root+'\\train_hf_embed.npy')
gen_hf_embed = np.load(file=root+'\\gen_hf_embed.npy')

# att_embed = np.load(file=root+'\\att_embed.npy')
# train_gen_att_embed = np.load(file=root+'\\train_gen_att_embed.npy')
# gen_gen_att_embed = np.load(file=root+'\\gen_gen_att_embed.npy')


# vf&hf visualization
fig1 = plt.figure(num=1, tight_layout=True)

# row 1 for vf
ax1 = fig1.add_subplot(221, projection='3d')
ax1.set_title('train_vf vs gen_vf(diff color)')
ax1.set_facecolor('#C0C0C0')
ax1.scatter(train_vf_embed[:,0], train_vf_embed[:,1], train_vf_embed[:,2], c=label, marker='o', cmap=plt.cm.Spectral)
ax1.scatter(gen_vf_embed[:,0], gen_vf_embed[:,1], gen_vf_embed[:,2], c=label, marker='s', cmap=plt.cm.Spectral)

ax2 = fig1.add_subplot(222, projection='3d')
ax2.set_title('train_vf vs gen_vf(same color)')
ax2.set_facecolor('#C0C0C0')
ax2.scatter(train_vf_embed[:,0], train_vf_embed[:,1], train_vf_embed[:,2], c='g', marker='o')
ax2.scatter(gen_vf_embed[:,0], gen_vf_embed[:,1], gen_vf_embed[:,2], c='r', marker='s')

# row 2 for hf
ax3 = fig1.add_subplot(223, projection='3d')
ax3.set_title('train_hf vs gen_hf(diff color)')
ax3.set_facecolor('#C0C0C0')
ax3.scatter(train_hf_embed[:,0], train_hf_embed[:,1], train_hf_embed[:,2], c=label, marker='o', cmap=plt.cm.Spectral)
ax3.scatter(gen_hf_embed[:,0], gen_hf_embed[:,1], gen_hf_embed[:,2], c=label, marker='s', cmap=plt.cm.Spectral)

ax4 = fig1.add_subplot(224, projection='3d')
ax4.set_title('train_hf vs gen_hf(same color)')
ax4.set_facecolor('#C0C0C0')
ax4.scatter(train_hf_embed[:,0], train_hf_embed[:,1], train_hf_embed[:,2], c='g', marker='o')
ax4.scatter(gen_hf_embed[:,0], gen_hf_embed[:,1], gen_hf_embed[:,2], c='r', marker='s')

plt.show()

# # att visualization
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