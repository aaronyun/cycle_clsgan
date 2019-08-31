from utilities import util, opts
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

opt = opts.parse()
data = util.DATA_LOADER(opt)

train_feature = data.train_feature[0:50,:]
train_label = data.train_label[0:50]

train_feature_embed = TSNE(n_components=3).fit_transform(train_feature)

fig1 = plt.figure(num=1, tight_layout=True)

ax1 = fig1.add_subplot(131, projection='3d')
ax1.scatter(train_feature_embed[:,0], train_feature_embed[:,1], train_feature_embed[0:,2], c=train_label, marker='1', cmap=plt.cm.Spectral)
ax1.view_init(30, 0)

ax2 = fig1.add_subplot(132, projection='3d')
ax2.scatter(train_feature_embed[:,0], train_feature_embed[:,1], train_feature_embed[0:,2], c=train_label, marker='1', cmap=plt.cm.Spectral)
ax2.view_init(0, 0)

ax3 = fig1.add_subplot(133, projection='3d')
ax3.scatter(train_feature_embed[:,0], train_feature_embed[:,1], train_feature_embed[0:,2], c='r', marker='1', cmap=plt.cm.Spectral)
ax3.view_init(0, 30)

plt.savefig(opt.dataset + '_rotate.pdf')
