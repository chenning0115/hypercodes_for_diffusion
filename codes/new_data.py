import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

TR_SIGN=1
TE_SIGN=2

def loadData():
    # 读入数据
    all_data = sio.loadmat('../../data/indian_pines/IndianPine.mat')
    data = all_data['input']
    TR = all_data['TR'] # train label
    TE = all_data['TE'] # test label
    labels = TR + TE

    return data, labels, TR, TE

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, TR, TE, windowSize=5):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesTR_TE = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            tempy = y[r-margin, c-margin]
            if tempy <= 0:
                continue
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = tempy
            patchesTR_TE[patchIndex] = TR[r-margin, c-margin] + TE[r-margin, c-margin]
            patchIndex = patchIndex + 1

    patchesData = patchesData[patchesLabels>0,:,:,:]
    patchesLabels = patchesLabels[patchesLabels>0]
    patchesLabels -= 1

    trainX = patchesData[patchesTR_TE==TR_SIGN, :, :]
    trainY = patchesLabels[patchesTR_TE==TR_SIGN]
    testX = patchesData[patchesTR_TE==TE_SIGN, :, :]
    testY = patchesLabels[patchesTR_TE==TE_SIGN]

    return trainX, trainY, testX, testY 


def norm_data(data):
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])
        input_min = np.min(data[:,:,i])
        norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
    return norm_data

batch_size = 512

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y, TR, TE = loadData()
    # X = norm_data(X)

    # 每个像素周围提取 patch 的尺寸
    patch_size = 9 
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 200 

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    Xtrain, Xtest, ytrain, ytest = createImageCubes(X_pca, y, TR, TE, windowSize=patch_size)
    print('\n... ... create train & test data ... ...')
    print('Xtrain shape: ', Xtrain.shape, 'Ytrain shape: ', ytrain.shape)
    print('Xtest  shape: ', Xtest.shape, 'Ytest shape: ', ytest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    # X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    # Xtrain = Xtrain.reshape(-1, pca_components,patch_size,patch_size)
    # Xtest = Xtest.reshape(-1, pca_components,patch_size,patch_size)
    # print('transpose: Xtrain shape: ', Xtrain.shape)
    # print('transpose: Xtest  shape: ', Xtest.shape)

    # # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_l oader和 test_loader
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )
    return train_loader, test_loader 

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len
# patches= 30;




if __name__ == "__main__":
    create_data_loader()