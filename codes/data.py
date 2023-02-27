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



class HSIDataLoader(object):
    def __init__(self, param) -> None:
        self.data_param = param['data']
        self.data_path_prefix = "../../data"
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)

        # 参数设置
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.patch_size = self.data_param.get('patch_size', 13) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', True)
        self.test_ratio = self.data_param.get('test_ratio', 0.9)
        self.batch_size = self.data_param.get('batch_size', 256)
        self.none_zero_num = self.data_param.get('none_zero_num', 0)
        self.spectracl_size = self.data_param.get("spectral_size", 0)
        self.append_dim = self.data_param.get("append_dim", False)
        self.use_norm = self.data_param.get("use_norm", True)


        self.diffusion_sign = self.data_param.get('diffusion_sign', False)
        self.diffusion_data_sign_path_prefix = self.data_param.get("diffusion_data_sign_path_prefix", '')
        self.diffusion_data_sign = self.data_param.get("diffusion_data_sign", "unet3d_27000.pkl")

    def load_data_from_diffusion(self):
        path = "%s/%s" % (self.diffusion_data_sign_path_prefix, self.diffusion_data_sign)
        if self.data_sign == "Indian":
            data_ori = sio.loadmat('%s/Indian_pines_corrected.mat' % self.data_path_prefix)['indian_pines_corrected']
            labels = sio.loadmat('%s/Indian_pines_gt.mat' % self.data_path_prefix)['indian_pines_gt']
        if self.data_sign == "Pavia":
            data_ori = sio.loadmat('%s/PaviaU.mat' % self.data_path_prefix)['paviaU']
            labels = sio.loadmat('%s/PaviaU_gt.mat' % self.data_path_prefix)['paviaU_gt'] 
        if self.data_sign == "Houston":
            data_ori = sio.loadmat('%s/Houston.mat' % self.data_path_prefix)['img']
            labels = sio.loadmat('%s/Houston_gt.mat' % self.data_path_prefix)['Houston_gt']
        else:
            pass
        data = np.load(path)
        ori_h, ori_w, _= data_ori.shape
        h, w, _= data.shape
        assert ori_h == h, ori_w == w
        print("load diffusion data shape is ", data.shape)
        return data, labels 

    def load_data(self):
        data, labels = None, None
        if self.diffusion_sign:
            return self.load_data_from_diffusion()

        if self.data_sign == "Indian":
            data = sio.loadmat('%s/Indian_pines_corrected.mat' % self.data_path_prefix)['indian_pines_corrected']
            labels = sio.loadmat('%s/Indian_pines_gt.mat' % self.data_path_prefix)['indian_pines_gt']
        elif self.data_sign == "Pavia":
            data = sio.loadmat('%s/PaviaU.mat' % self.data_path_prefix)['paviaU']
            labels = sio.loadmat('%s/PaviaU_gt.mat' % self.data_path_prefix)['paviaU_gt'] 
        elif self.data_sign == "Houston":
            data = sio.loadmat('%s/Houston.mat' % self.data_path_prefix)['img']
            labels = sio.loadmat('%s/Houston_gt.mat' % self.data_path_prefix)['Houston_gt']
        return data, labels

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX
    
    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]

    def createImageCubes(self, X, y, windowSize=5, removeZeroLabels = True):

        # 给 X 做 padding
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self._padding(X, margin=margin)
        # split patches
        validy_size = self.get_valid_num(y)
        patchesData = np.zeros((validy_size, windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((validy_size))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if not y[r-margin, c-margin] > 0:
                    continue
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1

        return patchesData, patchesLabels

    def get_patches(self, X, Y, patch_size=3, remove_zero=True, none_zeros_num=0):
        w,h,c = X.shape
        #1. padding
        margin = (patch_size - 1) // 2
        X_padding = self._padding(X, margin=margin)
        #2. split patchs
        if none_zeros_num == 0:
            none_zeros_num = w * h
        X_patchs = np.zeros((none_zeros_num, patch_size, patch_size, c)) #one pixel one patch with padding
        Y_patchs = np.zeros((none_zeros_num))
        # 循环X_padding上的每一个有效pixel
        patch_index = 0
        for r in range(margin, X_padding.shape[0]-margin):
            for c in range(margin, X_padding.shape[1]-margin):
                temp_patch = X_padding[r-margin:r+margin+1, c-margin:c+margin+1, :]
                if Y[r-margin, c-margin] > 0:
                    X_patchs[patch_index, :, :, :] = temp_patch
                    Y_patchs[patch_index] = Y[r-margin, c-margin]
                    patch_index += 1
        if remove_zero:
            X_patchs = X_patchs[Y_patchs>0]
            Y_patchs = Y_patchs[Y_patchs>0]
            Y_patchs -= 1
        return X_patchs, Y_patchs #(batch, w, h, c), (batch)

    def applyPCA(self,   X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def generate_torch_dataset(self):
        #1. 根据data_sign load data
        self.data, self.labels = self.load_data()

        #1.1 norm化
        if self.use_norm:
            norm_data = np.zeros(self.data.shape)
            for i in range(self.data.shape[2]):
                input_max = np.max(self.data[:,:,i])
                input_min = np.min(self.data[:,:,i])
                norm_data[:,:,i] = (self.data[:,:,i]-input_min)/(input_max-input_min)
        else:
            norm_data = self.data 
        if self.data_param['pca'] > 0:
            pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
            # norm_data = np.concatenate([norm_data, pca_data], axis=-1)
            norm_data = pca_data
        if self.spectracl_size > 0: # 按照给定的spectral size截取数据
            norm_data = norm_data[:,:,:self.spectracl_size]
        
        # else:
            # norm_data = np.concatenate([norm_data, norm_data], axis=-1) #TODO: 暂时double数据

        print('[data] load data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))
        #2. 获取patchs
        # X_patchs, Y_patchs =  self.get_patches(norm_data, self.labels, patch_size=self.patch_size, remove_zero=self.remove_zeros,
                                    # none_zeros_num=self.none_zero_num)
        
        X_patchs, Y_patchs = self.createImageCubes(norm_data, self.labels, windowSize=self.patch_size, removeZeroLabels = True)
        print('[data] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))
        #3. 随机区分train和test
        X_train, X_test, Y_train, Y_test = train_test_split(X_patchs,
                                                        Y_patchs,
                                                        test_size=self.test_ratio,
                                                        random_state=345,
                                                        stratify=Y_patchs)
        print('------[data] split data to train, test------')
        print("X_train shape : %s" % str(X_train.shape))
        print("Y_train shape : %s" % str(Y_train.shape))
        print("X_test shape : %s" % str(X_test.shape))
        print("Y_test shape : %s" % str(Y_test.shape))

        #4. 调整shape来满足torch使用
        if not self.append_dim:
            X_train = X_train.transpose((0, 3, 1, 2)) # (batch, spectral, h, w)                                               
            X_test = X_test.transpose((0, 3, 1, 2)) # (batch, spectral, h, w)                                               
        else:
            # 先增加维度 作为3d cnn使用
            X_train = np.expand_dims(X_train, 1)
            X_train = X_train.transpose((0, 1, 4, 2, 3)) # (batch, spectral, h, w)                                               
            X_test = np.expand_dims(X_test, 1)
            X_test = X_test.transpose((0, 1, 4, 2, 3)) # (batch, spectral, h, w)                                               


        print('------[data] after transpose train, test------')
        print("X_train shape : %s" % str(X_train.shape))
        print("Y_train shape : %s" % str(Y_train.shape))
        print("X_test shape : %s" % str(X_test.shape))
        print("Y_test shape : %s" % str(Y_test.shape))

        trainset = TrainDS(X_train, Y_train)
        testset = TestDS(X_test, Y_test)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=False
                                                )
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        return train_loader, test_loader

       


if __name__ == "__main__":
    dataloader = HSIDataLoader({})
    train_loader, test_loader = dataloader.generate_torch_dataset()
