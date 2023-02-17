import os, sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
import numpy as np
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
        self.relu = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        # x = self.linear(x)
        return x

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # --------Indian-----------
        # down_dims = [6400, 1000, 200]
        # up_dims = [200, 1000, 6400]
        # --------pavia-----------
        down_dims = [6400, 1000, 200]
        up_dims = [200, 1000, 6400]

        self.encoder = nn.ModuleList(
            [LinearBlock(down_dims[i], down_dims[i+1]) for i in range(len(down_dims)-1)]
        )
        self.decoder = nn.ModuleList(
            [LinearBlock(up_dims[i], up_dims[i+1]) for i in range(len(up_dims)-1)]
            # [nn.Sigmoid()]
        )

        self.save_feature = []

    def forward(self, x, save_feature=None):
        """
        :param [b, channel]:
        :return [b, channel]:
        """
        
        # encoder
        for e in self.encoder:
            x = e(x)
        # decoder
        if save_feature:
            temp = x.detach().cpu().numpy()
            self.save_feature.append(temp)
        for d in self.decoder:
            x = d(x)
        # reshape
        return x

    def clear_save_feature(self):
        self.save_feature = []
    
    def return_save_feature(self):
        return np.concatenate(self.save_feature, axis=0)
         


def test(model, test_loader, ori_shape, path_save_feature):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    model.clear_save_feature()
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs, save_feature=True)
        outputs = outputs.detach().cpu().numpy()
        if count == 0:
            y_pred_test = outputs
            y_test = inputs.detach().cpu().numpy()
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, inputs.detach().cpu().numpy()))
    save_feature = model.return_save_feature() 
    # reshape
    ori_h, ori_w, ori_c = ori_shape
    all_batch, new_c = save_feature.shape
    assert ori_h * ori_w == all_batch
    save_feature = save_feature.reshape((ori_h, ori_w, new_c))
    print("save_feature_shape is", save_feature.shape)
    loss = np.mean(np.square(y_pred_test - y_test))
    print("test loss = %s " % loss)
    np.save("%s" % (path_save_feature), save_feature)

    return y_pred_test, y_test

def main(path_data, prefix_path_save_feature):
    data = np.load(path_data)
    h, w, c = data.shape
    #1.1 norm化
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])
        input_min = np.min(data[:,:,i])
        norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
    data = norm_data
    print('data.shape=', data.shape)
    data = data.reshape((h*w, c))
    label = np.zeros((h*w))
    trainset = TrainDS(data, label)
    testset = TrainDS(data, label)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=2048,
                                                shuffle=True,
                                                drop_last=False
                                                )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=2048,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
    epochs = 1000
    lr = 5e-4
    model = AE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batchidx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, 'loss:', loss.item())
        if not os.path.exists(prefix_path_save_feature): 
            os.makedirs(prefix_path_save_feature)
            
        if epoch % 50 == 0:
            path_save_feature = "%s/%s" % (prefix_path_save_feature, epoch)
            test(model, test_loader, (h,w,c), path_save_feature)
            


if __name__ == '__main__':
    diffusion_data_sign_path_prefix =  "../../data/unet3d_patch16_without_downsample_kernal5_fix/save_feature"
    diffusion_data_sign = "t10_2_full.pkl.npy"
    path = "%s/%s" % (diffusion_data_sign_path_prefix, diffusion_data_sign)
    prefix_path_save_feature = "%s/%s_autoencoder_2layer" % (diffusion_data_sign_path_prefix, diffusion_data_sign)
    main(path, prefix_path_save_feature)
