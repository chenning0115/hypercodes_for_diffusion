
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
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return x

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        down_dims = [3328, 1000, 103]
        up_dims = [103, 1000, 3328]
        self.encoder = nn.ModuleList(
            [LinearBlock(down_dims[i], down_dims[i+1]) for i in range(len(down_dims)-1)]
        )
        self.decoder = nn.ModuleList(
            [LinearBlock(up_dims[i], up_dims[i+1]) for i in range(len(up_dims)-1)]
            # [nn.Sigmoid()]
        )

    def forward(self, x):
        """
        :param [b, channel]:
        :return [b, channel]:
        """
        
        # encoder
        for e in self.encoder:
            x = e(x)
        # decoder
        for d in self.decoder:
            x = d(x)
        # reshape
        return x

def test(model, test_loader, calloss=False):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    if calloss:
        print("test loss = %s " % np.mean(y_pred_test - y_test))
    return y_pred_test, y_test

def main(path_data):
    data = np.load(path_data)
    #1.1 norm化
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])
        input_min = np.min(data[:,:,i])
        norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
    data = norm_data

    print('data.shape=', data.shape)
    h, w, c = data.shape
    data = data.reshape((h*w, c))
    label = np.zeros((h*w))
    data_set = TrainDS(data, label)
    train_loader = DataLoader(data_set, batch_size=1024, shuffle=True)
    epochs = 1000
    lr = 1e-3
    model = AE()
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 不需要label，所以用一个占位符"_"代替
        model.train()
        for batchidx, (x, _) in enumerate(train_loader):
            x_hat = model(x)
            loss = criteon(x_hat, x)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, 'loss:', loss.item())
        if epoch % 10 == 0:
            test(model, train_loader)
            

if __name__ == '__main__':
    diffusion_data_sign_path_prefix =  "../../data/pavia_unet3d_patch16_without_downsample_kernal5_fix/save_feature"
    diffusion_data_sign = "t10_2_full.pkl.npy"
    path = "%s/%s" % (diffusion_data_sign_path_prefix, diffusion_data_sign)
    main(path)
