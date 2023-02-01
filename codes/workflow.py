import os, sys, time, json
import time
import utils
from utils import recorder

from data import HSIDataLoader 
from trainer import BaseTrainer, CrossTransformerTrainer
import evaluation


def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader, test_loader, all_data_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = CrossTransformerTrainer(param)
    trainer.train(train_loader, test_loader)
    eval_res = trainer.final_eval(test_loader)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    
    return eval_res


def run_default(param):
    path_prefix = './res/'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    print('start to train default...')
    param['diffusion_sign'] = False 
    param['pca'] = 200 
    param['spectral_size'] =200
    eval_res = train_by_param(param)
    print('model eval done of default...')
    path = '%s/default' % path_prefix 
    recorder.to_file(path)



def run_diffusion(param):
    path_prefix = './res/patch_8_pca_2000'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    for t in [5,10,100,200,500]:
        for index in [0,1,2]:
            name = "t%s_%s_full.pkl.npy" % (t, index)
            print('start to train %s...' % name)
            param['diffusion_data_sign'] = name
            eval_res = train_by_param(param)
            print('model eval done of %s...' % name)
            path = '%s/indian_diffusion_%s' % (path_prefix, name) 
            recorder.to_file(path)




if __name__ == "__main__":
    path_param = './params/cross_param.json'
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())

    
    # run_diffusion(param)
    run_default(param)
    
    




