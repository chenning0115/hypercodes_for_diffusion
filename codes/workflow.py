import os, sys, time, json
import time
import utils
from utils import recorder

from data import HSIDataLoader 
from new_data import create_data_loader
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation


def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader, test_loader = dataloader.generate_torch_dataset() 
    # train_loader, test_loader, all_data_loader, _ = create_data_loader() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, test_loader)
    eval_res = trainer.final_eval(test_loader)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    
    return eval_res


# include_path = {
#     'conv2d.json',
#     'vit_30.json',
# }

include_path = {
    # 'conv3d.json',
    # 'conv2d.json',
    # 'conv1d.json',
    # 'vit_30.json',
    # 'cross_param.json'
    # 'indian_cross_param_autoencoder.json'
    # 'pavia_cross_param_autoencoder.json'
    # 'pavia_cross_param.json'
    'houston_cross_param_autoencoder.json'
}

def run_all():
    save_path_prefix = './res/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    for name in include_path:
        path_param = './params/%s' % name
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        print('start to train %s...' % name)
        eval_res = train_by_param(param)
        print('model eval done of %s...' % name)
        path = '%s/%s' % (save_path_prefix, name) 
        recorder.to_file(path)

    

def run_diffusion():
    path_param = './params/pavia_cross_param.json'
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())
    path_prefix = './res/pavia'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    #for t in [5,10,100,200,500]:
    t = 10
    index = 2
    for t in [5, 10, 100, 200]:
        for patch_size in [9,15]:
            for test_ratio in [0.95, 0.97]:
                name = "t%s_%s_full.pkl.npy" % (10, 2)
                sign = "t%s_%s_%s" % (t, patch_size, test_ratio)
                print('start to train %s...' % name)
                param['data']['diffusion_data_sign'] = name
                param['data']['patch_size'] = patch_size
                param['data']['test_ratio'] = test_ratio
                eval_res = train_by_param(param)
                print('model eval done of %s...' % sign)
                path = '%s/pavia_diffusion_%s' % (path_prefix, sign) 
                recorder.to_file(path)




if __name__ == "__main__":
    # run_diffusion()
    run_all()
    
    




