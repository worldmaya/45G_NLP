# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import init_network,train
from importlib import import_module
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from utils import build_dataset, build_iterator, get_time_dif
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
memory_gpu[0]=0
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu)) #不要0块if int(np.argmax(memory_gpu))>0 else str(1)
print('GPUID:'+os.environ['CUDA_VISIBLE_DEVICES'])
os.system('rm tmp')
if __name__ == '__main__':
    print(torch.cuda.is_available())
    dataset = 'THUCNews'  # 数据集
    embedding = 'random'#90.0873%
    model_name = 'TextCNN'#96.13%,TextCNN96.73%,
    x = import_module('models.' + model_name)
    print('model:' + model_name + ' start')
    config = x.Config(dataset, embedding)
    mark = False
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data,test_data = build_dataset(config, mark)
    train_iter = build_iterator(train_data, config, set_type='train')
    dev_iter = build_iterator(dev_data, config, set_type='dev')
    test_iter = build_iterator(test_data, config, set_type='test')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    print("vocab_num : {}".format(config.n_vocab))
    print(config.device)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter,test_iter)
    print('model:' + model_name + 'train end')