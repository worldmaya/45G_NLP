# coding: UTF-8
import time
from functools import partial

import torch
import numpy as np
from tools.train_eval_bert import train,test
from importlib import import_module
import os
from tools.utils import get_time_dif
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')c
# memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
# memory_gpu[0]=0
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu)) #不要0块if int(np.argmax(memory_gpu))>0 else str(1)
# print('GPUID:'+os.environ['CUDA_VISIBLE_DEVICES'])
# os.system('rm tmp')
class DatasetProcessingIterator(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = [line.strip().split('\t') for line in f.readlines() if len(line.strip().split('\t')) == 2]
        self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def collate_fn(config, sentences_and_labels):
    sentences = [sal[0] for sal in sentences_and_labels]
    labels = [config.label_map.get(sal[1]) for sal in sentences_and_labels]
    batched_input_dict = config.tokenizer(sentences, padding=True, max_length=config.pad_size, return_tensors='pt', truncation=True,
                                          return_token_type_ids=False, return_attention_mask=True, return_length=True)
    batched_input_dict['labels'] = torch.tensor(labels)
    batched_input_dict.to(config.device)
    return batched_input_dict


# 80.6250%TextCNN,81.4250
if __name__ == '__main__':
    print(torch.cuda.is_available())
    dataset = '../THUCNews'  # 数据集
    embedding = 'bert'  # 90.0873%
    model_name = 'BertTextCNN_QA'  # 96.13%,TextCNN96.73%,
    x = import_module('models.' + model_name)
    print('model:' + model_name + ' start')
    config = x.Config(dataset, embedding)
    mark = True
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()
    print("Loading data...")
    train_dataset = DatasetProcessingIterator(config.train_path)
    dev_dataset = DatasetProcessingIterator(config.dev_path)
    test_dataset = DatasetProcessingIterator(config.test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=partial(collate_fn, config), num_workers=0, pin_memory=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=partial(collate_fn, config), num_workers=0, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                collate_fn=partial(collate_fn, config), num_workers=0, pin_memory=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print(config.device)
    model = x.Model(config).to(config.device)
    train(config, model, train_dataloader, dev_dataloader, test_dataloader)
    # test(config, model, dev_dataloader)
    print('model:' + model_name + 'train end')
