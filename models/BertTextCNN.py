# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'BertTextCNN'
        data_path = dataset + '/data/data'
        self.result_path = data_path + '/result/'
        self.train_path = data_path + '/train.txt'  # 训练集
        self.dev_path = data_path + '/dev.txt'  # 验证集   95.4379%
        self.test_path = data_path + '/dev.txt'
        self.class_list = [x.strip() for x in open(data_path + '/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.label_map = dict(zip(self.class_list, range(len(self.class_list))))
        self.vocab_path = data_path + '/vocab.pkl'  # 词表
        self.save_path = data_path + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = data_path + '/log/' + 'model'  # 模型训练结果
        self.embedding = embedding
        self.embedding_pretrained = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#,cache_dir='./transformers',local_files_only=True)                                   # 预训练词向量
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
        self.device = torch.device('cpu')  # 设备
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 500  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5
        # 学习率
        self.embed = 768  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.hidden_size = 256*2

'''Convolutional Neural Networks for Sentence Classification'''


# 78.8352%63.08
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding == 'bert':
            self.embedding = BertModel.from_pretrained('bert-base-chinese',from_tf=False)#,cache_dir='./transformers',local_files_only=True)
        self.convs = nn.ModuleList(
            [nn.Conv1d(config.embed, config.num_filters, k,padding=1) for k in config.filter_sizes]
        )
        self.final_num_2 = [3, 4, 5]
        self.fcs_2 = nn.ModuleList(
            [nn.Linear(256 * i, config.hidden_size) for i in self.final_num_2])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 3, config.num_classes)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        return x

    def final(self, out, k=None, fc=None):
        out = self.kmax_pooling(out, 2, k)
        out = out.view(out.size(0), out.size(1)*out.size(2), 1).squeeze(2)
        out = fc(out)
        return out
    def kmax_pooling(self, out, dim, k):
        x = torch.topk(out, k, dim=dim, largest=True, sorted=False)[0]  # [0]是值，[1]是index
        return  x# x是最大的

    # def forward(self, input_ids, attention_mask=None, labels=None):
    def forward(self, input_id):
        out = self.embedding(input_id[0])[0].permute(0, 2, 1)
        out = [self.conv_and_pool(out, conv) for conv in self.convs]
        out = [self.final(out[i], self.final_num_2[i], self.fcs_2[i]) for i in
               range(3)]
        out = torch.cat([i for i in out], 1)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc(out)
        return out
