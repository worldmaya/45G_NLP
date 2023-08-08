# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'test2'
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
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
            self.embedding = BertModel.from_pretrained('bert-base-chinese')#,cache_dir='./transformers',local_files_only=True)
        self.device = config.device
        self.layer_num_2 = 7
        self.final_num_2 = [7,6,5, 4, 3, 2,1]
        k = config.embed
        self.convs_2 = nn.ModuleList(
            [nn.Conv1d(k, k, 2, padding=1) for i in range(self.layer_num_2)])
        self.fcs_2 = nn.ModuleList(
            [nn.Linear(k * i, config.hidden_size) for i in self.final_num_2])
        self.re = nn.PReLU()
        self.fc = nn.Linear(config.hidden_size * (self.layer_num_2), config.num_classes)
        self.dropout = nn.Dropout(config.dropout)


    def conv_and_pool(self, x, conv):
            x = self.re(conv(x))
            return x
    def kmax_pooling(self, out, dim, k):
        index = out.topk(k,dim = dim)[1].sort(dim=dim)[0]
        return  out.gather(dim,index = index)# x是最大的
    def final(self, out, k=None, fc=None):
        out = self.kmax_pooling(out, 2, k)
        out = out.view(out.size(0), out.size(1) * out.size(2), 1).squeeze(2)
        out = self.dropout(out)
        out = self.re(out)
        out = fc(out)
        return out
    def DPN(self, out,conv,k):
        out = self.upsanmple(out)
        out = self.conv_and_pool(out, conv)
        out = self.kmax_pooling(out, 2, k)
        return out
    def upsanmple(self, out):
        out = nn.functional.interpolate(out, size=out.size(2) * 2)
        return out

    def forward(self, input_ids, attention_mask,length, labels=None):
        seq_lenth = int(length.type(torch.float16).mean())
        out = self.embedding(input_ids,attention_mask)[0].permute(0, 2, 1)
        out_puts_2 = []
        out_1 = self.DPN(out, self.convs_2[0], int(seq_lenth * 1.8))
        out_puts_2.append(out_1)
        out_2 = self.DPN(out_1, self.convs_2[1], int(seq_lenth * 1.5))
        out_puts_2.append(out_2)
        out_3 = self.DPN(out_2, self.convs_2[2], int(seq_lenth * 1.2))
        out_puts_2.append(out_3)
        out_4 = self.DPN(out_3, self.convs_2[3], int(seq_lenth * 0.9))
        out_puts_2.append(out_4)
        out_5 = self.DPN(out_4, self.convs_2[4], int(seq_lenth * 0.6))
        out_puts_2.append(out_5)
        out_6 = self.DPN(out_5, self.convs_2[5], int(seq_lenth * 0.3))
        out_puts_2.append(out_6)
        out_6 = self.DPN(out_5, self.convs_2[6], int(seq_lenth * 0.1))#98.08
        out_puts_2.append(out_6)
        out_puts_2 = [self.final(out_puts_2[i], self.final_num_2[i], self.fcs_2[i]) for i in
                      range(self.layer_num_2)]
        out_puts_2 = torch.cat([i for i in out_puts_2], 1)
        out = self.dropout(out_puts_2)
        out = F.relu(out)  #
        out = self.fc(out)
        return out
# 97.29