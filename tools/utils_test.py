# coding: UTF-8
import torch
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
MAX_VOCAB_SIZE = 200000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_vocab(file_path, tokenizer, max_size, min_freq,ues_word):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                if ues_word:
                    word = word.lower()#95.4736%
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(ues_word,sentence=None,config=None):
    vocab_path = "THUCNews/data/data/vocab.pkl"
    if ues_word:
        tokenizer = lambda x: x.lower().split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab = pkl.load(open(vocab_path, 'rb'))
    def load_dataset(path, pad_size):
        input_ids = []
        label_ids = []
        sequence_lengths = []
        with open(path, 'r', encoding='UTF-8') as f:
            count=0
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                a=lin.split('\t')
                if len(a)!=2:
                    count+=1
                    continue
                content = a
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)

                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                for word in token:
                    if vocab.get(word)==None:
                        # seq_len -=1
                        words_line.append(vocab.get(UNK))
                    else:
                        words_line.append(vocab.get(word))
                input_ids.append(words_line)
                # label_ids.append(config.label_map.get(label))
                sequence_lengths.append(seq_len)
            print("跳过数据:%d"%count)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)
        dataset = TensorDataset(input_ids, label_ids, sequence_lengths)
        return dataset
    def load_sentence(sentence):
        input_ids = []
        words_line = []
        token = tokenizer(sentence)
        for word in token:
            if vocab.get(word)==None:
                words_line.append(vocab.get(UNK))
            else:
                words_line.append(vocab.get(word))
        input_ids.append(words_line)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        dataset = TensorDataset(input_ids)
        return dataset
    if config!=None:
        test = load_dataset(config.test_path, config.pad_size)
    if sentence != None:
        test = load_sentence(sentence)
    return vocab,test

def build_iterator(dataset, config):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)
    return dataloader

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



