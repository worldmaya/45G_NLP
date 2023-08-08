# coding: UTF-8
import openpyxl
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

def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: [y for y in x]  # char-level
    else:
        tokenizer = lambda x: x.lower().split(' ')  # 以空格隔开，word-level
    min_freq = 1
    print("最小词频:%d"%min_freq)
    vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=min_freq,ues_word = ues_word)
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")
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
                content, label = a
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
                        words_line.append(vocab.get(UNK))
                    else:
                        words_line.append(vocab.get(word))
                if words_line ==  None or config.label_map.get(label) ==  None:
                    print(label)
                input_ids.append(words_line)
                label_ids.append(config.label_map.get(label))
                sequence_lengths.append(seq_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)
        dataset = TensorDataset(input_ids, label_ids, sequence_lengths)
        return dataset
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev,test

def build_iterator(dataset, config, set_type='train'):
    sampler = RandomSampler(dataset) if set_type == 'train' else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)
    return dataloader

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def getdata(path):
    data_path = r'THUCNews/data/data/'
    class_file = data_path + 'class.txt'
    train = data_path + 'train.txt'
    dev = data_path + 'dev.txt'
    wb = openpyxl.load_workbook(path)
    classes = {}
    with open(train, 'w+', encoding='utf-8') as tr, open(dev, 'w+', encoding='utf-8') as de,open(class_file, 'w+', encoding='utf-8') as Cl:
        for x in wb:
            print(x.title)
            Cl.write(x.title+'\n')
            sheet_ranges = wb[x.title]
            classes[x.title] = 1
            for val in sheet_ranges.values:
                classes[x.title] += 1
        for x in wb:
            count = 0
            sheet_ranges = wb[x.title]
            for val in sheet_ranges.values:
                if val[0] != None:
                    content = val[0].replace('\t', '').replace('\n', '') + val[2].replace('\t', '').replace('\n',
                                                                               '') + '\t' + x.title + '\n'
                else:
                    content = val[2].replace('\t', '').replace('\n', '') + '\t' + x.title + '\n'
                if count < classes[x.title] * 0.8:
                    tr.write(content)
                else:
                    de.write(content)
                count += 1

def adddata(path):
    data_path = r'THUCNews/data/data/'
    class_file = data_path + 'class.txt'
    train = data_path + 'train.txt'
    dev = data_path + 'dev.txt'
    wb = openpyxl.load_workbook(path)
    classes = {}
    with open(class_file, 'r', encoding='utf-8') as cl:
        for x in wb:
            print(x.title)
            sheet_ranges = wb[x.title]
            classes[x.title] = 1
            for val in sheet_ranges.values:
                classes[x.title] += 1
        i = cl.readlines()
        cla=[x.strip() for x in i]
        for x in wb:
            if x.title not in cla:
                cla.append(x.title)
    with open(class_file, 'w+', encoding='utf-8') as cl:
        for l in cla :
            cl.write(l+'\n')
    with open(train, 'a+', encoding='utf-8') as tr, open(dev, 'a+', encoding='utf-8') as de:
        for x in wb:
            count = 0
            sheet_ranges = wb[x.title]
            for val in sheet_ranges.values:
                if val[0] != None:
                    content = val[0].replace('\t', '').replace('\n', '') + val[2].replace('\t', '').replace('\n',
                                                                               '') + '\t' + x.title + '\n'
                else:
                    content = val[2].replace('\t', '').replace('\n', '') + '\t' + x.title + '\n'
                if count < classes[x.title] * 0.8:
                    tr.write(content)
                else:
                    de.write(content)
                count += 1


if __name__ == "__main__":
    path = r'C:\Users\zxy\Desktop\比赛\软件设计\test.xlsx'
    a = adddata(path)
    print(a)