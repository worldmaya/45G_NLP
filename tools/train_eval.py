# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import pickle as pk
from tools.utils import get_time_dif
# from tensorboard import SummaryWriter
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
# with amp.scale_loss(loss, optimizer) as scaled_loss:
#     scaled_loss.backward()
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude=['embedding','encoder','embedding2',
                                                  're.weight','re_1.weight','re_2.weight','re_6.weight',
                                                  're_4.weight','re_3.weight'], seed=123):
    for name, w in model.named_parameters():
        try:
            # print(name,w)
            if name not in exclude:
                if 'weight' in name :
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
        except:
            pass


def train(config, model, train_iter, dev_iter,test_iter):
    start_time = time.time()
    model.train()
    # print(len(train_iter))#410
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    # dev_best_loss = np.inf
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, batch in enumerate(train_iter):
            # print(trains)
            batch = tuple(t.to(config.device) for t in batch)
            outputs = model(batch)
            model.zero_grad()
            loss = F.cross_entropy(outputs, batch[1])
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = batch[1].data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_best_acc<dev_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.4},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                now_time = datetime.datetime.now()
                print(now_time)
              #  writer.add_scalar("loss/train", loss.item(), total_batch)
           #     writer.add_scalar("loss/dev", dev_loss, total_batch)
             #   writer.add_scalar("acc/train", train_acc, total_batch)
              #  writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
   #  test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.4},  Test Acc: {1:>6.4%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            batch = tuple(t.to(config.device) for t in batch)
            outputs = model(batch)
            loss = F.cross_entropy(outputs, batch[1])
            loss_total += loss
            labels = batch[1].data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # print(labels_all,predict_all,config.class_list)
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
