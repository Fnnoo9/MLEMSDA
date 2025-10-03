from __future__ import print_function



import h5py
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from vdeep4 import deep
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import torch
import time

print(torch.cuda.is_available())
device = torch.device('cuda')


dfile_bmi = h5py.File('../process_cbcic_hgd_bmi/target_cbcic_hgd_bmi/bmi/KU_mi_smt.h5', 'r')
dfile_hgd = h5py.File('../process_cbcic_hgd_bmi/target_cbcic_hgd_bmi/hgd/ku_mi_smt.h5', 'r')


outpath = './Tpretrain_model'

##########################################################################openbmi
def get_data_bmi(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile_bmi[pjoin(dpath, 'X')]
    Y = dfile_bmi[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)
def get_multi_data_bmi(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data_bmi(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y
###########################################################################hgd
def get_data_hgd(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile_hgd[pjoin(dpath, 'X')]
    Y = dfile_hgd[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)
def get_multi_data_hgd(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data_hgd(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

def evaluate(model, x, y):
    # print(x.shape, y.shape)
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0]


X_bmi,Y_bmi = get_multi_data_bmi(np.arange(1,55,1))#21600 11 2000
X_hgd,Y_hgd = get_multi_data_hgd(np.arange(1,15,1))#6742 11 2000
X= np.concatenate((X_bmi,X_hgd), axis=0)#28342
Y=np.concatenate((Y_bmi,Y_hgd), axis=0)

# 固定随机种子
np.random.seed(42)
# 生成相同的随机索引
random_index = np.random.permutation(len(X))
# 使用相同的索引对数据和标签进行重排
X_ = X[random_index]
Y_ = Y[random_index]

start = time.time()
kf = KFold(n_splits=10,shuffle=True)


patience = 20

cv_loss = np.ones([10])  # 交叉验证 最小验证损失
result = pd.DataFrame(columns=('cv_index', 'test_acc', 'loss'))
train_loss2 = np.zeros([10,600])
test_loss2 = np.zeros([10,600])
train_accuracy2 = np.zeros([10,600])
test_accuracy2 = np.zeros([10,600])

for cv_index, (train_index, test_index) in enumerate(kf.split(X_)):  # cv_index交叉验证次数
    tra_acc = np.zeros([600])
    tra_loss = np.zeros([600])
    tst_acc = np.zeros([600])
    tst_loss = np.zeros([600])

    X_train = X_[train_index]#25507 11 1000
    Y_train = Y_[train_index]
    X_test = X_[test_index]#2835 11 1000
    Y_test = Y_[test_index]


    X_train = X_train.transpose([0, 2, 1])#(25507, 1000, 11)
    X_test = X_test.transpose([0, 2, 1])#(2835, 1000, 11)

    print(Y_train.shape, Y_test.shape)
    print(np.sum(Y_train == 0) / Y_train.shape[0], np.sum(Y_test == 0) / Y_test.shape[0])

    X_train = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
    X_test = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    train_set = TensorDataset(X_train, Y_train)
    test_set = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    model = deep().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1 * 0.0001, weight_decay=0.5 * 0.001)

    train_losses = []
    test_losses = []
    train_loss_avg = []
    test_loss_avg = []
    early_stop = EarlyStopping(patience, delta=0.0001, path=pjoin(outpath, 'model_cv{}.pt'.format(cv_index)),
                               verbose=True)

    for epoch in tqdm(range(600)):
        model.train()
        t = 0
        for i, (train_fea, train_lab) in enumerate(train_loader):
            out = model(train_fea)
            _, pred = torch.max(out, dim=1)
            t += (pred == train_lab).sum().cpu().numpy()
            loss = F.nll_loss(out, train_lab)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('\n Train, epoch: {}, i: {}, loss: {}'.format(epoch, i, loss))

        e = 0
        model.eval()
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, dim=1)
            e += (pred == target).sum().cpu().numpy()
            loss = F.nll_loss(output, target)
            test_losses.append(loss.item())

        train_loss = np.average(train_losses)
        test_loss = np.average(test_losses)
        train_loss_avg.append(train_loss)
        test_loss_avg.append(test_loss)

        tra_loss[epoch] = train_loss
        tst_loss[epoch] = test_loss
        tra_acc[epoch] = t / X_train.shape[0]
        tst_acc[epoch] = e / X_test.shape[0]

        if test_loss < cv_loss[cv_index]:
            cv_loss[cv_index] = test_loss

        train_losses = []
        test_losses = []

        test_acc = evaluate(model, X_test, Y_test)
        print('\n Test: acc: {}'.format(test_acc))

        res = {'cv_index': cv_index, 'test_acc': test_acc.item(), 'loss': test_loss}
        result = result.append(res, ignore_index=True)

        early_stop(test_loss, model)
        if early_stop.early_stop:
            print('Early stopping')
            break
    train_loss2[cv_index] = tra_loss
    test_loss2[cv_index] = tst_loss
    train_accuracy2[cv_index] = tra_acc
    test_accuracy2[cv_index] = tst_acc

    # plt.subplot(2, 2, 1)
    # plt.plot(tra_acc)
    # plt.subplot(2, 2, 2)
    # plt.plot(tst_acc)
    # plt.subplot(2, 2, 3)
    # plt.plot(tra_loss)
    # plt.subplot(2, 2, 4)
    # plt.plot(tst_loss)
    # plt.show()
np.save('train_loss2',train_loss2)
np.save('test_loss2',test_loss2)
np.save('train_accuracy2',train_accuracy2)
np.save('test_accuracy2',test_accuracy2)

min_index = np.argmin(cv_loss, axis=0)
result.to_csv(pjoin(outpath, 'result_cv{}.csv'.format(min_index)), index=False)