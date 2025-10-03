import h5py
import numpy as np
import torch
import torch.nn.functional as F
# from model_T import T_Net
from model_deep import S_Net

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
import torch.nn as nn
from sklearn.model_selection import KFold
from data_generator import task_generator
import higher
import warnings
# import copy

warnings.filterwarnings("ignore", category=UserWarning)
#########################################################################
dfile = h5py.File('./process_cbcic_hgd_bmi/target_cbcic_hgd_bmi/cbcic/KU_mi_smt.h5', 'r')
subjs = [1,2,3,4,5,6,7,8]
#######################################################################
device = torch.device('cuda')

def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=40,shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model.predict(test_input)
            out = F.softmax(out, dim=1)
            out = out.view(out.size(0), -1)

            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0], indices.cpu().detach().numpy()

def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

######################################
acc_5_8 = np.zeros((5, 8))
for n in range(5):
    acc_8 = np.zeros(8)
    cv_set = np.array(subjs)
    kf = KFold(n_splits=8)
    for cv_index, (train_index, test_index) in enumerate(kf.split(cv_set)):  # cv_index交叉验证次数

        outpath1 = './pretrain_deep_cbcic/Tpretrain_model'
        ###########################################################

        train_subjs = cv_set[train_index]
        test_subjs = cv_set[test_index]
        X_train, Y_train = get_multi_data(train_subjs)#560
        X_test, Y_test = get_multi_data(test_subjs)
        T_X_train_meta = X_train.transpose([0, 2, 1])
        T_X_train_meta = torch.tensor(np.expand_dims(T_X_train_meta, axis=1), dtype=torch.float32)
        T_Y_train_meta = torch.tensor(Y_train, dtype=torch.long)

        T_X_train_meta, T_Y_train_meta = T_X_train_meta.to(device), T_Y_train_meta.to(device)
        ###############################################################################
        tasks_data, tasks_labels = task_generator(T_X_train_meta, T_Y_train_meta, cv_index)
        ###############################################################################
        random_seed=[199,189,279,169,459]
        np.random.seed(random_seed[n])
        X__=np.zeros((560,11,2000))
        Y__=np.zeros(560)
        random_indices = np.random.permutation(X_train.shape[0])
        for i, idx in enumerate(random_indices):
            X__[i,:]=X_train[idx,:]
        for l,idx in enumerate(random_indices):
            Y__[l]=Y_train[idx]
        T_X_train = X__
        T_Y_train = Y__
        X___=np.zeros((80,11,2000))
        Y___=np.zeros(80)
        random_indices = np.random.permutation(X_test.shape[0])
        for i, idx in enumerate(random_indices):
            X___[i,:]=X_test[idx,:]
        for l,idx in enumerate(random_indices):
            Y___[l]=Y_test[idx]
        T_X_test = X___
        T_Y_test = Y___


        T_X_train = T_X_train.transpose([0, 2, 1])
        T_X_test = T_X_test.transpose([0, 2, 1])

        T_X_train = torch.tensor(np.expand_dims(T_X_train, axis=1), dtype=torch.float32)
        T_X_test = torch.tensor(np.expand_dims(T_X_test, axis=1), dtype=torch.float32)
        T_Y_train = torch.tensor(T_Y_train, dtype=torch.long)
        T_Y_test = torch.tensor(T_Y_test, dtype=torch.long)

        T_X_train, T_Y_train = T_X_train.to(device), T_Y_train.to(device)
        T_X_test, T_Y_test = T_X_test.to(device), T_Y_test.to(device)

        target_set = TensorDataset(T_X_train, T_Y_train)
        target_loader = DataLoader(dataset=target_set, batch_size=40, shuffle=True)

        model=S_Net(outpath1=outpath1, cv1=5)
        model.to(device)

        optimizer = torch.optim.SGD([
            {'params': model.deep4.parameters(), 'lr': 0.0001},
        ], weight_decay=5 * 0.0001, momentum=0.7)

        inner_optimizer = torch.optim.SGD([
            {'params': model.deep4.parameters(), 'lr': 0.0005},
        ], weight_decay=5 * 0.0001, momentum=0.7)

        for epoch in tqdm(range(200)):
            model.train()
            total_loss = torch.tensor(0., device=device)
            task_idx_ = 0
            for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    print("tast_idx: ", task_idx)
                    outer_loss = torch.tensor(0., device=device)
                    spt_data, qry_data = task_data[0], task_data[1]
                    spt_label, qry_label = task_label[0], task_label[1]

                    src_tensor = TensorDataset(spt_data, spt_label)
                    src_loader = DataLoader(src_tensor,
                                            batch_size=32, shuffle=True,
                                            drop_last=False)
                    for batch_idx, (inputs, labels) in enumerate(src_loader):

                        spt_S = fnet(inputs)
                        spt_S = F.softmax(spt_S, dim=1)
                        spt_S = spt_S.view(spt_S.size(0), -1)
                        sptS_loss = F.cross_entropy(spt_S, labels)
                        spt_loss = sptS_loss
                        diffopt.step(spt_loss)

                    inputs, labels = qry_data.to(device), qry_label.to(device)
                    query_S = fnet(inputs)
                    query_S = F.softmax(query_S, dim=1)
                    query_S = query_S.view(query_S.size(0), -1)
                    qur_loss = F.cross_entropy(query_S, labels)
                    outer_loss = qur_loss
                    optimizer.zero_grad()
                    outer_loss.backward()
                    optimizer.step()

            for i, (train_fea, train_lab) in enumerate(target_loader):
                S_output = model(train_fea)
                S_output1 = F.log_softmax(S_output, dim=1)
                S_output1 = S_output1.view(S_output1.size(0), -1)
                _, pred_student = torch.max(S_output1, dim=1)
                s_loss = F.nll_loss(S_output1, train_lab)
                loss_total =  s_loss
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            test_acc, _ = evaluate(model, T_X_test, T_Y_test)
            print('\n STU Validate: acc: {}'.format(test_acc))
        test_acc, _ = evaluate(model, T_X_test, T_Y_test)
        torch.save(model.state_dict(), './save_modeldeep/model_run_{}_cvindex_{}.pt'.format(n, cv_index))
        acc_8[cv_index] = test_acc
        print(acc_8)
        print('\n STU Test: acc: {}'.format(test_acc))
    acc_5_8[n] = acc_8
    print('\n acc_5_8: {}'.format(acc_5_8))
np.save('./acc_kd_8_loocv_meta_deep_exp1_5',acc_5_8)
avg_8 = np.mean(acc_5_8, axis=0)
print('\navg_8=',avg_8)
avg = np.mean(avg_8, axis=0)
print('\navg_8=',avg)









