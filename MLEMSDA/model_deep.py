import torch.nn as nn
from vdeep4 import deep
import torch
from os.path import join as pjoin
from torch.nn import functional as F



class deep4(nn.Module):
    def __init__(self, outpath1, cv1):
        super(deep4, self).__init__()
        base_model = deep()
        checkpoint = torch.load(pjoin(outpath1, 'model_cv{}.pt'.format(cv1)))  # 加载模型参数
        base_model.load_state_dict(checkpoint)

        self.conv1 = base_model.conv1
        self.conv1_1 = base_model.conv1_1
        self.bn1 = base_model.bn1
        self.max1 = base_model.max1

        self.conv2 = base_model.conv2
        self.bn2 = base_model.bn2
        self.max2 = base_model.max2

        self.conv3 = base_model.conv3
        self.bn3 = base_model.bn3
        self.max3 = base_model.max3

        self.conv4 = base_model.conv4
        self.bn4 = base_model.bn4
        self.max4 = base_model.max4
        self.conv_classifier=base_model.conv_classifier


    def forward(self, x):  # (64, 1, 1000, 62)
        out = self.conv1(x)  # (64 , 25, 991, 62)
        out = self.conv1_1(out)  # (64 , 25, 991, 1)
        out = F.elu(self.bn1(out))
        out = F.dropout(self.max1(out), p=0.5, training=self.training)

        out = self.conv2(out)
        out = F.elu(self.bn2(out))
        out = F.dropout(self.max2(out), p=0.5, training=self.training)

        out = self.conv3(out)
        out = F.elu(self.bn3(out))
        out = F.dropout(self.max3(out), p=0.5, training=self.training)

        out = self.conv4(out)
        out = F.elu(self.bn4(out))
        out = self.max4(out)



        out = self.conv_classifier(out)

        return out
class S_Net(nn.Module):
    def __init__(self, outpath1='', cv1=0):
        super(S_Net, self).__init__()
        self.deep4 = deep4(outpath1, cv1)

    def forward(self, target):  # (40 1 1000 11)
        S_output = self.deep4(target)
        return S_output

    def predict(self, x):
        clf = self.deep4(x)
        return clf



if __name__ == '__main__':
    model = deep()

    print(model)
    for param in model.named_parameters():
        print(param[0])