import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        # LeakyReLU是ReLU函数的一个变体，解决了ReLU函数存在的问题，α的默认往往是非常小的，比如0.01，这样就保证了Dead Neurons的问题。
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))  # 初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        G = G.to(torch.float32)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G.to_sparse(), x)
        return x
class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,Q,A, dropout=0.1, momentum=0.1):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_dim, hidden_dim)
        self.hgc2 = HGNN_conv(hidden_dim, 64)
        # self.computeG = compute_G(W)
        self.batch_normalzation1 = nn.BatchNorm1d(input_dim)
        self.batch_normalzation2 = nn.BatchNorm1d(hidden_dim)
        self.batch_normalzation3 = nn.BatchNorm1d(64)
        self.A = A

        self.lin = nn.Linear(64, num_classes)
        self.lin2 = nn.Linear(input_dim, input_dim)
        self.Q=Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
    def forward(self, x):
        (h, w, c) = x.shape
        x_flaten = x.reshape([h * w, -1])
        # x_flaten = self.batch_normalzation1(x_flaten)
        # x_flaten=self.lin2(x_flaten)
        # x_flaten = self.batch_normalzation1(x_flaten)
        supX = torch.mm(self.norm_col_Q.t(), x_flaten)
        supX = self.hgc1(supX, self.A)
        supX = self.batch_normalzation2(supX)
        supX = F.relu(supX)
        supX = F.dropout(supX, self.dropout)
        supX = self.hgc2(supX, self.A)
        supX = self.batch_normalzation3(supX)
        supX = F.relu(supX)
        supX=self.lin(supX)
        Y = torch.matmul(self.Q, supX)
        return F.softmax(Y, -1)

class SAHGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, alph,
                 ):
        super(SAHGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.alph = alph
        layers_count = 2


        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(self.channel , 128, kernel_size=3))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))


        self.HGNN_Branch = HGCN(self.channel,128, 64,self.Q,self.A,dropout=0.)
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))  # 128

    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape
        clean_x = x

        hx = clean_x
        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q

        H = clean_x
        HGNN_result= self.HGNN_Branch(H)

        Y = self.alph * CNN_result + (1 - self.alph) * HGNN_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y

