import os, sys
import time
import copy
import matplotlib
import random
matplotlib.use('SVG')
import torch
import math
import warnings
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
import setproctitle  
setproctitle.setproctitle("traffic_gene@zsy")
#os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'

use_cuda = torch.cuda.is_available()
gpu =0
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data)
        self.node_embedding = np.load('/data5/zhangshiyuan/urban_data/shanghai/node_embedding/node_embedding_KL_log.npz')['node_embedding']
        try:
            self.bs_id = self.data['bs_id_kge']
            self.bs_record = torch.from_numpy(self.data['bs_record_kge'].astype(np.float32)).reshape(self.bs_id.shape[0],1,LENGTH)
            #self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
            self.kge = torch.from_numpy(self.node_embedding.astype(np.float32))/-400.0
            self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
            self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
            self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
            self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
            self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
            self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
            self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'].astype(np.float32))
            self.hours_in_weekday_patterns = torch.from_numpy(self.data['hours_in_weekday_patterns'].astype(np.float32))
            self.hours_in_weekend_patterns = torch.from_numpy(self.data['hours_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_patterns = torch.from_numpy(self.data['days_in_weekday_patterns'].astype(np.float32))
            self.days_in_weekend_patterns = torch.from_numpy(self.data['days_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_residual_patterns = torch.from_numpy(self.data['days_in_weekday_residual_patterns'].astype(np.float32))
            self.days_in_weekend_residual_patterns = torch.from_numpy(self.data['days_in_weekend_residual_patterns'].astype(np.float32))

        except:
            self.bs_id = self.data['bs_id']
            self.bs_record = torch.from_numpy(self.data['bs_record'].astype(np.float32)).reshape(self.bs_id.shape[0],1,LENGTH)
            #self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
            self.kge = torch.from_numpy(self.node_embedding.astype(np.float32))/-400.0
            self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
            self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
            self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
            self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
            self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
            self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
            self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'].astype(np.float32))
            self.hours_in_weekday_patterns = torch.from_numpy(self.data['hours_in_weekday_patterns'].astype(np.float32))
            self.hours_in_weekend_patterns = torch.from_numpy(self.data['hours_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_patterns = torch.from_numpy(self.data['days_in_weekday_patterns'].astype(np.float32))
            self.days_in_weekend_patterns = torch.from_numpy(self.data['days_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_residual_patterns = torch.from_numpy(self.data['days_in_weekday_residual_patterns'].astype(np.float32))
            self.days_in_weekend_residual_patterns = torch.from_numpy(self.data['days_in_weekend_residual_patterns'].astype(np.float32))
			
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        bs_id = self.bs_id[idx]
        bs_record = self.bs_record[idx]
        kge = self.kge[idx]
        hours_in_weekday = self.hours_in_weekday[idx]
        hours_in_weekend = self.hours_in_weekend[idx]
        days_in_weekday = self.days_in_weekday[idx]
        days_in_weekend = self.days_in_weekend[idx]
        return bs_id, bs_record, kge, hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend
    def __len__(self):
        return self.bs_id.shape[0] 

gpu=0
torch.manual_seed(5)
use_cuda = torch.cuda.is_available()

NOISE_SIZE = 32
KGE_SIZE = 32
SHAPE_M0 = 4
SHAPE = [ (4, 7*24*SHAPE_M0), (4*7, 24*SHAPE_M0), (4*7*24, SHAPE_M0) ]
LENGTH = 672
BATCH_SIZE = 256
BATCH_FIRST = False 

save_dir_head = "./generated_data_0426_gd10_gkl0.0279"
EXP_NOISE = True
ACT = True
USE_KGE = True#False
TimeList = []
D_costList = []
G_costList = []
sparsityList = []
WDList = []
dst_list = []
dataset=MyDataset('bs_record_energy_normalized_sampled.npz')#('D:\实验室\物联网云平台\\traffic_generation\\feature\\data_train.npz')
gene_size = 1024
DATASET_SIZE = len(dataset)
hours_in_weekday_patterns = dataset.hours_in_weekday_patterns
hours_in_weekend_patterns = dataset.hours_in_weekend_patterns
days_in_weekday_patterns = dataset.days_in_weekday_patterns
days_in_weekend_patterns = dataset.days_in_weekend_patterns
days_in_weekday_residual_patterns = dataset.days_in_weekday_residual_patterns
days_in_weekend_residual_patterns = dataset.days_in_weekend_residual_patterns
if use_cuda:
    hours_in_weekday_patterns = hours_in_weekday_patterns.cuda(gpu)
    hours_in_weekend_patterns = hours_in_weekend_patterns.cuda(gpu)
    days_in_weekday_patterns = days_in_weekday_patterns.cuda(gpu)
    days_in_weekend_patterns = days_in_weekend_patterns.cuda(gpu)
    days_in_weekday_residual_patterns = days_in_weekday_residual_patterns.cuda(gpu)
    days_in_weekend_residual_patterns = days_in_weekend_residual_patterns.cuda(gpu)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

dropout = 0.3
num_layers = 6
nhead = [4, 2, 1]
LAMBDA = 10


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        return x.contiguous()

class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        #return (torch.exp(x) - 1) + 1.0
        #print('soft_exponential', 'alpha', self.alpha)
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
#            print('sssss', (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha)
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha
            
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, padding_mode='circular'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation))
        self.chomp1 = Chomp1d(padding/2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation))
        self.chomp2 = Chomp1d(padding/2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        #out_0 = self.conv1(x)
        #out_1 = self.dropout1(self.relu1(self.chomp1(out_0)))
        #out_2 = self.conv2(out_1)
        #out = self.dropout2(self.relu2(self.chomp2(out_2)))
        index_2 = int((out.shape[2]-x.shape[2])/2)
        out = out[:,:,index_2:index_2+x.shape[2]]
        res = x if self.downsample is None else self.downsample(x)
        #print(x.shape, out.shape, self.downsample, res.shape)#out_0.shape, out_1.shape, out_2.shape, out.shape, 
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_size_list=''):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            if dilation_size_list:
                dilation_size = dilation_size_list[i]
            else:
                dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = int(672/kernel_size) if kernel_size*dilation_size>672 else dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     #padding=kernel_size*dilation_size*2, dropout=dropout)]
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        #print('xx', x.shape)
        return self.network(x)

class MLPNet(nn.Module):
    def __init__(self, dim_list, dropout=0.5):
        super(MLPNet, self).__init__()
        layers = []
        num_layers = len(dim_list) - 1
        for i in range(num_layers):
            layers += [nn.Linear(dim_list[i], dim_list[i+1])]
        layers += [nn.ReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GeneratorP_HD0922(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(GeneratorP_HD0922, self).__init__()
        self.linear =  nn.Linear(NOISE_SIZE+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.linear_4hd =  nn.Linear(NOISE_SIZE+KGE_SIZE, 32)       
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hd_patterns = hd_dataset

        if SHAPE_M0 > 1:
            self.linear_out = nn.Linear(SHAPE_M0, 1)


    def forward(self, x):
        BZ = x.shape[0]
        x_p = self.softmax(self.linear_4hd(x))
        hours_in_day = self.relu(x_p@self.hd_patterns)

        return 0., hours_in_day, x_p, 0.
		
class GeneratorP_HD0925(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(GeneratorP_HD0925, self).__init__()
        self.linear_4hd =  nn.Linear(32+KGE_SIZE, 32)       
        self.linear_4hdres = nn.Linear(32+KGE_SIZE, 24)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.hd_patterns = hd_dataset

    def forward(self, x):
        BZ = x.shape[0]
#        print(x.shape)
        x_p = self.softmax(self.linear_4hd(x[:, 32:]))
        hours_in_day = self.gelu(x_p@self.hd_patterns + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))

        return 0., hours_in_day, x_p, 0.

class GeneratorP_DW0925(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(GeneratorP_DW0925, self).__init__()
        self.linear_4dw =  nn.Linear(NOISE_SIZE+KGE_SIZE, 32)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hd_patterns = hd_dataset
        self.hd_patterns.requires_grad = True
        self.dwr_patterns = dwr_dataset
        self.dwr_patterns.requires_grad = True

        if SHAPE_M0 > 1:
            self.linear_out = nn.Linear(SHAPE_M0, 1)


    def forward(self, x):
        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.linear_4dw(x))
        days_in_week_residual = self.tanh(days_in_week_residual@self.dwr_patterns)
        return 0., hours_in_day, days_in_week_residual, 0.

class LightGenerator(nn.Module):
    def __init__(self, input_size=20, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10):
        super(LightGenerator, self).__init__()
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*len(kernel_size), num_channels[-1])
        self.init_weights()
        #self.relu = nn.ReLU()
        self.soft_exponential = soft_exponential(num_channels[-1], alpha = 1.0)
        self.tanh = nn.Tanh()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, kge):
        kge = self.linear_kge(kge)
        x = torch.cat((x, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, x.size(2))), 1)
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2)
        return self.tanh(y)#, y_d, y_w, y_m, kge

class GeneratorP_ALL0924(nn.Module):#ALL & ALL_non_pre
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL0924, self).__init__()
        #self.linear =  nn.Linear(32+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.linear_4hd =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4dw =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4wm =  nn.Linear(32, 10*672)

        self.lg = LightGenerator()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hd_patterns = hd_dataset
        self.hd_patterns.requires_grad = True
        self.dwr_patterns = dwr_dataset
        self.dwr_patterns.requires_grad = True

        if SHAPE_M0 > 1:
            self.linear_out = nn.Linear(SHAPE_M0, 1)


    def forward(self, x):
        BZ = x.shape[0]
        hours_in_day = self.softmax(self.linear_4hd(x))
        hours_in_day = self.relu(hours_in_day@self.hd_patterns)
        days_in_week_residual = self.softmax(self.linear_4dw(x))
        days_in_week_residual = self.tanh(days_in_week_residual@self.dwr_patterns)
        noise4wm = self.linear_4wm(x[:,0:32]).reshape(BZ, 10, 672)
        weeks_in_month_residual = self.lg(noise4wm, x[:, 32:])
        tfc = hours_in_day.repeat(1, 4*7).reshape(BZ,1,-1) + days_in_week_residual.repeat(1, 4).reshape(BZ,1,-1) + weeks_in_month_residual
        return tfc, hours_in_day, days_in_week_residual, weeks_in_month_residual

class GeneratorP_ALL0925(nn.Module):#ALL_sigmoid_out & ALL_relu_out_exp_noise_in
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL0925, self).__init__()
        self.linear =  nn.Linear(32+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.linear_4hd =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4dw =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4wm =  nn.Linear(32, 10*672)

        self.lg = LightGenerator()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hd_patterns = hd_dataset
        self.hd_patterns.requires_grad = True
        self.dwr_patterns = dwr_dataset
        self.dwr_patterns.requires_grad = True

        self.linear_out = nn.Linear(3, 1)


    def forward(self, x):
        BZ = x.shape[0]
        hours_in_day = self.softmax(self.linear_4hd(x))
        hours_in_day = self.relu(hours_in_day@self.hd_patterns)
        days_in_week_residual = self.softmax(self.linear_4dw(x))
        days_in_week_residual = self.tanh(days_in_week_residual@self.dwr_patterns)
        noise4wm = self.linear_4wm(x[:,0:32]).reshape(BZ, 10, 672)
        weeks_in_month_residual = self.lg(noise4wm, x[:, 32:])
        tfc = hours_in_day.repeat(1, 4*7).reshape(BZ,1,-1) + days_in_week_residual.repeat(1, 4).reshape(BZ,1,-1) + weeks_in_month_residual
        return self.relu(tfc), hours_in_day, days_in_week_residual, weeks_in_month_residual

class GeneratorP_ALL0926(nn.Module):#ALL_linear_sigmoid_out
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL0926, self).__init__()
        self.linear =  nn.Linear(32+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.linear_4hd =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4dw =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4wm =  nn.Linear(32, 10*672)

        self.lg = LightGenerator()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hd_patterns = hd_dataset
        self.hd_patterns.requires_grad = True
        self.dwr_patterns = dwr_dataset
        self.dwr_patterns.requires_grad = True

        self.linear_out = nn.Linear(3, 1)


    def forward(self, x):
        BZ = x.shape[0]
        hours_in_day = self.softmax(self.linear_4hd(x))
        hours_in_day = self.relu(hours_in_day@self.hd_patterns)
        days_in_week_residual = self.softmax(self.linear_4dw(x))
        days_in_week_residual = self.tanh(days_in_week_residual@self.dwr_patterns)
        noise4wm = self.linear_4wm(x[:,0:32]).reshape(BZ, 10, 672)
        weeks_in_month_residual = self.lg(noise4wm, x[:, 32:])
        #print(weeks_in_month_residual.shape)
        tfc = self.linear_out(torch.cat(\
              (hours_in_day.repeat(1, 4*7).reshape(BZ, 1, 672),\
               days_in_week_residual.repeat(1, 4).reshape(BZ, 1, 672),\
               weeks_in_month_residual), 1).permute(0,2,1)).permute(0,2,1)
        return self.sigmoid(tfc), hours_in_day, days_in_week_residual, weeks_in_month_residual

class GeneratorP_ALL(nn.Module):#ALL_linear_relu_out & ALL_linear_exp_noise_in
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL, self).__init__()
        self.linear =  nn.Linear(32+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.linear_4hd =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4dw =  nn.Linear(32+KGE_SIZE, 32)
        self.linear_4wm =  nn.Linear(32, 10*672)

        self.lg = LightGenerator()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.hd_patterns = hd_dataset
        self.hd_patterns.requires_grad = True
        self.dwr_patterns = dwr_dataset
        self.dwr_patterns.requires_grad = True

        self.linear_out = nn.Linear(3, 1)


    def forward(self, x):
        BZ = x.shape[0]
        hours_in_day = self.softmax(self.linear_4hd(x))
        hours_in_day = self.relu(hours_in_day@self.hd_patterns)
        days_in_week_residual = self.softmax(self.linear_4dw(x))
        days_in_week_residual = self.tanh(days_in_week_residual@self.dwr_patterns)
        noise4wm = self.linear_4wm(x[:,0:32]).reshape(BZ, 10, 672)
        weeks_in_month_residual = self.lg(noise4wm, x[:, 32:])
        #print(weeks_in_month_residual.shape)
        tfc = self.linear_out(torch.cat(\
              (hours_in_day.repeat(1, 4*7).reshape(BZ, 1, 672),\
               days_in_week_residual.repeat(1, 4).reshape(BZ, 1, 672),\
               weeks_in_month_residual), 1).permute(0,2,1)).permute(0,2,1)
        return tfc, hours_in_day, days_in_week_residual, weeks_in_month_residual

class GeneratorP_HD(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = hours_in_weekday_patterns):
        super(GeneratorP_HD, self).__init__()
        self.linear_4hd =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num) if USE_KGE else  \
                           nn.Linear(NOISE_SIZE, pattern_num)		
#        self.linear_4hdres = nn.Linear(32+KGE_SIZE, 24)
        self.norm_4hd = nn.LayerNorm(pattern_num)
#        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.hd_patterns = patterns

    def forward(self, x):
#        BZ = x.shape[0]
#        print(x.shape)
        x_p = self.softmax(self.norm_4hd(self.linear_4hd(x))) if USE_KGE else  \
              self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))#[:, 32:])))
        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
#        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        return hours_in_day

class GeneratorP_HD_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = hours_in_weekday_patterns):
        super(GeneratorP_HD_SUM, self).__init__()
        self.linear_4hd =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num) if USE_KGE else  \
                           nn.Linear(NOISE_SIZE, pattern_num)		
#        self.linear_4hdres = nn.Linear(32+KGE_SIZE, 24)
        self.norm_4hd = nn.LayerNorm(pattern_num)
#        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.hd_patterns = patterns

    def forward(self, x):
#        BZ = x.shape[0]
#        print(x.shape)
        x_p = self.softmax(self.norm_4hd(self.linear_4hd(x))) if USE_KGE else  \
              self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))#[:, 32:])))
        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
#        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        return hours_in_day
		
class GeneratorP_DWD(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = days_in_weekday_residual_patterns):
        super(GeneratorP_DWD, self).__init__()
        self.linear_4dw =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num)
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.dwr_patterns = patterns
        self.dwr_patterns.requires_grad = True

    def forward(self, x):
#        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x)))
        if ACT:
            days_in_week_residual = self.tanh(self.norm(days_in_week_residual@self.dwr_patterns))
        else:
            days_in_week_residual = self.norm(days_in_week_residual@self.dwr_patterns)
        return days_in_week_residual#_weekday

class GeneratorP_DWE(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = days_in_weekend_residual_patterns):
        super(GeneratorP_DWE, self).__init__()
        self.linear_4dw =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num)
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.dwr_patterns = patterns
        self.dwr_patterns.requires_grad = True

    def forward(self, x):
#        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x)))
        if ACT:
            days_in_week_residual = self.tanh(self.norm(days_in_week_residual@self.dwr_patterns))
        else:
            days_in_week_residual = self.norm(days_in_week_residual@self.dwr_patterns)
        return days_in_week_residual#_weekend

class GeneratorP_DW_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = days_in_weekday_residual_patterns):
        super(GeneratorP_DW_SUM, self).__init__()
        self.linear_4dw =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num) if USE_KGE else  \
                           nn.Linear(NOISE_SIZE, pattern_num)		
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.dwr_patterns = patterns
        self.dwr_patterns.requires_grad = True

    def forward(self, x, hours_in_day):
#        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x))) if USE_KGE else  \
                                self.softmax(self.norm_4dw(self.linear_4dw(x[:, 0:32])))
        days_in_week_residual = self.norm(days_in_week_residual@self.dwr_patterns)
        days_in_week = days_in_week_residual + hours_in_day.repeat(1, int(days_in_week_residual.shape[1]/hours_in_day.shape[1]))
        return self.act(days_in_week)#_weekend

class GeneratorP_WM(nn.Module):
    def __init__(self, input_size=20, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10):
        super(GeneratorP_WM, self).__init__()
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.norm_kge = nn.LayerNorm(kge_squeeze_size)
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear_4wm = nn.Linear(32, kge_squeeze_size*672)
        self.norm_4wm = nn.LayerNorm(kge_squeeze_size*672)
        self.norm = nn.LayerNorm([1,672])

        self.linear = nn.Linear(num_channels[-1]*len(kernel_size), num_channels[-1])
        self.init_weights()
        #self.relu = nn.ReLU()
        #self.soft_exponential = soft_exponential(num_channels[-1], alpha = 1.0)
        self.tanh = nn.Tanh()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        BZ = x.shape[0]
        kge = self.norm_kge(self.linear_kge(x[:, 32:]))
        x = self.norm_4wm(self.linear_4wm(x[:,0:32])).reshape(BZ, -1, 672)
        x = torch.cat((x, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, x.size(2))), 1)
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2)
        if ACT:
            return self.tanh(self.norm(y))#, y_d, y_w, y_m, kge
        else:
            return self.norm(y)#, y_d, y_w, y_m, kge

class GeneratorP_WM_SUM(nn.Module):
    def __init__(self, input_size=20, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10, activation="relu"):
        super(GeneratorP_WM_SUM, self).__init__()
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.norm_kge = nn.LayerNorm(kge_squeeze_size)
        input_size = 20 if USE_KGE else 10
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear_4wm = nn.Linear(32, kge_squeeze_size*672)
        self.norm_4wm = nn.LayerNorm(kge_squeeze_size*672)
        self.norm = nn.LayerNorm([1,672])
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 

        self.linear = nn.Linear(num_channels[-1]*len(kernel_size), num_channels[-1])
        self.init_weights()
        #self.relu = nn.ReLU()
        #self.soft_exponential = soft_exponential(num_channels[-1], alpha = 1.0)
        self.tanh = nn.Tanh()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, days_in_week):
        BZ = x.shape[0]
        kge = self.norm_kge(self.linear_kge(x[:, 32:]))
        x = self.norm_4wm(self.linear_4wm(x[:,0:32])).reshape(BZ, -1, 672)
        x = torch.cat((x, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, x.size(2))), 1) if USE_KGE else x
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.norm(self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2))
        y = y + days_in_week.repeat(1, 4).reshape(BZ,1,-1)
        return self.act(y.squeeze(1))#, y_d, y_w, y_m, kge

class GeneratorP_ALL_LN(nn.Module):#"./generated_data_0926_residual_Partly_exp_noise"
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL_LN, self).__init__()
        self.generator_hd =  GeneratorP_HD()
        self.generator_dwd =  GeneratorP_DWD()
        self.generator_dwe =  GeneratorP_DWE()
        self.generator_wm =  GeneratorP_WM()
        #self.norm = nn.LayerNorm(
        #self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax(dim=1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        BZ = x.shape[0]
        hours_in_day = self.generator_hd(x)
        days_in_week_residual_weekday = self.generator_dwd(x)
        days_in_week_residual_weekend = self.generator_dwe(x)
        days_in_week_residual = torch.cat((days_in_week_residual_weekday, days_in_week_residual_weekend), 1)
        weeks_in_month_residual = self.generator_wm(x)
        tfc = hours_in_day.repeat(1, 4*7).reshape(BZ,1,-1) + days_in_week_residual.repeat(1, 4).reshape(BZ,1,-1) + weeks_in_month_residual
        return tfc, hours_in_day, days_in_week_residual, weeks_in_month_residual

class GeneratorP_ALL_LN_Matrioska(nn.Module):#
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL_LN_Matrioska, self).__init__()
        self.generator_hdd =  GeneratorP_HD_SUM(patterns = hours_in_weekday_patterns, activation=activation)
        self.generator_hde =  GeneratorP_HD_SUM(patterns = hours_in_weekend_patterns, activation=activation)
        self.generator_dwd =  GeneratorP_DW_SUM(patterns = days_in_weekday_residual_patterns, activation=activation)
        self.generator_dwe =  GeneratorP_DW_SUM(patterns = days_in_weekend_residual_patterns, activation=activation)
        self.generator_wm =  GeneratorP_WM_SUM(activation=activation)
        #self.norm = nn.LayerNorm(
        #self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax(dim=1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        BZ = x.shape[0]
        hours_in_weekday = self.generator_hdd(x)
        hours_in_weekend = self.generator_hde(x)
        days_in_weekday = self.generator_dwd(x, hours_in_weekday)
        days_in_weekend = self.generator_dwe(x, hours_in_weekend)
        days_in_week = torch.cat((days_in_weekday, days_in_weekend), 1)
        tfc = self.generator_wm(x, days_in_week)
#        tfc = hours_in_day.repeat(1, 4*7).reshape(BZ,1,-1) + days_in_week_residual.repeat(1, 4).reshape(BZ,1,-1) + weeks_in_month_residual
        return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, tfc

class Discriminator_HD(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="sigmoid", hidden_size=8, patterns=hours_in_weekday_patterns):
        super(Discriminator_HD, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.hd_patterns = patterns
    def forward(self, x, kge):        
#        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x_p = x@self.hd_patterns.T
        x = self.act(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class Discriminator_DW(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="sigmoid", hidden_size=8, patterns = days_in_weekday_patterns):
        super(Discriminator_DW, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.dwr_patterns = patterns
    def forward(self, x, kge):        
#        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x_p = x@self.dwr_patterns.T
        x = self.act(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class Discriminator_DWD(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", hidden_size=8):
        super(Discriminator_DWD, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.dwr_patterns = days_in_weekday_patterns
    def forward(self, x, kge):        
#        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x_p = x@self.dwr_patterns.T
        x = self.sigmoid(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class Discriminator_DWE(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", hidden_size=8):
        super(Discriminator_DWE, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.dwr_patterns = days_in_weekend_patterns
    def forward(self, x, kge):        
#        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x_p = x@self.dwr_patterns.T
        x = self.sigmoid(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class DiscriminatorTCN(nn.Module):
    def __init__(self, input_size=1, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10, activation="sigmoid"):
        super(DiscriminatorTCN, self).__init__()
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*len(kernel_size)+2+kge_squeeze_size, 1)
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.init_weights()
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, kge):
        x = x.unsqueeze(1)
        x_mean = x.mean(2)
        x_min = x.min(2).values
        y_d = self.tcn_d(x)[:,:,-1]
        y_w = self.tcn_w(x)[:,:,-1]
        y_m = self.tcn_m(x)[:,:,-1]
#        print(y_d.shape, y_w.shape, y_m.shape, x_mean.shape, x_min.shape, self.linear_kge(kge).shape)
        y = self.linear(torch.cat((y_d, y_w, y_m, x_mean, x_min, self.linear_kge(kge)), 1))
        return self.act(y)
		
class DiscriminatorTCN0(nn.Module):
    def __init__(self, input_size=1, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=5):
        super(DiscriminatorTCN0, self).__init__()
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*len(kernel_size)+2+kge_squeeze_size, 1)
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, kge):
#        x = x.squeeze(1)
#        print(x.shape)
        x_mean = x.mean(2)#.unsqueeze(1)
        x_min = x.min(2).values#.unsqueeze(1)
        y_d = self.tcn_d(x)[:,:,-1]
        y_w = self.tcn_w(x)[:,:,-1]
        y_m = self.tcn_m(x)[:,:,-1]
        y = self.linear(torch.cat((y_d, y_w, y_m, x_mean, x_min, self.linear_kge(kge)), 1))
        return self.sigmoid(y)
		
class FFTGenerator(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(FFTGenerator, self).__init__()
        self.linear =  nn.Linear(NOISE_SIZE+KGE_SIZE, SHAPE[0][0]*SHAPE[0][1])
        self.block_month = TransBlock(SHAPE[0][1], nhead[0], dim_feedforward, dropout, activation, num_layers)
        self.block_week = TransBlock(SHAPE[1][1], nhead[1], dim_feedforward, dropout, activation, num_layers)
        self.block_day = TransBlock(SHAPE[2][1], nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.blocks = nn.ModuleList([ self.block_month, self.block_week, self.block_day ])
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        if SHAPE_M0 > 1:
            self.linear_out = nn.Linear(SHAPE_M0, 1)


    def forward(self, x):
        BZ = x.shape[0]
        x = self.linear(x)
        for index, blk in enumerate(self.blocks):
            x = blk(x.view((SHAPE[index][0], BZ, SHAPE[index][1])))
        #print(x.max(), x.min())
        f = torch.rfft(x.permute(2,1,0), 1)[:,:,1:,:]
        x = torch.cat((f[:,:,:,0], f[:,:,:,1]), 2).permute(2,1,0)
#        print(x.max(), x.min())
        if SHAPE_M0 > 1:
            x = self.linear_out(x)      
#        print(x.max(), x.min())
        return self.gelu(x), f
#        return torch.abs(x), f
		
class FFTDiscriminator(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(FFTDiscriminator, self).__init__()
        self.linear =  nn.Linear(int(SHAPE[0][0]*SHAPE[0][1]/SHAPE_M0*8) + KGE_SIZE, 2048)
        self.block_month = TransBlock(int(SHAPE[0][1]/SHAPE_M0*8), nhead[0], dim_feedforward, dropout, activation, num_layers)
        self.block_week = TransBlock(int(SHAPE[1][1]/SHAPE_M0*4), nhead[1], dim_feedforward, dropout, activation, num_layers)
        self.block_day = TransBlock(int(SHAPE[2][1]/SHAPE_M0*2), nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.blocks = nn.ModuleList([ self.block_day, self.block_week, self.block_month ])
        self.pe_month = PositionalEncoding(int(SHAPE[0][1]/SHAPE_M0*4), dropout)#TemporalEncoding
        self.pe_week = PositionalEncoding(int(SHAPE[1][1]/SHAPE_M0*2), dropout)#TemporalEncoding
        self.pe_day = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)#TemporalEncoding
        #self.pe = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)
        self.pes = nn.ModuleList([ self.pe_day, self.pe_week, self.pe_month ])
		
    def forward(self, x, kge):        
        BZ = kge.shape[0]
        #print(x.shape)
        #x = self.pe(x)
        f = torch.rfft(x.permute(2,1,0), 1)[:,:,1:,:]
        x = torch.cat((f[:,:,:,0], f[:,:,:,1]), 2).permute(2,1,0)
        #print(x.shape)
        for index, blk in enumerate(self.blocks):
            x = self.pes[index](x.reshape((SHAPE[len(SHAPE)-1-index][0], BZ, -1)))
            x = blk(x)#int(SHAPE[len(SHAPE)-1-index][1]/SHAPE_M0))))
        #    xx = 
        x = self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))      
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x
class Discriminator_HD2(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator_HD2, self).__init__()
        self.linear0 =  nn.Linear(32+32, 8)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(8, 1)	
        self.hd_patterns = hd_dataset
    def forward(self, x, kge):        
        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x_p = x@self.hd_patterns.T
        x = self.sigmoid(self.linear1(self.sigmoid(self.linear0(torch.cat((x_p, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x, x_p

class Discriminator_HD0(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator_HD0, self).__init__()
        self.linear0 =  nn.Linear(24+32, 8)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(8, 1)	
    def forward(self, x, kge):        
        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x = self.sigmoid(self.linear1(self.sigmoid(self.linear0(torch.cat((x, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class Discriminator_HD1(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator_HD1, self).__init__()
        self.block_day = TransBlock(int(SHAPE[2][1]/SHAPE_M0*2), nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.pe_day = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)#TemporalEncoding
        self.linear =  nn.Linear(48 + KGE_SIZE, 1)

    def forward(self, x, kge):        
        BZ = kge.shape[0]
        x = self.pe_day(x.reshape((24, BZ, -1)))     
        x = self.block_day(x)
#        print(x.shape)
        x = self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))
        return x

class Discriminator_DW1(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator_DW1, self).__init__()
        self.linear0 =  nn.Linear(32+32, 8)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(8, 1)	
        self.dwr_patterns = dwr_dataset
    def forward(self, x, kge):        
        BZ = kge.shape[0]
#        print(x.shape, kge.shape)
        x = x@self.dwr_patterns.T
        x = self.sigmoid(self.linear1(self.sigmoid(self.linear0(torch.cat((x, kge), 1)))))    
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class Discriminator_DW0(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator_DW0, self).__init__()
        self.linear0 =  nn.Linear(24*7+32, 7)
        self.sigmoid = nn.Sigmoid()
        self.linear1 =  nn.Linear(7, 1)		
    def forward(self, x, kge):        
        x = self.sigmoid(self.linear1(self.sigmoid(self.linear0(torch.cat((x, kge), 1)))))      
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x
		
class Discriminator(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(Discriminator, self).__init__()
        self.linear =  nn.Linear(int(SHAPE[0][0]*SHAPE[0][1]/SHAPE_M0*8) + KGE_SIZE, 2048)
        self.block_month = TransBlock(int(SHAPE[0][1]/SHAPE_M0*8), nhead[0], dim_feedforward, dropout, activation, num_layers)
        self.block_week = TransBlock(int(SHAPE[1][1]/SHAPE_M0*4), nhead[1], dim_feedforward, dropout, activation, num_layers)
        self.block_day = TransBlock(int(SHAPE[2][1]/SHAPE_M0*2), nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.blocks = nn.ModuleList([ self.block_day, self.block_week, self.block_month ])
        self.pe_month = PositionalEncoding(int(SHAPE[0][1]/SHAPE_M0*4), dropout)#TemporalEncoding
        self.pe_week = PositionalEncoding(int(SHAPE[1][1]/SHAPE_M0*2), dropout)#TemporalEncoding
        self.pe_day = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)#TemporalEncoding
        #self.pe = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)
        self.pes = nn.ModuleList([ self.pe_day, self.pe_week, self.pe_month ])
		
    def forward(self, x, kge):        
        BZ = kge.shape[0]
        #xx = torch.tensor()
        #x = self.pe(x)
        for index, blk in enumerate(self.blocks):
            x = self.pes[index](x.reshape((SHAPE[len(SHAPE)-1-index][0], BZ, -1)))
            x = blk(x)#int(SHAPE[len(SHAPE)-1-index][1]/SHAPE_M0))))
        #    xx = 
        x = self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))      
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class MultiDiscriminator(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(MultiDiscriminator, self).__init__()
        self.linear =  nn.Linear(int(SHAPE[0][0]*SHAPE[0][1]/SHAPE_M0*8) + KGE_SIZE, 2048)
        self.block_month = TransBlock(int(SHAPE[0][1]/SHAPE_M0*8), nhead[0], dim_feedforward, dropout, activation, num_layers)
        self.block_week = TransBlock(int(SHAPE[1][1]/SHAPE_M0*4), nhead[1], dim_feedforward, dropout, activation, num_layers)
        self.block_day = TransBlock(int(SHAPE[2][1]/SHAPE_M0*2), nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.blocks = nn.ModuleList([ self.block_day, self.block_week, self.block_month ])
        self.pe_month = MultiTemporalEncoding(int(SHAPE[0][1]/SHAPE_M0*4), dropout)#PositionalEncoding
        self.pe_week = MultiTemporalEncoding(int(SHAPE[1][1]/SHAPE_M0*2), dropout)#PositionalEncoding
        self.pe_day = MultiTemporalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)#PositionalEncoding
        #self.pe = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)
        self.pes = nn.ModuleList([ self.pe_day, self.pe_week, self.pe_month ])
		
    def forward(self, x, kge):        
        BZ = kge.shape[0]
        #xx = torch.tensor()
        #x = self.pe(x)
        for index, blk in enumerate(self.blocks):
            x = self.pes[index](x.reshape((SHAPE[len(SHAPE)-1-index][0], BZ, -1)))
            x = blk(x)#int(SHAPE[len(SHAPE)-1-index][1]/SHAPE_M0))))
        #    xx = 
        x = self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))      
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x

class BiDiscriminator(nn.Module):
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.1, activation="relu", num_layers=6):
        super(BiDiscriminator, self).__init__()
        self.linear =  nn.Linear(int(SHAPE[0][0]*SHAPE[0][1]/SHAPE_M0*2) + KGE_SIZE, 2048)
        self.block_month = TransBlock(int(SHAPE[0][1]/SHAPE_M0*2), nhead[0], dim_feedforward, dropout, activation, num_layers)
        self.block_week = TransBlock(int(SHAPE[1][1]/SHAPE_M0*2), nhead[1], dim_feedforward, dropout, activation, num_layers)
        self.block_day = TransBlock(int(SHAPE[2][1]/SHAPE_M0*2), nhead[2], dim_feedforward, dropout, activation, num_layers)
        self.blocks = nn.ModuleList([ self.block_day, self.block_week, self.block_month ])
        self.pe = PositionalEncoding(int(SHAPE[2][1]/SHAPE_M0), dropout)
    def forward(self, x, kge):        
        BZ = kge.shape[0]
        #xx = torch.tensor()
        x = self.pe(x)
        for index, blk in enumerate(self.blocks):
            x = blk(x.view((SHAPE[len(SHAPE)-1-index][0], BZ, int(SHAPE[len(SHAPE)-1-index][1]/SHAPE_M0*2))))
        #    xx = 
        x = self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))      
        #x = self.relu(self.linear3(self.linear2(self.linear1(self.linear(torch.cat((x.permute(1,0,2).reshape(BZ, -1), kge), 1))))))     
        return x



def calc_gradient_penalty(netD, real_data, fake_data, kge):
    #use_cuda = 0
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)

    if use_cuda:
        alpha = alpha.cuda(gpu)#.to('cuda:6')# if use_cuda else alpha
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

#    if use_cuda:
#        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, kge)
    try:
        disc_interpolates = disc_interpolates[0]
    except:
        pass;

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_data(netG, kge, gene_size=128):
    #noise_attribute = kge#torch.rand(gene_size,12)
    noise = torch.randn(gene_size, NOISE_SIZE)#torch.randn(BATCH_SIZE, NOISE_SIZE, dim_list_g[0])
    if use_cuda:
        #noise_attribute = noise_attribute.cuda(gpu)
        noise = noise.cuda(gpu) 
        kge = kge.cuda(gpu)
    #noisev = autograd.Variable(noise, volatile=True)
    #noise_kge = torch.cat((noise, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, LENGTH)), 1)
    output, hours_in_day, days_in_week, weeks_in_month_residual = netG(torch.cat((noise, kge), 1))#, kge)
    #print(output.max(), output.min())
    #return output, kge, output_d, output_w, output_m, z
    return output, hours_in_day, days_in_week, weeks_in_month_residual
	

'''
netG = GeneratorP(nhead=nhead, dropout=dropout, num_layers=num_layers)
netG_HD = GeneratorP_HD(nhead=nhead, dropout=dropout, num_layers=num_layers)
netG_DW = GeneratorP_DW(nhead=nhead, dropout=dropout, num_layers=num_layers)
netD = FFTDiscriminator(nhead=nhead, dropout=dropout, num_layers=num_layers)
netD_HD = Discriminator_HD(nhead=nhead, dropout=dropout, num_layers=num_layers)
netD_DW = Discriminator_DW(nhead=nhead, dropout=dropout, num_layers=num_layers)
start_it = 0

TRAIN = False
TRAIN_WM = False
TRAIN_HD = False
TRAIN_DW = False
LOAD_TRAIN_HD = False
LOAD_TRAIN_DW = False
LOAD_TRAIN_WM = False
LOAD_PRE_TRAIN = False

ITERS = 315
CRITIC_ITERS = 5

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
    netD_HD = netD_HD.cuda(gpu)
    netD_DW = netD_DW.cuda(gpu)
    netG_HD = netG_HD.cuda(gpu)
    netG_DW = netG_DW.cuda(gpu)
#    hd_dataset = hd_dataset.cuda(gpu)
#print(netG)
#print(netD)

if LOAD_PRE_TRAIN:
    pretrained_netG = torch.load('./generated_data_0914_trans_gan_residual_PreT/iteration-34/netG',map_location=torch.device('cpu'))
    netG.load_state_dict(pretrained_netG)

LOAD_TRAIN_HD_temp = False
if LOAD_TRAIN_HD_temp:
    pretrained_netG_HD = torch.load('./generated_data_0920_trans_gan_residual_Pred/HD/iteration-289/netG.linear_hours_in_day',map_location=torch.device('cpu'))
    netG.linear_hours_in_day.load_state_dict(pretrained_netG_HD)
    pretrained_netD_HD = torch.load('./generated_data_0920_trans_gan_residual_Pred/HD/iteration-289/netD_HD',map_location=torch.device('cpu'))
    netD_HD.load_state_dict(pretrained_netD_HD)

if TRAIN_HD:
    optimizerD_HD = optim.Adam(netD_HD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG_HD = optim.Adam(netG_HD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
    
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        idx = 0
        for idx, data in enumerate(data_loader):
            
            if True:#(idx < 1500) | (idx%(CRITIC_ITERS*3) == 0):        
                ############################
                # (1) Update D network
                ###########################
                for p in netD_HD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update            
                data_batch, kge_batch, hours_in_day, days_in_week_residual, weeks_in_month_residual= data[1:]
                if not(BATCH_FIRST):
                    data_batch = data_batch.permute(2,0,1)                
                netD_HD.zero_grad()                
                real_data = data_batch
                if use_cuda:
                    #real_data = real_data.cuda(gpu)
                    kge_batch = kge_batch.cuda(gpu)
                    hours_in_day = hours_in_day.cuda(gpu)
                    #days_in_week_residual = days_in_week_residual.cuda(gpu)
                    #weeks_in_month_residual = weeks_in_month_residual.cuda(gpu)
                D_real = netD_HD(hours_in_day, kge_batch)
                D_real = D_real.mean()
                # print D_real
                # TODO: Waiting for the bug fix from pytorch
                D_real.backward(mone)
                
                # generate noise
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                fake_data, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG_HD(torch.cat((noise_batch, kge_batch), 1))
                D_fake = netD_HD(fake_hours_in_day, kge_batch)
                D_fake = D_fake.mean()
                # TODO: Waiting for the bug fix from pytorch
                D_fake.backward(one)
                
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD_HD, hours_in_day, fake_hours_in_day, kge_batch)
                gradient_penalty.backward()
                
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD_HD.step()
                #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if idx%CRITIC_ITERS == 0:
                ############################
                # (2) Update G network
                ###########################
                for p in netD_HD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG_HD.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG_HD(torch.cat((noise_batch, kge_batch), 1))
                G = netD_HD(fake_hours_in_day, kge_batch)
                G = G.mean()
                G.backward(mone)
                G_cost = -G
                optimizerG_HD.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
                #WDList.append(Wasserstein_D.cpu().data.numpy())
            if False:#idx%CRITIC_ITERS == 1:
                ############################
                # (2) Update G network
                ###########################
                for p in netD_HD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG_HD.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_f = netG_HD(torch.cat((noise_batch, kge_batch), 1))
                #print(fake.shape, fake_f.shape)
                sparsity = (torch.sqrt(torch.sum(ones)) - \
                            F.l1_loss(fake_f, zeros, reduction='sum') / \
                            F.mse_loss(fake_f, zeros, reduction='sum')) /\
                            (torch.sqrt(torch.sum(ones)) - 1)
                sparsity.backward()
                optimizerG_DW.step()
                #print(G_cost, D_cost, Wasserstein_D)
                sparsityList.append(sparsity.cpu().data.numpy())
            TimeList.append(time.time() - start_time)
            D_costList.append(D_cost.cpu().data.numpy())
            ##SD_costList.append(SD_cost.cpu().data.numpy())
            WDList.append(Wasserstein_D.cpu().data.numpy())            
            #print(fake.max(), fake.min())
        if iteration % 15 == 4:#True
            save_dir = save_dir_head + '/HD/iteration-' + str(iteration+start_it)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    
            torch.save(netD_HD.state_dict(), os.path.join(save_dir, 'netD_HD'))
            torch.save(netG_HD.linear_4hd.state_dict(), os.path.join(save_dir, 'netG.linear_4hd'))
            torch.save(netG_HD.hd_patterns, os.path.join(save_dir, 'netG.hd_patterns'))
            np.savez(os.path.join(save_dir, 'cost_generated.npz'), \
                     time=np.array(TimeList), \
                     D_cost=np.array(D_costList), \
                     sparsity=np.array(sparsityList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            generated_data, hours_in_day, days_in_week_residual, weeks_in_month_residual = \
                     generate_data(netG_HD, dataset.kge[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
#            generated_data = generated_data.view(gene_size, -1).cpu().detach().numpy()
            #kge_used = kge_gene.cpu().detach().numpy()
            hours_in_day = hours_in_day.view(gene_size, 24).cpu().detach().numpy()
#            days_in_week_residual = days_in_week_residual.view(gene_size, 24*7).cpu().detach().numpy()
#            weeks_in_month_residual = weeks_in_month_residual.view(gene_size, LENGTH).cpu().detach().numpy()
            #generated_data_f = generated_data_f.cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(24, 16))
            n_bins = 100
            line_w = 2
            use_cumulative = -1
            use_log = True
            n_real, bins, patches = ax.hist(real_dataset_HD.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='real', log=use_log, facecolor='g', linewidth=line_w)
            n_gene, bins, patches = ax.hist(hours_in_day.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='gene', log=use_log, facecolor='b', linewidth=line_w)
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('Value')
            ax.set_ylabel('Likelihood of occurrence')
            plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
            plt.close()
            dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)        
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            #generated_data = generated_data, \
            distance = dst, \
            distances = np.array(dst_list), \
            #kge_used = np.array(kge_used), \
            #squeezed_kge = squeezed_kge, \
            hours_in_day = hours_in_day)#, \
            #days_in_week_residual = days_in_week_residual, \
            #weeks_in_month_residual = weeks_in_month_residual)            
			#np.savez(os.path.join(save_dir, 'distance.npz'), distances = np.array(dst_list))
        if iteration % 45 == 4:
            print(G_cost, D_cost, Wasserstein_D, dst)#, sparsity)   
print('HD finished!')			
if LOAD_TRAIN_HD:
    pretrained_netG_HD_linear_4hd = torch.load('./generated_data_0921_trans_gan_residual_Pre/HD/iteration-304/netG.linear_4hd',map_location=torch.device(gpu))
    netG_DW.linear_4hd.load_state_dict(pretrained_netG_HD_linear_4hd)
    netG_DW.hd_patterns = torch.load('./generated_data_0921_trans_gan_residual_Pre/HD/iteration-304/netG.hd_patterns',map_location=torch.device(gpu))
print('HD loaded!')
if TRAIN_DW:
    optimizerD_DW = optim.Adam(netD_DW.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG_DW = optim.Adam(netG_DW.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
    
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        idx = 0
        for idx, data in enumerate(data_loader):
            
            if True:#(idx < 1500) | (idx%(CRITIC_ITERS*3) == 0):        
                ############################
                # (1) Update D network
                ###########################
                for p in netD_DW.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update            
                data_batch, kge_batch, hours_in_day, days_in_week_residual, weeks_in_month_residual= data[1:]
                if not(BATCH_FIRST):
                    data_batch = data_batch.permute(2,0,1)
                
                netD_DW.zero_grad()                
                real_data = data_batch
                if use_cuda:
                    #real_data = real_data.cuda(gpu)
                    kge_batch = kge_batch.cuda(gpu)
                    hours_in_day = hours_in_day.cuda(gpu)
                    days_in_week_residual = days_in_week_residual.cuda(gpu)
                    #weeks_in_month_residual = weeks_in_month_residual.cuda(gpu)
                D_real = netD_DW(days_in_week_residual, kge_batch)
                D_real = D_real.mean()
                # print D_real
                # TODO: Waiting for the bug fix from pytorch
                D_real.backward(mone)
                
                # generate noise
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu)            
                # train with fake
                fake_data, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG_DW(torch.cat((noise_batch, kge_batch), 1))
                D_fake = netD_DW(fake_days_in_week_residual, kge_batch)
                D_fake = D_fake.mean()
                # TODO: Waiting for the bug fix from pytorch
                D_fake.backward(one)
                
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD_DW, days_in_week_residual, fake_days_in_week_residual, kge_batch)
                gradient_penalty.backward()
                
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD_DW.step()
                #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if idx%CRITIC_ITERS == 0:
                ############################
                # (2) Update G network
                ###########################
                for p in netD_DW.parameters():
                    p.requires_grad = False  # to avoid computation
                netG_DW.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG_DW(torch.cat((noise_batch, kge_batch), 1))
                G = netD_DW(fake_days_in_week_residual, kge_batch)
                G = G.mean()
                #detach layers for HD
                fake_hours_in_day.detach_()
                G.backward(mone)
                G_cost = -G
                optimizerG_DW.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
                #WDList.append(Wasserstein_D.cpu().data.numpy())
            if False:#idx%CRITIC_ITERS == 1:
                ############################
                # (2) Update G network
                ###########################
                for p in netD_DW.parameters():
                    p.requires_grad = False  # to avoid computation
                netG_DW.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_f = netG_DW(torch.cat((noise_batch, kge_batch), 1))
                #print(fake.shape, fake_f.shape)
                sparsity = (torch.sqrt(torch.sum(ones)) - \
                            F.l1_loss(fake_f, zeros, reduction='sum') / \
                            F.mse_loss(fake_f, zeros, reduction='sum')) /\
                            (torch.sqrt(torch.sum(ones)) - 1)
                sparsity.backward()
                optimizerG_DW.step()
                #print(G_cost, D_cost, Wasserstein_D)
                sparsityList.append(sparsity.cpu().data.numpy())
            TimeList.append(time.time() - start_time)
            D_costList.append(D_cost.cpu().data.numpy())
            ##SD_costList.append(SD_cost.cpu().data.numpy())
            WDList.append(Wasserstein_D.cpu().data.numpy())            
            #print(fake.max(), fake.min())
        if iteration % 15 == 4:#True
            save_dir = save_dir_head + '/DW/iteration-' + str(iteration+start_it)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    
            torch.save(netD_DW.state_dict(), os.path.join(save_dir, 'netD_DW'))
            torch.save(netG_DW.linear_4hd.state_dict(), os.path.join(save_dir, 'netG.linear_4hd'))
            torch.save(netG_DW.hd_patterns, os.path.join(save_dir, 'netG.hd_patterns'))
            torch.save(netG_DW.linear_4dw.state_dict(), os.path.join(save_dir, 'netG.linear_4dw'))
            torch.save(netG_DW.dwr_patterns, os.path.join(save_dir, 'netG.dwr_patterns'))
            np.savez(os.path.join(save_dir, 'cost_generated.npz'), \
                     time=np.array(TimeList), \
                     D_cost=np.array(D_costList), \
                     sparsity=np.array(sparsityList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            generated_data, hours_in_day, days_in_week_residual, weeks_in_month_residual = \
                     generate_data(netG_DW, dataset.kge[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
#            generated_data = generated_data.view(gene_size, -1).cpu().detach().numpy()
            #kge_used = kge_gene.cpu().detach().numpy()
            hours_in_day = hours_in_day.view(gene_size, 24).cpu().detach().numpy()
            days_in_week_residual = days_in_week_residual.view(gene_size, 24*7).cpu().detach().numpy()
#            weeks_in_month_residual = weeks_in_month_residual.view(gene_size, LENGTH).cpu().detach().numpy()
            #generated_data_f = generated_data_f.cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(24, 16))
            n_bins = 100
            line_w = 2
            use_cumulative = -1
            use_log = True
            n_real, bins, patches = ax.hist(real_dataset_DW.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='real', log=use_log, facecolor='g', linewidth=line_w)
            n_gene, bins, patches = ax.hist(days_in_week_residual.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='gene', log=use_log, facecolor='b', linewidth=line_w)
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('Value')
            ax.set_ylabel('Likelihood of occurrence')
            plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
            plt.close()
            dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)        
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            #generated_data = generated_data, \
            distance = dst, \
            distances = np.array(dst_list), \
            #kge_used = np.array(kge_used), \
            #squeezed_kge = squeezed_kge, \
            hours_in_day = hours_in_day, \
            days_in_week_residual = days_in_week_residual)#, \
            #weeks_in_month_residual = weeks_in_month_residual)            
			#np.savez(os.path.join(save_dir, 'distance.npz'), distances = np.array(dst_list))
        if iteration % 45 == 4:
            print(G_cost, D_cost, Wasserstein_D, dst)#, sparsity)   
print('DW finished!')
if LOAD_TRAIN_DW:
    pretrained_netG_DW_linear_4hd = torch.load('./generated_data_0921_trans_gan_residual_Pre/DW/iteration-304/netG.linear_4hd',map_location=torch.device(gpu))
    netG.linear_4hd.load_state_dict(pretrained_netG_DW_linear_4hd)
    netG.hd_patterns = torch.load('./generated_data_0921_trans_gan_residual_Pre/DW/iteration-304/netG.hd_patterns',map_location=torch.device(gpu))
    pretrained_netG_DW_linear_4hd = torch.load('./generated_data_0921_trans_gan_residual_Pre/DW/iteration-304/netG.linear_4dw',map_location=torch.device(gpu))
    netG.linear_4dw.load_state_dict(pretrained_netG_DW_linear_4hd)
    netG.dwr_patterns = torch.load('./generated_data_0921_trans_gan_residual_Pre/DW/iteration-304/netG.dwr_patterns',map_location=torch.device(gpu))
print('DW loaded!')
			
if TRAIN_WM:    
    BATCH_SIZE = 32
    gene_size = 32
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
#    ones = torch.ones(672, BATCH_SIZE, SHAPE_M0)    
#    zeros = torch.zeros(SHAPE_M0, BATCH_SIZE, 336, 2)
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
#        ones = ones.cuda(gpu)
#        zeros = zeros.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        #print(time.localtime(), ' iteration: ', iteration)
        idx = -1
        for idx, data in enumerate(data_loader):
            idx = idx + 1
            if True:#(idx < 1500) | (idx%(CRITIC_ITERS*3) == 0):        
                ############################
                # (1) Update D network
                ###########################
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

            
                #id_batch = data[0]
                data_batch, kge_batch, hours_in_day, days_in_week_residual, weeks_in_month_residual= data[1:]
                weeks_in_month_residual = weeks_in_month_residual.unsqueeze(1)
                if not(BATCH_FIRST):
                    weeks_in_month_residual = weeks_in_month_residual.permute(2,0,1)
                
                netD.zero_grad()
                
                real_data = weeks_in_month_residual
                #kge = kge_batch
                if use_cuda:
                    real_data = real_data.cuda(gpu)
                    kge_batch = kge_batch.cuda(gpu)
                D_real = netD(real_data, kge_batch)
                D_real = D_real.mean()
                # print D_real
                # TODO: Waiting for the bug fix from pytorch
                D_real.backward(mone)
                
                # generate noise
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                #noisev = autograd.Variable(noise, volatile=True)                
                # train with fake
                fake_data, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG(torch.cat((noise_batch, kge_batch), 1))
                D_fake = netD(fake_weeks_in_month_residual, kge_batch)
                D_fake = D_fake.mean()
                # TODO: Waiting for the bug fix from pytorch
                D_fake.backward(one)
                
                # train with gradient penalty
                #print(fake_data.view(BATCH_SIZE,LENGTH).shape)
                gradient_penalty = calc_gradient_penalty(netD, real_data, fake_weeks_in_month_residual, kge_batch)
                gradient_penalty.backward()
                
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()
                #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if idx%CRITIC_ITERS == 0:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG(torch.cat((noise_batch, kge_batch), 1))
                G = netD(fake_weeks_in_month_residual, kge_batch)
                G = G.mean()
                #detach layers for HD
                fake_hours_in_day.detach_()
                fake_days_in_week_residual.detach_()
                G.backward(mone)
                G_cost = -G
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
                #WDList.append(Wasserstein_D.cpu().data.numpy())
            if False:#idx%CRITIC_ITERS == 1:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_f = netG(torch.cat((noise_batch, kge_batch), 1))
                #print(fake.shape, fake_f.shape)
                sparsity = (torch.sqrt(torch.sum(ones)) - \
                            F.l1_loss(fake_f, zeros, reduction='sum') / \
                            F.mse_loss(fake_f, zeros, reduction='sum')) /\
                            (torch.sqrt(torch.sum(ones)) - 1)
                sparsity.backward()
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                sparsityList.append(sparsity.cpu().data.numpy())
            TimeList.append(time.time() - start_time)
            D_costList.append(D_cost.cpu().data.numpy())
            ##SD_costList.append(SD_cost.cpu().data.numpy())
            WDList.append(Wasserstein_D.cpu().data.numpy())            
            #print(fake.max(), fake.min())
        if iteration % 15 == 4:#True
            save_dir = save_dir_head + '/WM/iteration-' + str(iteration+start_it)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    
            torch.save(netD.state_dict(), os.path.join(save_dir, 'netD'))
            ##torch.save(netSD.state_dict(), os.path.join(save_dir, 'netSD'))
            torch.save(netG.state_dict(), os.path.join(save_dir, 'netG'))
            #torch.save(netD.module.state_dict(), os.path.join(save_dir, 'netD'))
            ##torch.save(netSD.module.state_dict(), os.path.join(save_dir, 'netSD'))
            #torch.save(netG.module.state_dict(), os.path.join(save_dir, 'netG'))
            np.savez(os.path.join(save_dir, 'cost_generated.npz'), \
                     time=np.array(TimeList), \
                     D_cost=np.array(D_costList), \
                     sparsity=np.array(sparsityList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            generated_data, hours_in_day, days_in_week_residual, weeks_in_month_residual = \
                     generate_data(netG, dataset.kge[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
            generated_data = generated_data.reshape(gene_size, -1).cpu().detach().numpy()
            #kge_used = kge_gene.cpu().detach().numpy()
            hours_in_day = hours_in_day.view(gene_size, 24).cpu().detach().numpy()
            days_in_week_residual = days_in_week_residual.view(gene_size, 24*7).cpu().detach().numpy()
            weeks_in_month_residual = weeks_in_month_residual.view(gene_size, LENGTH).cpu().detach().numpy()
            #generated_data_f = generated_data_f.cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(24, 16))
            n_bins = 100
            line_w = 2
            use_cumulative = -1
            use_log = True
            n_real, bins, patches = ax.hist(real_dataset_WM.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='real', log=use_log, facecolor='g', linewidth=line_w)
            n_gene, bins, patches = ax.hist(weeks_in_month_residual.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='gene', log=use_log, facecolor='b', linewidth=line_w)
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('Value')
            ax.set_ylabel('Likelihood of occurrence')
            plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
            plt.close()
            dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)        
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            generated_data = generated_data, \
            distance = dst, \
            distances = np.array(dst_list), \
            #kge_used = np.array(kge_used), \
            #squeezed_kge = squeezed_kge, \
            hours_in_day = hours_in_day, \
            days_in_week_residual = days_in_week_residual, \
            weeks_in_month_residual = weeks_in_month_residual)            
			#np.savez(os.path.join(save_dir, 'distance.npz'), distances = np.array(dst_list))
        #if iteration % 10 == 0:
            print(G_cost, D_cost, Wasserstein_D, dst)#, sparsity)   
print('WM finished!')

if LOAD_TRAIN_WM:
    pretrained_netG = torch.load('./generated_data_0921_trans_gan_residual_Pre/WM/iteration-314/netG',map_location=torch.device(gpu))
    netG.load_state_dict(pretrained_netG)
	
if TRAIN:    
    BATCH_SIZE = 32
    gene_size = 32
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
#    ones = torch.ones(672, BATCH_SIZE, SHAPE_M0)    
#    zeros = torch.zeros(SHAPE_M0, BATCH_SIZE, 336, 2)
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
#        ones = ones.cuda(gpu)
#        zeros = zeros.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        #print(time.localtime(), ' iteration: ', iteration)
        idx = -1
        for idx, data in enumerate(data_loader):
            idx = idx + 1
            if True:#(idx < 1500) | (idx%(CRITIC_ITERS*3) == 0):        
                ############################
                # (1) Update D network
                ###########################
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

            
                #id_batch = data[0]
                data_batch = data[1]
                kge_batch = data[2]
                if not(BATCH_FIRST):
                    data_batch = data_batch.permute(2,0,1)
                
                netD.zero_grad()
                
                real_data = data_batch
                #kge = kge_batch
                if use_cuda:
                    real_data = real_data.cuda(gpu)
                    kge_batch = kge_batch.cuda(gpu)
                D_real = netD(real_data, kge_batch)
                D_real = D_real.mean()
                # print D_real
                # TODO: Waiting for the bug fix from pytorch
                D_real.backward(mone)
                
                # generate noise
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                #noisev = autograd.Variable(noise, volatile=True)                
                # train with fake
                fake_data, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG(torch.cat((noise_batch, kge_batch), 1))
                D_fake = netD(fake_data, kge_batch)
                D_fake = D_fake.mean()
                # TODO: Waiting for the bug fix from pytorch
                D_fake.backward(one)
                
                # train with gradient penalty
                #print(fake_data.view(BATCH_SIZE,LENGTH).shape)
                gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, kge_batch)
                gradient_penalty.backward()
                
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()
                #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if idx%CRITIC_ITERS == 0:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_hours_in_day, fake_days_in_week_residual, fake_weeks_in_month_residual = \
                          netG(torch.cat((noise_batch, kge_batch), 1))
                G = netD(fake, kge_batch)
                G = G.mean()
                G.backward(mone)
                G_cost = -G
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
                #WDList.append(Wasserstein_D.cpu().data.numpy())
            if False:#idx%CRITIC_ITERS == 1:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, fake_f = netG(torch.cat((noise_batch, kge_batch), 1))
                #print(fake.shape, fake_f.shape)
                sparsity = (torch.sqrt(torch.sum(ones)) - \
                            F.l1_loss(fake_f, zeros, reduction='sum') / \
                            F.mse_loss(fake_f, zeros, reduction='sum')) /\
                            (torch.sqrt(torch.sum(ones)) - 1)
                sparsity.backward()
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                sparsityList.append(sparsity.cpu().data.numpy())
            TimeList.append(time.time() - start_time)
            D_costList.append(D_cost.cpu().data.numpy())
            ##SD_costList.append(SD_cost.cpu().data.numpy())
            WDList.append(Wasserstein_D.cpu().data.numpy())            
            #print(fake.max(), fake.min())
        if iteration % 15 == 4:#True
            save_dir = save_dir_head + '/ALL/iteration-' + str(iteration+start_it)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    
            torch.save(netD.state_dict(), os.path.join(save_dir, 'netD'))
            ##torch.save(netSD.state_dict(), os.path.join(save_dir, 'netSD'))
            torch.save(netG.state_dict(), os.path.join(save_dir, 'netG'))
            #torch.save(netD.module.state_dict(), os.path.join(save_dir, 'netD'))
            ##torch.save(netSD.module.state_dict(), os.path.join(save_dir, 'netSD'))
            #torch.save(netG.module.state_dict(), os.path.join(save_dir, 'netG'))
            np.savez(os.path.join(save_dir, 'cost_generated.npz'), \
                     time=np.array(TimeList), \
                     D_cost=np.array(D_costList), \
                     sparsity=np.array(sparsityList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            generated_data, hours_in_day, days_in_week_residual, weeks_in_month_residual = \
                     generate_data(netG, dataset.kge[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
            generated_data = generated_data.view(gene_size, -1).cpu().detach().numpy()
            #kge_used = kge_gene.cpu().detach().numpy()
            hours_in_day = hours_in_day.view(gene_size, 24).cpu().detach().numpy()
            days_in_week_residual = days_in_week_residual.view(gene_size, 24*7).cpu().detach().numpy()
            weeks_in_month_residual = weeks_in_month_residual.view(gene_size, LENGTH).cpu().detach().numpy()
            #generated_data_f = generated_data_f.cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(24, 16))
            n_bins = 100
            line_w = 2
            use_cumulative = -1
            use_log = True
            n_real, bins, patches = ax.hist(real_dataset.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='real', log=use_log, facecolor='g', linewidth=line_w)
            n_gene, bins, patches = ax.hist(generated_data.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='gene', log=use_log, facecolor='b', linewidth=line_w)
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('Value')
            ax.set_ylabel('Likelihood of occurrence')
            plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
            plt.close()
            dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)        
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            generated_data = generated_data, \
            distance = dst, \
            distances = np.array(dst_list), \
            #kge_used = np.array(kge_used), \
            #squeezed_kge = squeezed_kge, \
            hours_in_day = hours_in_day, \
            days_in_week_residual = days_in_week_residual, \
            weeks_in_month_residual = weeks_in_month_residual)            
			#np.savez(os.path.join(save_dir, 'distance.npz'), distances = np.array(dst_list))
        #if iteration % 10 == 0:
            print(G_cost, D_cost, Wasserstein_D, dst)#, sparsity)   
			
			'''
