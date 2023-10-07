import os
import time
import matplotlib

matplotlib.use('SVG')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import setproctitle

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'


use_cuda = torch.cuda.is_available()
gpu =3


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        return x.contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 padding_mode='circular'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding / 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding / 2)
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
        # out_0 = self.conv1(x)
        # out_1 = self.dropout1(self.relu1(self.chomp1(out_0)))
        # out_2 = self.conv2(out_1)
        # out = self.dropout2(self.relu2(self.chomp2(out_2)))
        index_2 = int((out.shape[2] - x.shape[2]) / 2)
        out = out[:, :, index_2:index_2 + x.shape[2]]
        res = x if self.downsample is None else self.downsample(x)
        # print(x.shape, out.shape, self.downsample, res.shape)#out_0.shape, out_1.shape, out_2.shape, out.shape,
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
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation_size = int(168 / kernel_size) if kernel_size * dilation_size > 168 else dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     # padding=kernel_size*dilation_size*2, dropout=dropout)]
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print('xx', x.shape)
        return self.network(x)

class MyDataset(Dataset):
    def __init__(self, data, node_data, label_num_idx=0):
        self.data = np.load(data)
        self.node_embedding = node_data / np.mean(node_data)#node_embedding.repeat(4, 1).reshape(nebd_size[0]*4, nebd_size[1]) / np.mean(node_embedding)
        self.bs_record = torch.from_numpy(self.data['bs_record'].astype(np.float32))#.reshape(
            #    self.node_embedding.shape[0], 1, LENGTH)
        # self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
        self.kge = torch.from_numpy(self.node_embedding.astype(np.float32))
        self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
        self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
        self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
        self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
        self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
        self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
        #self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'].astype(np.float32))
        self.hours_in_weekday_patterns = torch.from_numpy(self.data['hours_in_weekday_patterns'].astype(np.float32))
        self.hours_in_weekend_patterns = torch.from_numpy(self.data['hours_in_weekend_patterns'].astype(np.float32))
        self.days_in_weekday_patterns = torch.from_numpy(self.data['days_in_weekday_patterns'].astype(np.float32))
        self.days_in_weekend_patterns = torch.from_numpy(self.data['days_in_weekend_patterns'].astype(np.float32))
        self.days_in_weekday_residual_patterns = torch.from_numpy(
            self.data['days_in_weekday_residual_patterns'].astype(np.float32))
        self.days_in_weekend_residual_patterns = torch.from_numpy(
            self.data['days_in_weekend_residual_patterns'].astype(np.float32))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #bs_id = self.bs_id[idx]
        bs_record = self.bs_record[idx]
        kge = self.kge[idx]
        hours_in_weekday = self.hours_in_weekday[idx]
        hours_in_weekend = self.hours_in_weekend[idx]
        days_in_weekday = self.days_in_weekday[idx]
        days_in_weekend = self.days_in_weekend[idx]
        return idx, bs_record, kge, hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend

    def __len__(self):
        return self.node_embedding.shape[0]

class GeneratorP_HD_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns=0, pattern_length=24):
        super(GeneratorP_HD_SUM, self).__init__()
        self.linear_4hd = nn.Linear(NOISE_SIZE + KGE_SIZE, pattern_num) if USE_KGE else \
            nn.Linear(NOISE_SIZE, pattern_num)
        #        self.linear_4hdres = nn.Linear(32+KGE_SIZE, 24)
        self.norm_4hd = nn.LayerNorm(pattern_num)
        #        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        #        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(pattern_length)
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()
        self.hd_patterns = torch.empty(pattern_num, pattern_length)
        self.hd_patterns.requires_grad = True
        self.init_weights()
        self.patterns = patterns
        if self.patterns:
            self.hd_patterns = patterns

    def init_weights(self):
        nn.init.orthogonal_(self.linear_4hd.weight)
        nn.init.orthogonal_(self.hd_patterns)

    def init_patterns(self, patterns):
        if not self.patterns:
            self.hd_patterns = patterns

    def forward(self, x):
        #        BZ = x.shape[0]
        #        print(x.shape)
        x_p = self.softmax(self.norm_4hd(self.linear_4hd(x))) if USE_KGE else \
            self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))  # [:, 32:])))
        hours_in_day = self.act(self.norm(
            x_p @ self.hd_patterns))  # + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        #        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        return hours_in_day

class GeneratorP_DW_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns=0, pattern_length=168):
        super(GeneratorP_DW_SUM, self).__init__()
        self.linear_4dw = nn.Linear(NOISE_SIZE + KGE_SIZE, pattern_num) if USE_KGE else \
            nn.Linear(NOISE_SIZE, pattern_num)
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.norm = nn.LayerNorm(pattern_length)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        #        self.relu = nn.ReLU()
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()
        self.dwr_patterns = torch.empty(pattern_num, pattern_length)
        self.dwr_patterns.requires_grad = True
        self.init_weights()
        self.patterns = patterns
        if self.patterns:
            self.dwr_patterns = patterns

    def init_weights(self):
        nn.init.orthogonal_(self.linear_4dw.weight)
        nn.init.orthogonal_(self.dwr_patterns)

    def init_patterns(self, patterns):
        if not self.patterns:
            self.dwr_patterns = patterns

    def forward(self, x, hours_in_day):
        #        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x))) if USE_KGE else \
            self.softmax(self.norm_4dw(self.linear_4dw(x[:, 0:32])))
        days_in_week_residual = self.norm(days_in_week_residual @ self.dwr_patterns)
        days_in_week = days_in_week_residual + hours_in_day.repeat(1, int(
            days_in_week_residual.shape[1] / hours_in_day.shape[1]))
        return self.act(days_in_week)  # _weekend

class GeneratorP_WM_SUM(nn.Module):
    def __init__(self, input_size=20, num_channels=[1] * 6, kernel_size=[24, 2 * 24, 7 * 24], dropout=0.3, kge_size=32,
                 kge_squeeze_size=20, activation="relu"):
        super(GeneratorP_WM_SUM, self).__init__()
        self.kge_size = kge_size
        self.linear_kge = nn.Linear(self.kge_size, kge_squeeze_size)
        self.norm_kge = nn.LayerNorm(kge_squeeze_size)
        if USE_KGE:
            input_size = input_size + kge_squeeze_size
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear_4wm = nn.Linear(self.kge_size, kge_squeeze_size * 168)
        self.norm_4wm = nn.LayerNorm(kge_squeeze_size * 168)
        self.norm = nn.LayerNorm([1, 168])
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()

        self.linear = nn.Linear(num_channels[-1] * len(kernel_size), num_channels[-1])
        self.init_weights()
        # self.relu = nn.ReLU()
        # self.soft_exponential = soft_exponential(num_channels[-1], alpha = 1.0)
        self.tanh = nn.Tanh()


    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, days_in_week):
        BZ = x.shape[0]
        kge = self.norm_kge(self.linear_kge(x[:, self.kge_size:]))
        x = self.norm_4wm(self.linear_4wm(x[:, 0:self.kge_size])).reshape(BZ, -1, 168)
        x = torch.cat((x, kge.view(kge.size(0), kge.size(1), 1).expand(-1, -1, x.size(2))), 1) if USE_KGE else x
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.norm(self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2))
        y = y + days_in_week.reshape(BZ, 1, -1)#.repeat(1, 4).reshape(BZ, 1, -1)
        return self.act(y.squeeze(1))  # , y_d, y_w, y_m, kge

class GeneratorP_ALL_LN_Matrioska(nn.Module):  #
    def __init__(self, nhead=[1, 1, 1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL_LN_Matrioska, self).__init__()
        self.generator_hdd = GeneratorP_HD_SUM(activation=activation, pattern_length=24)
        self.generator_hde = GeneratorP_HD_SUM(activation=activation, pattern_length=24)
        self.generator_dwd = GeneratorP_DW_SUM(activation=activation, pattern_length=24*5)
        self.generator_dwe = GeneratorP_DW_SUM(activation=activation, pattern_length=24*2)
        self.generator_wm = GeneratorP_WM_SUM(activation=activation,kge_size=KGE_SIZE)
        # self.norm = nn.LayerNorm(
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU()

    def init_patterns(self, hours_in_weekday_patterns, hours_in_weekend_patterns, days_in_weekday_residual_patterns, days_in_weekend_residual_patterns):
        self.generator_hdd.init_patterns(hours_in_weekday_patterns)
        self.generator_hde.init_patterns(hours_in_weekend_patterns)
        self.generator_dwd.init_patterns(days_in_weekday_residual_patterns)
        self.generator_dwe.init_patterns(days_in_weekend_residual_patterns)


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


def generate_data(netG, kge, gene_size=128):
    noise = torch.randn(gene_size, NOISE_SIZE)  # , LENGTH)#torch.randn(BATCH_SIZE, NOISE_SIZE, dim_list_g[0])
    noise = noise.exponential_() if EXP_NOISE else noise
    if use_cuda:
        noise = noise.cuda(gpu)
        kge = kge.cuda(gpu)
    hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output = netG(
        torch.cat((noise, kge), 1))  # , kge)
    return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output

def load_G_model(netG, pre_train_dir):
    pretrained_netG = torch.load(os.path.join(pre_train_dir, 'netG'), map_location=torch.device(gpu))
    # print(pretrained_netG)
    # print(netG.generator_hdd.hd_patterns)
    netG.load_state_dict(pretrained_netG)

    netG.generator_dwd.dwr_patterns = torch.load(
        os.path.join(pre_train_dir, 'netG.generator_dwd.dwr_patterns'),
        map_location=torch.device(gpu))
    netG.generator_dwe.dwr_patterns = torch.load(
        os.path.join(pre_train_dir, 'netG.generator_dwe.dwr_patterns'),
        map_location=torch.device(gpu))
    netG.generator_hdd.hd_patterns = torch.load(
        os.path.join(pre_train_dir, 'netG.generator_hdd.hd_patterns'),
        map_location=torch.device(gpu))
    netG.generator_hde.hd_patterns = torch.load(
        os.path.join(pre_train_dir, 'netG.generator_hde.hd_patterns'),
        map_location=torch.device(gpu))
    # print(netG.generator_hdd.hd_patterns)

    # print('ALL loaded!')
    '''
    weights = [netG.generator_dwd.linear_4dw.weight, netG.generator_dwe.linear_4dw.weight,
               netG.generator_hdd.linear_4hd.weight, netG.generator_hde.linear_4hd.weight,
               netG.generator_wm.linear_kge.weight, netG.generator_wm.linear_4wm.weight,
               netG.generator_wm.linear.weight]
    for w in weights:
        print(w.shape, w.T@w)#torch.linalg.matrix_rank(w))
    '''
    return netG

def distribution_jsd(generated_data, real_dataset, save_dir='', n_bins=100):
    generated_data[generated_data<0]=0
    fig, ax = plt.subplots(figsize=(24, 6))
    line_w = 2
    use_cumulative = -1
    use_log = True
    n_real, bins, patches = ax.hist(real_dataset.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='real', log=use_log, facecolor='g',
                                    linewidth=line_w)
    n_gene, bins, patches = ax.hist(generated_data.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='gene', log=use_log, facecolor='b',
                                    linewidth=line_w)
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Value')
    ax.set_ylabel('Likelihood of occurrence')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
    plt.close()

    fig, ax = plt.subplots(figsize=(24, 6))
    real_diff = real_dataset[1:] - real_dataset[:-1]
    generated_diff = generated_data[1:] - generated_data[:-1]
    n_real, bins, patches = ax.hist(real_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='real', log=use_log, facecolor='g',
                                    linewidth=line_w)
    n_gene, bins, patches = ax.hist(generated_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='gene', log=use_log, facecolor='b',
                                    linewidth=line_w)
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Value')
    ax.set_ylabel('Likelihood of occurrence')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_diff_hist.jpg'))
    plt.close()
    JSD_diff = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    return JSD, JSD_diff

def pattern_freq_ratio_rmse(generated_data, real_dataset):
    data_f = abs(np.fft.rfft(real_dataset))
    daily_real = data_f[:, 7] / data_f.sum(1)

    data_f = abs(np.fft.rfft(generated_data))
    daily = data_f[:, 7] / data_f.sum(1) if data_f.sum(1).all() > 0 else data_f[:, 7]
    daily = np.concatenate([daily,daily,daily,daily])
    rmse_daily = np.sqrt(np.mean((daily - daily_real) ** 2))
    return rmse_daily

load_dir = ''
work_dir = ''
save_dir_today = ''
save_dir_today = os.path.join(work_dir,save_dir_today)
if not os.path.exists(save_dir_today):
    os.mkdir(save_dir_today)


EXP_NOISE = True
USE_KGE = True
KGE_SIZE = 32
NOISE_SIZE = KGE_SIZE
LENGTH = 168
BATCH_SIZE = 32
BATCH_FIRST = True
metric_results = np.zeros((20, 3))

for seed in np.arange(20):
    torch.manual_seed(seed)
    #idx_use = np.load(os.path.join(work_dir, 'bs_record_w_4g_use.npz'))['idx_use']
    save_dir_head = 'original_kge'
    save_dir_head = os.path.join(save_dir_today, save_dir_head)
    if not os.path.exists(save_dir_head):
        os.mkdir(save_dir_head)
    node_file = os.path.join(work_dir, 'bs4graph.npz')
    node_data = np.load(node_file)['kge']

    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w.npz'),
                        node_data=node_data)
    DATASET_SIZE = len(dataset)
    gene_size = DATASET_SIZE  # 1024

    real_dataset_list = [dataset.data['hours_in_weekday'], dataset.data['hours_in_weekend'], dataset.data['days_in_weekday'],
                         dataset.data['days_in_weekend'], dataset.data['bs_record']]
    hours_in_weekday_patterns = dataset.hours_in_weekday_patterns.cuda(gpu)
    hours_in_weekend_patterns = dataset.hours_in_weekend_patterns.cuda(gpu)
    days_in_weekday_patterns = dataset.days_in_weekday_patterns.cuda(gpu)
    days_in_weekend_patterns = dataset.days_in_weekend_patterns.cuda(gpu)
    days_in_weekday_residual_patterns = dataset.days_in_weekday_residual_patterns.cuda(gpu)
    days_in_weekend_residual_patterns = dataset.days_in_weekend_residual_patterns.cuda(gpu)
    #data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    dropout = 0.3
    num_layers = 2
    netG = GeneratorP_ALL_LN_Matrioska().cuda(gpu)
    #netG.init_patterns(hours_in_weekday_patterns, hours_in_weekend_patterns, days_in_weekday_residual_patterns,
    #                   days_in_weekend_residual_patterns)


    LOAD_PRE_TRAIN = True
    GENE = True


    if LOAD_PRE_TRAIN:
        pre_train_dir = os.path.join(work_dir, '')
        netG = load_G_model(netG, pre_train_dir)
    if GENE:
        save_dir = os.path.join(save_dir_head, 'sample_rate_0')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        real_data = real_dataset_list[4]
        fake_data = generate_data(netG, dataset.kge, gene_size)
        generated_data = fake_data[4].reshape(gene_size, -1).cpu().detach().numpy()
        # kge_used = kge_gene.cpu().detach().numpy()
        hours_in_weekday = fake_data[0].view(gene_size, 24).cpu().detach().numpy()
        hours_in_weekend = fake_data[1].view(gene_size, 24).cpu().detach().numpy()
        days_in_weekday = fake_data[2].view(gene_size, 24 * 5).cpu().detach().numpy()
        days_in_weekend = fake_data[3].view(gene_size, 24 * 2).cpu().detach().numpy()
        jsd, jsd_diff = distribution_jsd(generated_data, real_data, save_dir=save_dir)
        rmse = pattern_freq_ratio_rmse(generated_data, real_data)
        np.savez(os.path.join(save_dir, 'generated.npz'),
                 generated_data=generated_data,
                 distance=[jsd, jsd_diff, rmse],
                 #distances=np.array(dst_list),
                 # kge_used = np.array(kge_used),
                 hours_in_weekday=hours_in_weekday,
                 hours_in_weekend=hours_in_weekend,
                 days_in_weekend=days_in_weekend,
                 days_in_weekday=days_in_weekday)
        print(save_dir_head, jsd, jsd_diff, rmse)#, sparsity)
        metric_results[seed, :] = [jsd, jsd_diff, rmse]
    #print('Generating finished!')
np.savez(os.path.join(save_dir_today, 'metric_results_kge_relu.npz'),
         metric_results = metric_results)



EXP_NOISE = True
USE_KGE = True
KGE_SIZE = 14
NOISE_SIZE = KGE_SIZE
LENGTH = 168
BATCH_SIZE = 32
BATCH_FIRST = True
metric_results = np.zeros((20, 3))

for seed in np.arange(20):
    torch.manual_seed(seed)
    #idx_use = np.load(os.path.join(work_dir, 'bs_record_w_4g_use.npz'))['idx_use']
    save_dir_head = 'poi_cate_dtb'
    save_dir_head = os.path.join(save_dir_today, save_dir_head)
    if not os.path.exists(save_dir_head):
        os.mkdir(save_dir_head)
    node_file = os.path.join(work_dir, 'bs4graph.npz')
    poi_cate_dtb = np.load(node_file)['poi_cate_dtb']
    poi_count4cate = np.array([64859., 57524., 276009., 347685., 26691., 17844., 39550., 40196., 10161., 69976., 226999., 303655., 23304., 213398.])
    poi_cate_dtb_n1 = np.array(poi_cate_dtb)
    poi_cate_dtb_n1 = poi_cate_dtb_n1 / np.tile(poi_count4cate.reshape(-1, 1), poi_cate_dtb_n1.shape[0]).T
    node_data = poi_cate_dtb_n1 / np.tile((poi_cate_dtb_n1.sum(1) + 1e-99).reshape(-1, 1),
                                          poi_cate_dtb_n1.shape[1])
    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w.npz'),
                        node_data=node_data)
    DATASET_SIZE = len(dataset)
    gene_size = DATASET_SIZE  # 1024

    real_dataset_list = [dataset.data['hours_in_weekday'], dataset.data['hours_in_weekend'], dataset.data['days_in_weekday'],
                         dataset.data['days_in_weekend'], dataset.data['bs_record']]
    hours_in_weekday_patterns = dataset.hours_in_weekday_patterns.cuda(gpu)
    hours_in_weekend_patterns = dataset.hours_in_weekend_patterns.cuda(gpu)
    days_in_weekday_patterns = dataset.days_in_weekday_patterns.cuda(gpu)
    days_in_weekend_patterns = dataset.days_in_weekend_patterns.cuda(gpu)
    days_in_weekday_residual_patterns = dataset.days_in_weekday_residual_patterns.cuda(gpu)
    days_in_weekend_residual_patterns = dataset.days_in_weekend_residual_patterns.cuda(gpu)
    #data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    dropout = 0.3
    num_layers = 2
    netG = GeneratorP_ALL_LN_Matrioska().cuda(gpu)
    #netG.init_patterns(hours_in_weekday_patterns, hours_in_weekend_patterns, days_in_weekday_residual_patterns,
    #                   days_in_weekend_residual_patterns)


    LOAD_PRE_TRAIN = True
    GENE = True


    if LOAD_PRE_TRAIN:
        pre_train_dir = os.path.join(work_dir, 'generate_data_0507_poi_cate_dtb/ALL/iteration-330')
        netG = load_G_model(netG, pre_train_dir)
    if GENE:
        save_dir = os.path.join(save_dir_head, 'sample_rate_0')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        real_data = real_dataset_list[4]
        fake_data = generate_data(netG, dataset.kge, gene_size)
        generated_data = fake_data[4].reshape(gene_size, -1).cpu().detach().numpy()
        # kge_used = kge_gene.cpu().detach().numpy()
        hours_in_weekday = fake_data[0].view(gene_size, 24).cpu().detach().numpy()
        hours_in_weekend = fake_data[1].view(gene_size, 24).cpu().detach().numpy()
        days_in_weekday = fake_data[2].view(gene_size, 24 * 5).cpu().detach().numpy()
        days_in_weekend = fake_data[3].view(gene_size, 24 * 2).cpu().detach().numpy()
        jsd, jsd_diff = distribution_jsd(generated_data, real_data, save_dir=save_dir)
        rmse = pattern_freq_ratio_rmse(generated_data, real_data)
        np.savez(os.path.join(save_dir, 'generated.npz'),
                 generated_data=generated_data,
                 distance=[jsd, jsd_diff, rmse],
                 #distances=np.array(dst_list),
                 # kge_used = np.array(kge_used),
                 hours_in_weekday=hours_in_weekday,
                 hours_in_weekend=hours_in_weekend,
                 days_in_weekend=days_in_weekend,
                 days_in_weekday=days_in_weekday)
        print(save_dir_head, jsd, jsd_diff, rmse)#, sparsity)
        metric_results[seed, :] = [jsd, jsd_diff, rmse]
    #print('Generating finished!')
np.savez(os.path.join(save_dir_today, 'metric_results_poi_relu.npz'),
         metric_results = metric_results)
