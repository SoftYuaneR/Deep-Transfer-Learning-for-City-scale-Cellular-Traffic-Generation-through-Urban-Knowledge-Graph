import os, sys
import time
import matplotlib
import random
matplotlib.use('SVG')
import torch
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

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
torch.manual_seed(5)
use_cuda = torch.cuda.is_available()

import tcgan
class MyDataset_old(Dataset):
    def __init__(self, data, node_data, label_num_idx=0):
        self.data = np.load(data)
        self.node_embedding = node_data / np.mean(node_data)#node_embedding.repeat(4, 1).reshape(nebd_size[0]*4, nebd_size[1]) / np.mean(node_embedding)
        self.bs_record = torch.from_numpy(self.data['bs_record'].astype(np.float32)).reshape(
                self.node_embedding.shape[0], 1, 168)
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
from tcgan import LightGenerator
from tcgan import Discriminator
from RESGAN_Partly import MyDataset

def calc_gradient_penalty(netD, real_data, fake_data, kge):
    #use_cuda = 0
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

#    disc_interpolates, d_z, d_d, d_w, d_m = netD(interpolates.view(BATCH_SIZE,1,LENGTH), kge.view(BATCH_SIZE, -1, KGE_SIZE))
    disc_interpolates = netD(interpolates.view(BATCH_SIZE,1,LENGTH), kge.view(BATCH_SIZE, KGE_SIZE))
    
    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_data(netG, kge, gene_size=128):
    #noise_attribute = kge#torch.rand(gene_size,12)
    noise = torch.randn(gene_size, NOISE_SIZE,  LENGTH)#torch.randn(BATCH_SIZE, NOISE_SIZE, dim_list_g[0])
    if use_cuda:
        #noise_attribute = noise_attribute.cuda(gpu)
        noise = noise.cuda(gpu) 
        kge = kge.cuda(gpu)
    #noisev = autograd.Variable(noise, volatile=True)
    #noise_kge = torch.cat((noise, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, LENGTH)), 1)
#    output, output_d, output_w, output_m, z = netG(torch.cat((noise, kge.view(gene_size, -1, KGE_SIZE)), 2))#, kge)
    output, output_d, output_w, output_m, z = netG(noise, kge)
    #return output, kge, output_d, output_w, output_m, z
    return output, output_d, output_w, output_m, z
def normalize(X):
    X_std = (X - np.min(X)) / (np.max(X) -np.min(X))
   
    return X_std

from sklearn import preprocessing

def distribution_jsd(generated_data, real_dataset, save_dir='', n_bins=100):
    generated_data[generated_data<0]=0
    
    n_real=real_dataset.flatten()
    n_gene = generated_data.flatten()
    
    n_real[np.isnan(n_real)]=0
    n_gene[np.isnan(n_gene)]=0
    n_real[np.isinf(n_real)]=0
    n_gene[np.isinf(n_gene)]=0
#    n_real = preprocessing.scale(n_real)
 #   n_gene = preprocessing.scale(n_gene)
    
    
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)

   

    
    real_diff = abs(real_dataset[1:] - real_dataset[:-1])
    generated_diff =abs( generated_data[1:] - generated_data[:-1])
    n_real =real_diff.flatten()
    n_gene = generated_diff.flatten()
    n_real[np.isnan(n_real)]=0
    n_gene[np.isnan(n_gene)]=0
    n_real[np.isinf(n_real)]=0
    n_gene[np.isinf(n_gene)]=0

  #  n_real = preprocessing.scale(n_real)
   # n_gene = preprocessing.scale(n_gene)
    JSD_diff =0.2* distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    return JSD, JSD_diff

def pattern_freq_ratio_rmse(generated_data, real_dataset):
    data_f = abs(np.fft.rfft(real_dataset))
    daily_real = data_f[:, 7] / data_f.sum(1)

    data_f = abs(np.fft.rfft(generated_data))
    daily = data_f[:, 7] / data_f.sum(1) if data_f.sum(1).all() > 0 else data_f[:, 7]
    daily_real[np.isnan(daily_real)]=0
    daily[np.isnan(daily)]=0

    rmse_daily = np.sqrt(np.mean((daily - daily_real) ** 2))
    return rmse_daily
class TransferDataset(Dataset):
    def __init__(self, data, node_data, sample_rate=1):
        self.data = np.load(data)
        self.node_embedding = node_data / np.mean(node_data)#node_embedding.repeat(4, 1).reshape(nebd_size[0]*4, nebd_size[1]) / np.mean(node_embedding)
        sample_list = np.arange(self.node_embedding.shape[0])
        if sample_rate < 1:
            sample_list = np.random.choice(sample_list, int(self.node_embedding.shape[0]*sample_rate), replace=False)
        self.bs_record = torch.from_numpy(self.data['bs_record'].astype(np.float32))#.reshape(
#                int(self.node_embedding.shape[0]*sample_rate), 1, LENGTH)
        # self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
        self.kge = torch.from_numpy(self.node_embedding.astype(np.float32))
        self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
        self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
        self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
        self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
        self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
        self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
        #self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'][sample_list].astype(np.float32))
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
        return self.kge.shape[0]
BATCH_SIZE = 1
LENGTH = 24*7#744
LAMBDA = 10
ITERS = 331 # How many iterations to train for
#SEQ_LEN = 32 # Sequence length in characters
#DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 5
#MAX_N_EXAMPLES = 10000000#10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

gpu=6
TimeList = []
D_costList = []
##SD_costList = []
G_costList = []
WDList = []
dst_list = []
kge_type = 'node'
if kge_type == 'node':
    KGE_SIZE = 160  # 32
    type_dir = ''
    work_dir = ''
    save_dir_head = os.path.join(work_dir, type_dir)
    transfer_generation_dir = ''
    transfer_save_dir = os.path.join(transfer_generation_dir, type_dir)
    if not os.path.exists(transfer_save_dir):
        os.mkdir(transfer_save_dir)
    idx_use = np.load(os.path.join(transfer_generation_dir, 'bs_record_nj.npz'))['idx_use']
    transfer_node_file = os.path.join(transfer_generation_dir,
                                      'node_embedding_KL_log_layerscat_d14_c2.npz')
    transfer_node_data = np.load(transfer_node_file)['node_embedding'][idx_use]
    transfer_dataset = TransferDataset(data=os.path.join(transfer_generation_dir, 'bs_record_nj_norm.npz'),
                                       node_data=transfer_node_data)

    # dataset=MyDataset('/data5/huishuodi/cross-city/urban_data/shanghai/bs_record_w_4g_use.npz')#('D:\实验室\物联网云平台\\traffic_generation\\feature\\data_train.npz')
    node_file = os.path.join(work_dir, 'node_embedding_KL_log_layerscat_d8_c2.npz')
    node_data = np.load(node_file)['node_embedding']
    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w_4g_use.npz'),
                        node_data=node_data)
elif kge_type == 'original_kge':
    KGE_SIZE = 32
    type_dir = ''
    work_dir = ''
    save_dir_head = os.path.join(work_dir, type_dir)
    transfer_generation_dir = ''
    transfer_save_dir = os.path.join(transfer_generation_dir, type_dir)
    if not os.path.exists(transfer_save_dir):
        os.mkdir(transfer_save_dir)
    idx_use = np.load(os.path.join(transfer_generation_dir, 'bs_record_nj_norm.npz'))['idx_use']
    transfer_node_file = os.path.join(transfer_generation_dir,
                                      'bs4graph.npz')
    transfer_node_data = np.load(transfer_node_file)['kge'][idx_use]
    transfer_dataset = TransferDataset(data=os.path.join(transfer_generation_dir, 'bs_record_nj_norm.npz'),
                                       node_data=transfer_node_data)

    # dataset=MyDataset('/data5/huishuodi/cross-city/urban_data/shanghai/bs_record_w_4g_use.npz')#('D:\实验室\物联网云平台\\traffic_generation\\feature\\data_train.npz')
    node_file = os.path.join(work_dir, 'bs4graph_4g.npz')
    node_data = np.load(node_file)['kge']
    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w_4g_use.npz'),
                        node_data=node_data)
elif kge_type == 'poi_cate_dtb':
    KGE_SIZE = 14
    type_dir = ''
    work_dir = ''
    save_dir_head = os.path.join(work_dir, type_dir)
    transfer_generation_dir = '/'
    transfer_save_dir = os.path.join(transfer_generation_dir, type_dir)
    if not os.path.exists(transfer_save_dir):
        os.mkdir(transfer_save_dir)
    idx_use = np.load(os.path.join(transfer_generation_dir, 'bs_record_nj.npz'))['idx_use']
    transfer_node_file = os.path.join(transfer_generation_dir,
                                      'bs4graph.npz')
    transfer_node_data = np.load(transfer_node_file)['poi_cate_dtb'][idx_use]
    transfer_dataset = TransferDataset(data=os.path.join(transfer_generation_dir, 'bs_record_nj_norm.npz'),
                                       node_data=transfer_node_data)

    # dataset=MyDataset('/data5/huishuodi/cross-city/urban_data/shanghai/bs_record_w_4g_use.npz')#('D:\实验室\物联网云平台\\traffic_generation\\feature\\data_train.npz')
    node_file = os.path.join(work_dir, 'bs4graph_4g.npz')
    node_data = np.load(node_file)['poi_cate_dtb']
    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w_4g_use.npz'),
                        node_data=node_data)
transfer_real_dataset_list = [transfer_dataset.data['hours_in_weekday'], transfer_dataset.data['hours_in_weekend'],
                         transfer_dataset.data['days_in_weekday'],
                         transfer_dataset.data['days_in_weekend'], transfer_dataset.data['bs_record']]
transfer_real_data = transfer_real_dataset_list[4]
TRAIN = True
TRANSFER_GENE = True
GENE = False
PRE_TRAIN = False
LOAD_PRE_TRAIN = False
GENE_NUM = 20

DATASET_SIZE = len(dataset)
gene_size = DATASET_SIZE#1024
real_dataset = dataset.data['bs_record']#*dataset.data['bs_record_max']
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)


KGE_SQUEEZE_SIZE = 10
NOISE_SIZE = 10
input_size = KGE_SQUEEZE_SIZE + NOISE_SIZE
output_size = 1
layer_num = 6
num_channels = [1] * layer_num #[32, 16, 8, 4, 2, 1, 1]
kernel_size = [24, 3*24, 7*24]
dropout = 0.3
netG = LightGenerator(input_size, num_channels, kernel_size, dropout, KGE_SIZE, KGE_SQUEEZE_SIZE)
netD = Discriminator(output_size, num_channels, kernel_size, dropout, KGE_SIZE, KGE_SQUEEZE_SIZE)#.double()


if PRE_TRAIN:
    pretrained_netG = torch.load('./generated_data_0622_lg_10/iteration-99/netG',map_location=torch.device('cpu'))
    netG.load_state_dict(pretrained_netG)

#netG = torch.nn.DataParallel(netG, device_ids=[3,2,1])
#netD = torch.nn.DataParallel(netD, device_ids=[3,2,1])
#netG.to(f'cuda:{netG.device_ids[0]}')
#netD.to(f'cuda:{netD.device_ids[0]}')

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
#print(netG)
#print(netD)

if GENE:
    save_dir = "./generated_data_0623_lg_10_gene"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    generated_data, kge_gene, generated_d, generated_w, generated_m, squeezed_kge = generate_data(netG, dataset.kge, len(dataset))
    generated_data = generated_data.view(len(dataset), LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
    generated_d = generated_d.view(len(dataset), LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
    generated_w = generated_w.view(len(dataset), LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
    generated_m = generated_m.view(len(dataset), LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
    squeezed_kge = squeezed_kge.cpu().detach().numpy()

    kge_gene = kge_gene.cpu().detach().numpy()
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
    kge_used = np.array(kge_gene), \
	squeezed_kge = squeezed_kge, \
    generated_d = generated_d, \
    generated_w = generated_w, \
    generated_m = generated_m)



if TRAIN:    
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
    
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        #print(time.localtime(), ' iteration: ', iteration)
        iter_d = 0
        for idx, data in enumerate(data_loader):
        
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            iter_d = iter_d + 1
            
            #id_batch = data[0]
            data_batch = data[1]
            kge_batch = data[2]
            
            
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
            noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE, LENGTH)
            if use_cuda:
                noise_batch = noise_batch.cuda(gpu) 
            #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
            #noisev = autograd.Variable(noise, volatile=True)                
            # train with fake
            fake_data, _, _, _, _ = netG(noise_batch, kge_batch)
            D_fake = netD(fake_data, kge_batch)
            D_fake = D_fake.mean()
            # TODO: Waiting for the bug fix from pytorch
            D_fake.backward(one)
            
            # train with gradient penalty
            #print(fake_data.view(BATCH_SIZE,LENGTH).shape)
            gradient_penalty = calc_gradient_penalty(netD, real_data.view(BATCH_SIZE,LENGTH), fake_data.view(BATCH_SIZE,LENGTH), kge_batch)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
            #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if iter_d%CRITIC_ITERS == 0:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()
                
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE, LENGTH)
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                #noisev = autograd.Variable(noise)
                #noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                fake, _, _, _, _ = netG(noise_batch, kge_batch)
                G = netD(fake, kge_batch)
                G = G.mean()
                G.backward(mone)
                G_cost = -G
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
                TimeList.append(time.time() - start_time)
                D_costList.append(D_cost.cpu().data.numpy())
                ##SD_costList.append(SD_cost.cpu().data.numpy())
                WDList.append(Wasserstein_D.cpu().data.numpy())
            
        if iteration % 30 == 0:
            save_dir = save_dir_head + '/iteration-' + str(iteration)
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
                     ##SD_cost=np.array(SD_costList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            generated_data, generated_d, generated_w, generated_m, squeezed_kge = generate_data(netG, dataset.kge, gene_size)#[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
            generated_data = generated_data.view(gene_size, LENGTH).cpu().detach().numpy()#*dataset.data['bs_record_max']
            #kge_used = kge_gene.cpu().detach().numpy()
            #generated_d = generated_d.view(gene_size, LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
            #generated_w = generated_w.view(gene_size, LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
            #generated_m = generated_m.view(gene_size, LENGTH).cpu().detach().numpy()*dataset.data['bs_record_max']
            #squeezed_kge = squeezed_kge.cpu().detach().numpy()
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
            jsd, jsd_diff = distribution_jsd(generated_data, real_dataset, save_dir=save_dir)
            rmse = pattern_freq_ratio_rmse(generated_data, real_dataset)
            dst = [jsd, jsd_diff, rmse]
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            generated_data = generated_data, \
            distance = dst, \
            distances = np.array(dst_list), \
            #kge_used = np.array(kge_used), \
            #squeezed_kge = squeezed_kge, \
            #generated_d = generated_d, \
            #generated_w = generated_w, \
            #generated_m = generated_m
			)            
			#np.savez(os.path.join(save_dir, 'distance.npz'), distances = np.array(dst_list))
        #if iteration % 10 == 0:
            print(G_cost, D_cost, Wasserstein_D, dst)    
    #    except:
    #        print('#############error#############',iteration) 
    #        ErrorList.append(iteration)
             
    #torch.save(netD.module.state_dict(), './generated_data_kge_500/netD')
    ##torch.save(netSD.state_dict(), './generated_data/netSD')
    #torch.save(netG.module.state_dict(), './generated_data_kge_500/netG')
    
    #np.savez(os.path.join('./generated_data_kge_500/', 'cost_generated.npz'), \
    #                     time=np.array(TimeList), \
    #                     D_cost=np.array(D_costList), \
    #                     ##SD_cost=np.array(SD_costList),\
    #                     G_cost=np.array(G_costList),\
    #                     WD=np.array(WDList))#,\
    #                     #Error=np.array(ErrorList))         
    
if TRANSFER_GENE:
    metric_results = np.zeros((GENE_NUM, 3))
    #load_dir_20220622 = '/data5/huishuodi/cross-city/urban_data/shanghai/generated_data_0531_tcn_gan_d10_c4/iteration-330/netG'
    pretrained_netG = torch.load(save_dir_head + '/iteration-300/netG', map_location=torch.device('cpu'))
    #pretrained_netG = torch.load(load_dir_20220622, map_location=torch.device('cpu'))
    netG.load_state_dict(pretrained_netG,False)
    netG = netG.cuda(gpu)
    slice_list = np.arange(0, len(transfer_dataset), BATCH_SIZE).tolist()
    slice_list.append(len(transfer_dataset))
    # print(slice_list)
    for seed in trange(GENE_NUM):
        torch.manual_seed(seed)
        generated_data, generated_d, generated_w, generated_m, squeezed_kge = generate_data(netG,
               transfer_dataset.kge.cuda(gpu),len(transfer_dataset))  # [random.sample(range(DATASET_SIZE), gene_size)], gene_size)
        generated_data = generated_data.view(len(transfer_dataset), -1).cpu().detach().numpy()
        jsd, jsd_diff = distribution_jsd(generated_data, transfer_real_data, save_dir=transfer_save_dir)
        rmse = pattern_freq_ratio_rmse(generated_data, transfer_real_data)
        metric_results[seed, :] = [jsd, jsd_diff, rmse]
        # print('Generating finished!')
    np.savez(os.path.join(transfer_save_dir, 'metric_results.npz'),
                 metric_results=metric_results)
    print(metric_results.mean(0))
