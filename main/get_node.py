import setproctitle
from torch_geometric.nn import GCNConv
import numpy as np
import json
import os
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from collections import Counter


class Net(torch.nn.Module):
    def __init__(self, POI_cate_num=14, hidden_gcn=128, hidden_mlp=[128], num_node_features=32):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_gcn)
        self.conv2 = GCNConv(hidden_gcn, num_node_features)

        self.linears = nn.ModuleList([])
        self.hidden_mlp = [num_node_features + hidden_gcn] + hidden_mlp
        for ii in range(len(hidden_mlp)):
            self.linears.append(nn.Linear(self.hidden_mlp[ii], self.hidden_mlp[ii + 1]))
            self.linears.append(nn.LayerNorm(self.hidden_mlp[ii + 1]))
        self.linears.append(nn.Linear(self.hidden_mlp[ii + 1], POI_cate_num))
        self.linears.append(nn.Softmax(dim=1))

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)  # convolution 1
        x = F.sigmoid(x)
        y = F.dropout(x, training=self.training)
        y = self.conv2(y, edge_index)  # convolution 2

        return torch.cat((x, F.relu(y)), 1)

    def link_predict(self, x, edge_index_, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat([edge_index_, neg_edge_index], dim=-1)  # concatenate pos and neg edges
        logits = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def node_classify(self, x):
        for linear in self.linears:
            x = linear(x)
        return x


def criterion_tc_bj(node_embeddings, cluster_label_in, node_center_s, node_embedding_source):
    l = torch.zeros(1, dtype=torch.float, device=device)
    for node_ebd in node_embeddings:
        node_source_idx = F.cosine_similarity(node_ebd.reshape(1,-1), node_embedding_source).argmax()
        node_source_label = cluster_label_in[node_source_idx*4:(node_source_idx+1)*4]
        label = node_source_label[torch.randint(4, (1,))[0]]
        node_source_sim = node_center_s[label]
        p = torch.exp(F.cosine_similarity(node_ebd.reshape(1, -1), node_source_sim.reshape(-1, 160))).mean() / torch.exp(
            F.cosine_similarity(node_ebd.reshape(1, -1), node_center_s)).mean()
        l = l - torch.log(p + 1e-49)
    return l

def criterion_tc_bj_allclu(node_embeddings, cluster_label_in, node_center_s, node_embedding_source):
    l = torch.zeros(1, dtype=torch.float, device=device)
    for node_ebd in node_embeddings:
        node_source_idx = F.cosine_similarity(node_ebd.reshape(1,-1), node_embedding_source).argmax()
        node_source_label = cluster_label_in[node_source_idx*4:(node_source_idx+1)*4]
        #label = node_source_label[torch.randint(4, (1,))[0]]
        node_source_sim = node_center_s[node_source_label]#label
        p = torch.exp(F.cosine_similarity(node_ebd.reshape(1, -1), node_source_sim)).mean() / torch.exp(
            F.cosine_similarity(node_ebd.reshape(1, -1), node_center_s)).mean()
        l = l - torch.log(p + 1e-49)
    return l

def criterion_tc_bj_abs(node_embeddings, cluster_label_in, node_center_s, node_embedding_source):
    l = torch.zeros(1, dtype=torch.float, device=device)
    for node_ebd in node_embeddings:
        node_source_idx = F.cosine_similarity(node_ebd.reshape(1,-1), node_embedding_source).argmax()
        node_source_label = cluster_label_in[node_source_idx*4:(node_source_idx+1)*4]
        #label = node_source_label[torch.randint(4, (1,))[0]]
        node_source_sim = node_center_s[node_source_label]#label
        d = F.cosine_similarity(node_ebd.reshape(1, -1), node_source_sim.reshape(-1, 160)).mean()
        l = l - d
    return l/node_embeddings.shape[0]

def criterion_tc_bj_major(node_embeddings, cluster_label_in, node_center_s, node_embedding_source):
    l = torch.zeros(1, dtype=torch.float, device=device)
    for node_ebd in node_embeddings:
        node_source_idx = F.cosine_similarity(node_ebd.reshape(1,-1), node_embedding_source.reshape(-1, 160)).argmax()
        node_source_label = cluster_label_in[node_source_idx]
        node_source_sim = node_center_s[node_source_label]
        p = torch.exp(F.cosine_similarity(node_ebd.reshape(1, -1), node_source_sim.reshape(-1, 160))).mean() / torch.exp(
            F.cosine_similarity(node_ebd.reshape(1, -1), node_center_s.reshape(-1, 160))).mean()
        l = l - torch.log(p + 1e-49)
    return l

bs4graph = np.load('bs4graph.npz')
bs_id = bs4graph['bs_id']
distances = bs4graph['distances']
kge = bs4graph['kge']
poi_cate_dtb = bs4graph['poi_cate_dtb']
# traffic_cluster_labels = bs4graph['traffic_cluster_labels']
poi_count4cate = np.array([ 40369., 56410., 156471., 223717., 30436., 12420., 53065.,41573., 15514., 59066., 202499., 262227., 27324., 185905.])
poi_cate_dtb_n1 = np.array(poi_cate_dtb)
poi_cate_dtb_n1 = poi_cate_dtb_n1/np.tile(poi_count4cate.reshape(-1,1), poi_cate_dtb_n1.shape[0]).T
poi_cate_dtb_n2 = poi_cate_dtb_n1/np.tile((poi_cate_dtb_n1.sum(1)+1e-99).reshape(-1,1), poi_cate_dtb_n1.shape[1])

with open('', 'r') as f:
    edge4graph = json.load(f)
edge_index_list = edge4graph['edge_index_list']
edge_attr_list = edge4graph['edge_attr_list']

labels_list_sh = np.load('')['labels_list']

cluster_num_list = [2,4,8,16,32,64]
degree_list = [4,6,8,10,12,14]
idx_used_sh_list = []
idx_used_label_sh_list = []
idx_list_list = []
for cluster_num_idx in np.arange(len(cluster_num_list)):
    labels = labels_list_sh[cluster_num_idx]
    idx_used_sh = []
    idx_used_label_sh = []
    idx_list = []
    for ii in np.arange(cluster_num_list[cluster_num_idx]):
        idx_list.append([])
    for ii in np.arange(int(len(labels) / 4)):
        label_4 = labels[ii * 4:(ii + 1) * 4]
        dict_count = Counter(label_4).most_common(2)
        if len(dict_count) == 1:
            label_major = dict_count[0][0]
            idx_list[label_major].append(ii)
            idx_used_sh.append(ii)
            idx_used_label_sh.append(label_major)
        elif dict_count[0][1] > dict_count[1][1]:
            label_major = dict_count[0][0]
            idx_list[label_major].append(ii)
            idx_used_sh.append(ii)
            idx_used_label_sh.append(label_major)
    idx_used_sh_list.append(idx_used_sh)
    idx_used_label_sh_list.append(idx_used_label_sh)
    idx_list_list.append(idx_list)

gpu = 3
edge_degree_idx_now = 5
setproctitle.setproctitle("node_trans_"+str(edge_degree_idx_now)+"@zsy")#("node_trans_all@hsd")
torch.manual_seed(5)
patience_loss = 250
save_dir = '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for edge_degree_idx in [edge_degree_idx_now]:#np.arange(6):  # [4,6,8,10,12,14]
    print('edge_degree:', edge_degree_idx * 2 + 4)
    edge_index = edge_index_list[edge_degree_idx]
    edge_attr = edge_attr_list[edge_degree_idx]
    data_bj = Data(x=torch.tensor(kge, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))
    node_embedding_sh = \
    np.load('node_embedding_KL_log_layerscat_d'
            + str(degree_list[edge_degree_idx]) + '_c' + str(cluster_num_list[cluster_num_idx]) + '.npz')['node_embedding']
    for cluster_num_idx in np.arange(len(cluster_num_list)):
        print('cluster_num:', cluster_num_list[cluster_num_idx])

        cluster_idx_list = idx_list_list[cluster_num_idx]#[list(set(cluster_idx)) for cluster_idx in idx_list_list[cluster_num_idx]]
        cluster_centers = np.zeros((cluster_num_list[cluster_num_idx], node_embedding_sh.shape[1]))
        for ii in np.arange(cluster_num_list[cluster_num_idx]):
            cluster_idx = cluster_idx_list[ii]
            cluster_centers[ii, :] = node_embedding_sh[cluster_idx].sum(0)
        node_center_source = torch.tensor(cluster_centers, dtype=torch.float)
        cluster_label = torch.tensor(idx_used_label_sh_list[cluster_num_idx], dtype=torch.long)
        node_embedding_source = torch.tensor(node_embedding_sh[idx_used_sh_list[cluster_num_idx]], dtype=torch.float)

        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        model, data = Net(POI_cate_num=poi_cate_dtb_n2.shape[1], num_node_features=kge.shape[1]).to(
            device), data_bj.to(device)
        node_center_source, cluster_label, node_embedding_source = node_center_source.to(device), cluster_label.to(
            device), node_embedding_source.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        B_nc, B_lp, B_tc = 1e2, 1, 1  # 1, 1e-6, 1e-3
        criterion_nc = nn.KLDivLoss()

        loss_best = 1e9
        loss_best_it = 0
        loss_list = []
        loss_nc_list = []
        loss_tc_list = []
        for epoch in trange(1, 1001):
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            node_embedding = model.encode(data.x, data.edge_index)
            nc_results = model.node_classify(node_embedding)
            nc_labels = torch.tensor(poi_cate_dtb_n2, dtype=torch.float).to(device)
            loss_nc = criterion_nc(torch.log(nc_results + 1e-49), nc_labels)

            loss_tc = criterion_tc_bj_major(node_embedding, cluster_label, node_center_source, node_embedding_source)
            if loss_nc != 0:
                B_nc = 1./abs(loss_nc)
            if abs(loss_tc) > 0.04:
                B_tc = 1./abs(loss_tc)
            else:
                B_tc = B_nc

            loss = B_nc * loss_nc + B_tc * loss_tc  # + B_lp * loss_lp + B_tc * loss_tc
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            scheduler.step(loss)
            if epoch % 10 == 0:
                print(optimizer)
                print(
                f'Epoch: {epoch:05d}, Loss: {loss.item():.10f}, Loss_nc: {loss_nc.item():.10f}, Loss_tc: {loss_tc.item():.10f}')
            #if (abs(loss_his_nc - loss_nc) < 1e-7) & (abs(loss_his_tc - loss_tc) < 1e-7):
            #    break
            if loss_best > loss:
                loss_best = loss
                loss_best_it = epoch
                node_embedding_np = node_embedding.cpu().detach().numpy()
                torch.save(model.state_dict(), os.path.join(save_dir, 'Net' + str(edge_degree_idx * 2 + 4) + '_c' + str(
                    cluster_num_list[cluster_num_idx])))
            elif epoch - loss_best_it > patience_loss:
                break

            loss_list.append(loss.cpu().detach().numpy())
            loss_nc_list.append(loss_nc.cpu().detach().numpy())
            loss_tc_list.append(loss_tc.cpu().detach().numpy())


        np.savez(os.path.join(save_dir,
                              'node_embedding_KL_log_layerscat_d' + str(edge_degree_idx * 2 + 4) + '_c' + str(
                                  cluster_num_list[cluster_num_idx])),
                 node_embedding=node_embedding_np,
                 loss_list = np.array(loss_list),
                 loss_nc_list = np.array(loss_nc_list),
                 loss_tc_list = np.array(loss_tc_list),
                 epoch_num = loss_best_it)
