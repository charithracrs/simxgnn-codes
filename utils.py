import os.path

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

def accuracy(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = 0.5
    outputs = outputs.ge(threshold_value).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    if labels.size() == 0:
        return np.nan
    return corrects.sum().item() / labels.size()[0]


def precision(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return precision_score(labels, outputs)


def recall(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs)

def f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.5).type(torch.float64)
    labels = labels.type(torch.float64)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


def auc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return roc_auc_score(labels, outputs)

def aupr(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)

def roc_auc_curve(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return (labels, outputs)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def load_sim(device):

    "data/similarity/sim_mat_drug_disease.txt",
    "data/similarity/sim_mat_drug_drug.txt",
    "data/similarity/sim_mat_drug.txt",
    "data/similarity/sim_mat_drug_se.txt"
    sim_mat_drug_disease = np.loadtxt("./data/similarity/sim_mat_drug_disease.txt")
    sim_mat_drug_drug = np.loadtxt("./data/similarity/sim_mat_drug_drug.txt")
    sim_mat_drug = np.loadtxt("./data/similarity/sim_mat_drug.txt")
    sim_mat_drug_se = np.loadtxt("./data/similarity/sim_mat_drug_se.txt")
    sim_mat_drug_protein = np.loadtxt("./data/similarity/sim_mat_drug_protein.txt")

    "data/similarity/sim_mat_protein_disease.txt",
    "data/similarity/sim_mat_protein_protein.txt",
    "data/similarity/sim_mat_protein_normalize.txt"
    sim_mat_protein_disease = np.loadtxt("./data/similarity/sim_mat_protein_disease.txt")
    sim_mat_protein_protein = np.loadtxt("./data/similarity/sim_mat_protein_protein.txt")
    sim_mat_protein = np.loadtxt("./data/similarity/sim_mat_protein.txt")
    sim_mat_protein_drug = np.loadtxt("./data/similarity/sim_mat_protein_drug.txt")
    sim_mat_protein = sim_mat_protein / 100

    """Put the drug similarity network together and the protein similarity network together"""
    DrugSimNet = []
    ProteinSimNet = []
    DrugSimNet_ori = []
    ProteinSimNet_ori = []
    DrugSimNet.append(sim_mat_drug_disease)
    DrugSimNet.append(sim_mat_drug_drug)
    DrugSimNet.append(sim_mat_drug)
    DrugSimNet.append(sim_mat_drug_se)
    DrugSimNet.append(sim_mat_drug_protein)

    ProteinSimNet.append(sim_mat_protein_disease)
    ProteinSimNet.append(sim_mat_protein_protein)
    ProteinSimNet.append(sim_mat_protein)
    ProteinSimNet.append(sim_mat_protein_drug)

    DrugSimNet_ori.append(sim_mat_drug_disease)
    DrugSimNet_ori.append(sim_mat_drug_drug)
    DrugSimNet_ori.append(sim_mat_drug)
    DrugSimNet_ori.append(sim_mat_drug_se)
    DrugSimNet_ori.append(sim_mat_drug_protein)

    ProteinSimNet_ori.append(sim_mat_protein_disease)
    ProteinSimNet_ori.append(sim_mat_protein_protein)
    ProteinSimNet_ori.append(sim_mat_protein)
    ProteinSimNet_ori.append(sim_mat_protein_drug)

    idx_drug = np.arange(len(DrugSimNet[0]))
    for i, net in enumerate(DrugSimNet):
        net = np.where(net > 0.5, 1, 0)
        net[idx_drug,idx_drug] = 0
        net = preprocess_adj(net)
        net = torch.Tensor(net)
        DrugSimNet[i] = net.to(device)

        net_ori = DrugSimNet_ori[i]
        # net_ori[idx_drug, idx_drug] = 0
        # net_ori = preprocess_adj(net_ori)
        net_ori = torch.Tensor(net_ori)
        DrugSimNet_ori[i] = net_ori.to(device)

    idx_protein = np.arange(len(ProteinSimNet[0]))
    for i, net in enumerate(ProteinSimNet):
        net = np.where(net > 0.5, 1, 0)
        net[idx_protein, idx_protein] = 0
        net = preprocess_adj(net)
        net = torch.Tensor(net)
        ProteinSimNet[i] = net.to(device)

        net_ori = ProteinSimNet_ori[i]
        # net_ori[idx_protein, idx_protein] = 0
        # net_ori = preprocess_adj(net_ori)
        net_ori = torch.Tensor(net_ori)
        ProteinSimNet_ori[i] = net_ori.to(device)

    return DrugSimNet, ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori


def multi2big_x(x_path):
    x_ori = torch.load(x_path)
    x_cat = torch.zeros(1, x_ori[0].shape[1])
    x_num_index = torch.zeros(len(x_ori))
    for i in range(len(x_ori)):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,len(x_num_index)):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.type(torch.int64)
    return batch

def multi2big_edge(edge_path, num_index):
    if edge_path[-2:] == "py":
        edge_ori = np.load(edge_path, allow_pickle=True)
        for i in range(len(edge_ori)):
            edge_ori[i] = torch.tensor(edge_ori[i])
        edge_ori = list(edge_ori)
    elif edge_path[-2:] =="pt":
        edge_ori = torch.load(edge_path)
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(len(num_index))
    for i in range(len(num_index)):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        if not len(edge_index_p) == 2:
            edge_index_p = torch.tensor(edge_index_p.T)
        else:
            edge_index_p = torch.tensor(edge_index_p)
        edge_num_index[i] = torch.tensor(edge_index_p.shape[1])
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def drug_multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,len(x_num_index)):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch


def drug_multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(len(num_index))
    for i in range(len(num_index)):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def get_ppi_ddi(ppi_path,ddi_path):
    mat_ppi = np.loadtxt(ppi_path)
    mat_ddi = np.loadtxt(ddi_path)
    ppi_index = np.where(mat_ppi == 1)
    ddi_index = np.where(mat_ddi == 1)
    ppi = np.ones(shape=(len(ppi_index[0]),3))
    ddi = np.ones(shape=(len(ddi_index[0]),3))

    ppi[:,0] = ppi_index[0]
    ppi[:, 1] = ppi_index[1]
    ppi = ppi.astype(int)

    ddi[:, 0] = ddi_index[0]
    ddi[:, 1] = ddi_index[1]
    ddi = ddi.astype(int)
    return ppi,ddi


def get_train_pos_from_fold(pos_fold,fold_num,device,row_num, col_num):
    train_pos  = [0]
    for i in range(len(pos_fold)):
        if not isinstance(pos_fold[0], torch.Tensor):
            temp_fold = torch.tensor(pos_fold[i])
        else:
            temp_fold = pos_fold[i]
        if i == fold_num:
            continue
        elif len(train_pos) == 1:
            train_pos = temp_fold
        else:
             train_pos = torch.cat((train_pos,temp_fold),dim=-1)

    train_pos_matrix = torch.zeros((row_num, col_num)).to(device)
    train_pos_matrix[train_pos[0,:],train_pos[1,:]] = 1

    return train_pos.to(device),train_pos_matrix

