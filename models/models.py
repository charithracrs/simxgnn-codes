import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.model_pretrain import Graph_encoder
from models.model_reconstruct import model_reconstruct

class CCL_ASPS(nn.Module):
    def __init__(self,args,train_pos_ddi_index,train_pos_ppi_index,
                 drug_emb, protein_emb,DrugSimNet, ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori):
        super(CCL_ASPS,self).__init__()
        self.gcn_drug_outdim = args.drug_dim
        self.gcn_protein_outdim = args.protein_dim
        self.input_dim = self.gcn_drug_outdim + self.gcn_protein_outdim
        self.args = args
        self.cnn = Model_CNN(self.input_dim, args.hidden_dim, self.gcn_drug_outdim, self.gcn_protein_outdim)
        if args.model_type == "pretrain_drug" or args.model_type == "pretrain_protein":
            self.Graph_encoder = Graph_encoder(args,train_pos_ddi_index,train_pos_ppi_index)
        if args.model_type =="CCL-ASPS":
            self.model_reconstruct = model_reconstruct(args, drug_emb, protein_emb, DrugSimNet, ProteinSimNet,
                                                       DrugSimNet_ori, ProteinSimNet_ori)

    def forward(self,d_data,p_data,target_ddi_index,target_ppi_index,
                dti_index,dti_labels,train_pos_dti):
        model_type = self.args.model_type
        if model_type=="pretrain_drug" or model_type =="pretrain_protein":
            embs,pre_score = self.Graph_encoder(model_type,d_data,p_data,target_ddi_index,target_ppi_index)
            return embs,pre_score
        if model_type =="CCL-ASPS":
            drug_embs_sim, protein_embs_sim, loss_contrast = self.model_reconstruct(dti_index, dti_labels, train_pos_dti)
            output = self.cnn(drug_embs_sim, protein_embs_sim, dti_index, dti_labels)
            return output,loss_contrast

class Model_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, drug_dim,protein_dim):
        super(Model_CNN, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(in_channels=1,out_channels=4, kernel_size=3,stride=1,padding=1, )
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.linear1 = torch.nn.Linear(int(self.input_dim*2), 128)
        self.linear2 = torch.nn.Linear(128,1)
        self.feat_drop = nn.Dropout(0.)

        self.linear3 = torch.nn.Linear(self.input_dim, 1024)
        self.linear4 = torch.nn.Linear(1024, 500)

    def forward(self,drug_embs, protein_embs, index, labels):

        drug_selected = drug_embs[index[0,:]]
        protein_seleced = protein_embs[index[1,:]]
        pair_nodes_features = torch.cat((drug_selected, protein_seleced),dim=-1)

        pair_nodes_features = pair_nodes_features.unsqueeze(1)
        pair_nodes_features = self.conv1(pair_nodes_features)
        pair_nodes_features = self.pooling1(pair_nodes_features)
        pair_nodes_features = torch.flatten(pair_nodes_features, 1, 2)
        pair_nodes_features = F.dropout(pair_nodes_features, 0.1)
        output = self.linear1(pair_nodes_features)

        output = F.relu(output)
        output = self.linear2(output)
        output = output.view(-1)
        output = torch.sigmoid(output)

        return output