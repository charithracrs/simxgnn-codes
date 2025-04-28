import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch.nn.utils.weight_norm import weight_norm
from utils import multi2big_batch


class Protein_graph(nn.Module):
    def __init__(self,args,pos_ppi_index,d_in_dim = 7,hidden=64):
        super(Protein_graph, self).__init__()
        self.args = args
        self.mlp_in_dim = args.mlp_dim[0]
        self.mlp_hidden_dim = args.mlp_dim[1]
        self.mlp_out_dim = args.mlp_dim[2]

        self.conv1 = SAGEConv(d_in_dim, hidden, args='sum')
        self.bn1 = nn.BatchNorm1d(hidden)
        self.sag1 = SAGPooling(hidden, 0.5) # ori: 0.5
        self.fc1 = nn.Linear(hidden, hidden)
        self.graph_pre_decoder = graph_pre_decoder(self.mlp_in_dim, self.mlp_hidden_dim, self.mlp_out_dim)

    def forward(self, protein_data,pre_ppi_index):
        """
        Input all nodes at once and learn embs in batches
        """
        batch_size = 128
        steps = int(1512 / batch_size)
        protein_index_all = torch.arange(1512)
        for step in range(steps):

            if step == steps - 1:
                protein_index = protein_index_all[batch_size * step:]
            else:
                protein_index = protein_index_all[batch_size * step:batch_size * (step + 1)]

            protein_x_num_pre = protein_data[1][:batch_size * step].sum()  # The number of atoms before the current batch
            protein_edge_num_pre = protein_data[3][:batch_size * step].sum()  # The number of edges before the current batch

            protein_x_num_end = protein_x_num_pre + protein_data[1][
                protein_index].sum()  # the max atom index of the current batch
            protein_edge_num_end = protein_edge_num_pre + protein_data[3][
                protein_index].sum()  # the max edge index o the current batch

            # get each iter protein edges and features
            p_x_num_index_batch = protein_data[1][protein_index]
            p_x_batch = protein_data[0][protein_x_num_pre:protein_x_num_end, :]
            p_edge_num_index_batch = protein_data[3][protein_index]

            p_x_num_index_batch = multi2big_batch(p_x_num_index_batch)
            p_x_num_index_batch = p_x_num_index_batch.to(self.args.device)

            p_edge_batch = protein_data[2][:, protein_edge_num_pre:protein_edge_num_end]
            # change edge id index in a batch
            p_edge_batch = p_edge_batch - protein_x_num_pre

            x_temp = self.conv1(p_x_batch, p_edge_batch)
            x_temp = self.fc1(x_temp)
            x_temp = F.relu(x_temp)
            x_temp = self.bn1(x_temp)

            y = self.sag1(x_temp, p_edge_batch, batch=p_x_num_index_batch)  # ratio 0.5, remove amino acids by learning their importance
            x_temp = y[0]
            batch_temp = y[3]
            edge_index_temp = y[1]

            if step == 0:
                x = global_mean_pool(x_temp, batch_temp)
            else:
                x_temp = global_mean_pool(x_temp, batch_temp)
                x = torch.cat((x, x_temp))

        ppi_pre = self.graph_pre_decoder(x, pre_ppi_index)

        return x,ppi_pre


class Drug_graph(nn.Module):
    def __init__(self, args,pos_ddi_index,in_dim=75, padding=True, hidden_dim=64, activation=None):
        super(Drug_graph, self).__init__()
        self.init_transform = nn.Linear(in_dim, hidden_dim, bias=True)
        self.mlp_in_dim = args.mlp_dim[0]
        self.mlp_hidden_dim = args.mlp_dim[1]
        self.mlp_out_dim = args.mlp_dim[2]
        self.gcn = SAGEConv(hidden_dim, hidden_dim, args='sum')
        self.graph_pre_decoder = graph_pre_decoder(self.mlp_in_dim, self.mlp_hidden_dim, self.mlp_out_dim)

    def forward(self, drug_x, drug_edge,batch,target_ddi_index):
        node_feats = drug_x
        node_feats = self.init_transform(node_feats)
        node_feats = self.gcn(node_feats, drug_edge)
        node_feats = global_mean_pool(node_feats, batch)
        ddi_pre = self.graph_pre_decoder(node_feats, target_ddi_index)

        return node_feats, ddi_pre

class graph_pre_decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,):
        super(graph_pre_decoder, self).__init__()

        self.input_dim = in_dim

        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=4,
                               kernel_size=3,
                               stride=1,
                               padding=1, )
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.linear1 = torch.nn.Linear(int(self.input_dim * 4), 64)
        self.linear2 = torch.nn.Linear(64, 1)
        self.feat_drop = nn.Dropout(0.)

        self.linear3 = torch.nn.Linear(self.input_dim, 1024)
        self.linear4 = torch.nn.Linear(1024, 500)


    def forward(self, x, index):
        x1 = x[index[0, :]]
        x2 = x[index[1, :]]
        x = torch.cat((x1, x2), dim=-1)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pooling1(x)
        x = torch.flatten(x, 1, 2)
        x = F.dropout(x, 0.1)

        output = self.linear1(x)
        output = F.relu(output)
        output = self.linear2(output)
        output = output.view(-1)
        output = torch.sigmoid(output)

        return output

class Graph_encoder(nn.Module):
    def __init__(self, args, train_pos_ddi_index,train_pos_ppi_index):
        super(Graph_encoder, self).__init__()

        self.drug_extractor = Drug_graph(args,pos_ddi_index=train_pos_ddi_index,)
        self.protein_extractor = Protein_graph(args,pos_ppi_index=train_pos_ppi_index,)

    def forward(self, model_type,d_data,p_data,target_ddi_index,target_ppi_index,mode="train"):
        if model_type =="pretrain_drug":
            embs, pre_score = self.drug_extractor(d_data[0],d_data[2],d_data[1], target_ddi_index)
        elif model_type == "pretrain_protein":
            embs,pre_score = self.protein_extractor(p_data,target_ppi_index)

        return embs, pre_score
