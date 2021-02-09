import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

class VGAE(nn.Module):
    def __init__(self,num_nodes, adj,dim=128, hidden_dim=200):
        super(VGAE,self).__init__()
        self.dim= dim
        self.hidden_dim=hidden_dim
        self.base_gcn = GraphConvSparse(num_nodes, self.hidden_dim, adj)
        self.gcn_mean = GraphConvSparse(self.hidden_dim, self.dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(self.hidden_dim, self.dim, adj, activation=lambda x:x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        self.z = self.encode(X)
        self.A_pred = dot_product_decode(self.z)
        return self.z, self.A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)



class HeriModel(nn.Module):
    def __init__(self, layers, node_nums,  adj, node_size=128, layer_size=128):
        super(HeriModel, self).__init__()
        self.layers = layers
        self.node_nums = node_nums
        self.node_dim = node_size
        self.layer_dim = layer_size
        self.base_embedding = VGAE(node_nums, adj,dim=self.node_dim) 
        self.l_embedding = nn.Parameter(torch.FloatTensor(self.layers, self.layer_dim))
        nn.init.normal_(self.l_embedding.data, mean=0, std=1)
    

    def forward(self, X):
        z, A_pred = self.base_embedding(X)
        base_mean = self.base_embedding.mean
        base_std = self.base_embedding.logstd
        v_embedding = [torch.zeros((self.node_nums, self.node_dim)) for i in range(self.layers)]
        for i in range( self.layers):
            v_embedding[i] = torch.exp(base_std) + base_mean + self.l_embedding[i]
        return z, base_mean, base_std, A_pred, v_embedding





class BatchVGAE(nn.Module):
    def __init__(self,num_nodes, adj,dim=128, hidden_dim=200):
        super(BatchVGAE,self).__init__()
        self.dim= dim
        self.hidden_dim=hidden_dim
        self.base_gcn = GraphConvSparse(num_nodes, self.hidden_dim, adj)
        self.gcn_mean = GraphConvSparse(self.hidden_dim, self.dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(self.hidden_dim, self.dim, adj, activation=lambda x:x)
    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        self.z = self.encode(X)
        return self.z


class BatchHeriModel(nn.Module):
    def __init__(self, layers, node_nums,  adj, node_size=60, layer_size=60):
        super(BatchHeriModel, self).__init__()
        self.layers = layers
        self.node_nums = node_nums
        self.node_dim = node_size
        self.layer_dim = layer_size
        self.base_embedding = BatchVGAE(node_nums, adj,dim=self.node_dim) 
        self.l_embedding = nn.Parameter(torch.FloatTensor(self.layers, self.layer_dim))
        nn.init.normal_(self.l_embedding.data, mean=0, std=1)
    

    def forward(self, X):
        z = self.base_embedding(X)
        base_mean = self.base_embedding.mean
        base_std = self.base_embedding.logstd
        v_embedding = [torch.zeros((self.node_nums, self.node_dim)) for i in range(self.layers)]
        for i in range( self.layers):
            v_embedding[i] = torch.exp(base_std) + base_mean + self.l_embedding[i]
        return z, base_mean, base_std, v_embedding

