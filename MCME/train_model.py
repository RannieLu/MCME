import torch
import torch.nn.functional as FF
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import torch.nn as nn
import joblib
from preprocessing import *
from model import HeriModel
import networkx as nx
import pickle
import torch
from argparse import ArgumentParser
import torch.nn.functional as F

parser = ArgumentParser()
parser.add_argument("--dataset", default="ckm")
parser.add_argument("--rate", default=0.8)
parser.add_argument("--learning_rate", default=0.01)
parser.add_argument("--alpha", default=1.0)
parser.add_argument("--beta", default=1.0)
parser.add_argument("--dim", default=128)
args = parser.parse_args()
dataset = args.dataset
rate = args.rate

def loss( model, node_h, layer_sim, adj, norm, pos_weight, features, alpha=0.1, b=0.1, r=1.0):
    base_embedding, base_mean, base_std, A_pred, v_embedding = model(features)
    loss = 0
    for i in range(1, model.layers+1):
        Z = v_embedding[i-1]
        pred = torch.sigmoid(torch.matmul(Z,Z.t())/(torch.norm(Z,dim=1)* torch.norm(Z.t(),dim=0)))
        tt = torch.Tensor(adj[i].todense())
        t_l = norm[i]* F.binary_cross_entropy(pred.view(-1), tt.view(-1), weight = pos_weight[i] )     
        loss += t_l
        for j in range(i+1, model.layers+1):
            tmp_inter = torch.Tensor([node_h[str(i)+'-'+str(j)]])
            tmp_layer = layer_sim[str(i)+'-'+str(j)]
            loss +=  alpha*torch.mm(tmp_inter,(torch.norm(v_embedding[i-1]- v_embedding[j-1],p=2,dim=-1, keepdim=True))).item()/(model.node_nums*model.node_nums) 
            loss += b*torch.norm(model.l_embedding[i-1]- model.l_embedding[j-1], p=2) * tmp_layer /model.layers

    tt = torch.Tensor(adj[0].todense())
    loss +=  norm[0]*FF.binary_cross_entropy(A_pred.view(-1), tt.view(-1), weight = pos_weight[0])
    kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*base_std - base_mean**2 - torch.exp(base_std)**2).sum(1).mean()
    loss -= kl_divergence
    return loss







try:
    new_train_edges = pickle.load(open("data/new_train_edges_" +dataset +"_" +str(rate)  +".pkl", 'rb'))
    new_test_edges = pickle.load(open("data/new_test_edges_" +dataset +"_" +str(rate)  +".pkl", 'rb'))
    new_test_edges_false = pickle.load(open("data/new_test_edges_false_" +dataset +"_" +str(rate)  +".pkl", 'rb'))

except:
    new_train_edges = joblib.load("data/new_train_edges" +dataset +"_" +str(rate)  +".pkl")
    new_test_edges = joblib.load("data/new_test_edges" +dataset +"_" +str(rate)  +".pkl")
    new_test_edges_false = joblib.load("data/new_test_edges_false_" +dataset +"_" +str(rate)  +".pkl")

N = np.max(new_train_edges[0])
layers = len(new_train_edges.keys())-1
new_adj_train = dict()
for l in new_train_edges.keys():
    data = np.ones(len(new_train_edges[l]))
    col = list()
    row = list()
    for edge in new_train_edges[l]:
        col.append(edge[0])
        row.append(edge[1])
    new_adj_train[l] = sp.csr_matrix((data, (col, row)), shape=(N,N))
    new_adj_train[l] = new_adj_train[l] + sp.eye(new_adj_train[l].shape[0])

# new_adj_norm = dict()

new_pos_weight = dict()
new_norm = dict()
new_weight_tensor = dict()
for layer_l in new_adj_train.keys():
    new_pos_weight[layer_l] = float(new_adj_train[layer_l].shape[0] * new_adj_train[layer_l].shape[0] - new_adj_train[layer_l].sum()) / new_adj_train[layer_l].sum()
    new_norm[layer_l] = new_adj_train[layer_l].shape[0] * new_adj_train[layer_l].shape[0] / float((new_adj_train[layer_l].shape[0] * new_adj_train[layer_l].shape[0] - new_adj_train[layer_l].sum()) * 2)
    weight_mask = torch.Tensor(new_adj_train[layer_l].todense()).view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = new_pos_weight[layer_l]
    new_weight_tensor[layer_l] = weight_tensor
tmp = sp.identity(new_adj_train[0].shape[0])   
features = sparse_to_tuple(tmp.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                        torch.FloatTensor(features[1]), 
                                        torch.Size(features[2]))
adj_norm_base = preprocess_graph(new_adj_train[0])
adj_norm_base = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_base[0].T), 
                                    torch.FloatTensor(adj_norm_base[1]), 
                                    torch.Size(adj_norm_base[2]))

    # tmp = sp.identity(new_adj_train[layer_l].shape[0])   
    # features[layer_l] = sparse_to_tuple(tmp.tocoo())


avg_roc = 0
avg_ap = 0
node_h, layer_sim = calc_node_h(new_adj_train,layers)
t3 = time.time()
print("calc node heri ", t3-t_2)
for e_i in range(5):
    t_1 = time.time()
    multi_roc = 0
    multi_ap = 0
    model = HeriModel(layers, N, adj_norm_base,node_size=128, layer_size=128)
    model.train()
    optimizer = Adam(model.parameters(), lr = args.learning_rate)
    t_1 = time.time()
    for epoch in range(2000):
        optimizer.zero_grad()
        loss_m = loss(model, node_h, layer_sim, new_adj_train, new_norm, new_weight_tensor,features, alpha=args.alpha, b =args.beta)
        loss_m.backward()
        optimizer.step()
        
        if epoch ==1999:
            for la in range(1, layers+1):
                base_embedding, base_mean, base_std, A_pred, v_embedding = model(features)
                Z= v_embedding[la-1]             
                pickle.dump(Z.cpu().detach().numpy(), open("model_results/" +dataset +"_" + str(la)+"_"+ str(rate)+"_mymodel" +".pkl", "wb"))
    
                test_roc, test_ap = get_lp_scores(new_test_edges[la], new_test_edges_false[la],Z)
                print("dataset ", dataset, "dim ",dim, " layer ", la,  "test_roc=", "{:.5f}".format(test_roc),
                "test_ap=", "{:.5f}".format(test_ap))
                multi_roc  += test_roc
                multi_ap += test_ap
                avg_roc += test_roc
                avg_ap += test_ap
    print("epoch time ", time.time()-t_1)
    print("-----------")
    print("dataset ", dataset," rate ", rate , " epoch ", e_i, " ", multi_roc/len(new_test_edges.keys()) , "  " , multi_ap/len(new_test_edges.keys()) )
print("----------------")
print("dataset ", dataset," multi ", rate, "  ",  avg_roc/(len(new_test_edges.keys())*5), ' ', avg_ap/(len(new_test_edges.keys())*5))


