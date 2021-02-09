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
from model import BatchHeriModel
import networkx as nx
import pickle
import torch
from argparse import ArgumentParser
import torch.nn.functional as F
import math 

parser = ArgumentParser()
parser.add_argument("--dataset", default="ckm")
parser.add_argument("--dataset", default="ckm")
parser.add_argument("--rate", default=0.8)
parser.add_argument("--learning_rate", default=0.01)
parser.add_argument("--batch_size", default=5000)
parser.add_argument("--neg", default=5)
parser.add_argument("--alpha", default=1.0)
parser.add_argument("--beta", default=1.0)
parser.add_argument("--node_dim", default=128)
parser.add_argument("--layer_dim", default=128)
args = parser.parse_args()
dataset = args.dataset
rate = args.rate

try:
    training_data_by_type = pickle.load(open("other_model/data/train_edges_other_model_" +dataset +"_" +str(rate)  +".pkl", 'rb'))
    selected_true_edges  = pickle.load(open("other_model/data/test_edges_other_model_" +dataset +"_" +str(rate)  +".pkl", 'rb'))
    selected_false_edges = pickle.load(open("other_model/data/test_false_edges_other_model_" +dataset +"_" +str(rate)  +".pkl", 'rb'))
except:
    training_data_by_type = joblib.load("data/train_edges_other_model" +dataset +"_" +str(rate)  +".pkl")
    selected_true_edges  = joblib.load("data/test_edges_other_model" +dataset +"_" +str(rate)  +".pkl")
    selected_false_edges = joblib.load("data/test_false_edges_other_model" +dataset +"_" +str(rate)  +".pkl")

layersNum = len(training_data_by_type.keys())-1
N = np.max(training_data_by_type[0])+1
print(N)
new_adj_train = dict()
new_pos_weight = dict()
new_norm = dict()
new_weight_tensor = dict()
for l in training_data_by_type.keys():
    data = np.ones(len(training_data_by_type[l]))
    col = list()
    row = list()
    for edge in training_data_by_type[l]:
        col.append(edge[0])
        row.append(edge[1])
    new_adj_train[l] = sp.csr_matrix((data, (col, row)), shape=(N,N))
    new_adj_train[l] = new_adj_train[l] + sp.eye(new_adj_train[l].shape[0])

    new_pos_weight[l] = float(new_adj_train[l].shape[0] * new_adj_train[l].shape[0] - new_adj_train[l].sum()) / new_adj_train[l].sum()
    new_norm[l] = new_adj_train[l].shape[0] * new_adj_train[l].shape[0] / float((new_adj_train[l].shape[0] * new_adj_train[l].shape[0] - new_adj_train[l].sum()) * 2)

tmp = sp.identity(N)   
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
tmp = list()

node_h, layer_sim = calc_node_h(new_adj_train,layersNum)
new_adj_train = dict()
numOfNodes = N
avg_roc = 0
avg_ap = 0
for e_i in range(5):

    multi_roc = 0
    multi_ap = 0
    model = BatchHeriModel(layersNum, N, adj_norm_base ,node_size=args.node_dim, layer_size=args.layer_dim)
    model.train()
    optimizer = Adam(model.parameters(), lr = args.learning_rate)
    
    for epoch in range(2000):

        optimizer.zero_grad()
        loss_total = 0
        base_embedding, base_mean, base_std, v_embedding = model(features)
        t = time.time()
        t_1 = time.time()
        for layer_b in training_data_by_type.keys():

            ep_data = Data(training_data_by_type[layer_b])
            batch_size = args.batch_size
            k =args.neg
            for batch_i in range( int(ep_data.length/batch_size) +1):
                u_i , u_j ,label = ep_data.gen_mini_batch(batch_size, k)
                if layer_b == 0:
                    Z_a = base_embedding[u_i]
                    Z_b = base_embedding[u_j]   
                else:                      
                    Z_a = v_embedding[layer_b-1][u_i]
                    Z_b = v_embedding[layer_b-1][u_j]

                pred = torch.sigmoid(torch.sum(torch.mul(Z_a,Z_b),dim=1)/(torch.norm(Z_a,dim=1)* torch.norm(Z_b,dim=1)))
                tt = torch.Tensor(label)
                weight_mask = torch.Tensor(label) == 1
                weight_tensor = torch.ones(len(pred))
                weight_tensor[weight_mask] =new_pos_weight[layer_b]
                t_l = new_norm[layer_b]*F.binary_cross_entropy(pred.view(-1), tt, weight=weight_tensor)
        
                loss_total += t_l 
                if  layer_b == 0:
                    kl_divergence = 0.5/ pred.size(0) * (1 + 2*base_std - base_mean**2 - torch.exp(base_std)**2).sum(1).mean()
                    loss_total -= kl_divergence
        t_2 = time.time() 
        print("gen time ", t_2- t_1) 
        for i in range(1,model.layers+1):

            for j in range(i+1, model.layers+1):
                tmp_inter = torch.Tensor([node_h[str(i)+'-'+str(j)]])
                tmp_layer = layer_sim[str(i)+'-'+str(j)]
                loss_total +=  args.alpha*torch.mm(tmp_inter,(torch.norm(v_embedding[i-1]- v_embedding[j-1],p=2,dim=-1, keepdim=True))).item()
                loss_total += args.beta*torch.norm(model.l_embedding[i-1]- model.l_embedding[j-1], p=2) * tmp_layer 
        print("one epoch time ", time.time() -t, loss_total)  
        loss_total.backward()
        optimizer.step()
        if epoch == 1999:
            for la in range(1, layersNum+1):
                base_embedding, base_mean, base_std, v_embedding = model(features)
                Z= v_embedding[la-1]

                pickle.dump(Z, open("model_results/" +dataset +"_" + str(la)+"_"+ str(rate)+"_mymodel" +".pkl", "wb"))


                results,test_roc, test_ap = get_lp_scores(selected_true_edges[la], selected_false_edges[la],Z)

    
                print("dataset ", dataset, "rate ",rate, " layer ", la,  "test_roc=", "{:.5f}".format(test_roc),
                "test_ap=", "{:.5f}".format(test_ap))
                multi_roc  += test_roc
                multi_ap += test_ap
                avg_roc += test_roc
                avg_ap += test_ap
    print("-----------")
    print("rate ", rate , " epoch ", e_i, " ", multi_roc/len(selected_true_edges.keys()) , "  " , multi_ap/len(selected_true_edges.keys()) )
print("----------------")
# print("single ", single_roc,  ' ', single_ap)
print("multi ", rate, "  ",  avg_roc/(len(selected_true_edges.keys())*5), ' ', avg_ap/(len(selected_true_edges.keys())*5))



