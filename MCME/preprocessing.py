
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import random
from numba import jit

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)




def get_lp_scores(edges_pos, edges_neg, Z):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    # print("adj_rec ", adj_rec)
    for e in edges_pos:

        vector1 = Z[e[0]]
        vector2 = Z[e[1]]
        preds.append(torch.sigmoid(torch.dot(vector1,vector2)/(torch.norm(vector1)* torch.norm(vector2))).detach().numpy())

    preds_neg = []
    neg = []
    for e in edges_neg:
        vector1 = Z[e[0]]
        vector2 = Z[e[1]]
        preds_neg.append(torch.sigmoid(torch.dot(vector1,vector2)/(torch.norm(vector1)* torch.norm(vector2))).detach().numpy())

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

@jit(nopython=True) 
def calc_h(deg_h, tmp_inter): 
    tmp_node_h = [0]*len(deg_h)
    for k in range(len(deg_h)):                
        node_deg_h = np.exp(-deg_h[k])  
        tmp_node_inter = tmp_inter[k]
        if tmp_node_inter == 0:
            tmp_node_h[k] = -1/(1+np.exp(deg_h[k]))
        else:
            inter_h = 1/(1+np.exp(-(tmp_node_inter)))
            if node_deg_h >0:
                tmp_node_h[k] = 2/(1/inter_h + 1/node_deg_h)
            else:
                tmp_node_h[k] = inter_h
    return tmp_node_h



def calc_node_h(new_adj_train,layers):
    node_h = dict()
    layer_sim = dict()
    for i in range(1, layers+1):
        # print("new_adj_train",new_adj_train[i])
        deg_i = np.sum(new_adj_train[i].toarray(), axis=1)
        # print("deg_i ", deg_i)
        #out_degree = np.sum(self.adjacency, axis=0).reshape((self.adjacency.shape[0], 1))
        for j in range(i+1, layers+1):         
            deg_j = np.sum(new_adj_train[j].toarray(), axis=1)
            deg_h = np.exp(-abs(deg_i-deg_j))
            tmp_1 = np.sum(new_adj_train[i].toarray() + new_adj_train[j].toarray(), axis=1)
            tmp_2 = np.sum(abs(new_adj_train[i].toarray() - new_adj_train[j].toarray()), axis=1)
            tmp_inter = (tmp_1 - tmp_2)/2 -1
            # print("deg_j ", deg_j)
            # print("tmp_inter ", tmp_inter)
        
            node_inter = sum(tmp_inter)
            # print(node_inter)
            node_union = np.sum(new_adj_train[i].toarray(), axis=1) + np.sum(new_adj_train[j].toarray(), axis=1)
            node_union = node_union - tmp_inter - 2
            tmp_node_h = calc_h(deg_h, tmp_inter)


            tmp_layer_sim = node_inter/sum(node_union)
            layer_name = str(i) + "-" + str(j)
            node_h[layer_name] = tmp_node_h
            layer_sim[layer_name] = tmp_layer_sim
    return node_h, layer_sim




class AliasRandom:
    def __init__(self,weights):
        n = len(weights)
        self.n = n

        #sum_weights = sum(weights)
        prob = [w*n for w in weights]
        inx = [-1]*n
        small_block = [i for i,p in enumerate(prob) if p<1]
        large_block = [i for i,p in enumerate(prob) if p>=1]
        while small_block and large_block:
            i = small_block.pop()
            k = large_block[-1]
            inx[i] = k
            prob[k] = prob[k] - (1-prob[i])
            if(prob[k]<1):
                small_block.append(k)
                large_block.pop()
        self.prob = prob
        self.inx = inx


    def sample(self):
        i = np.random.randint(0,self.n - 1)
        u = np.random.uniform(0,1)
        if(u<self.prob[i]):
            sample = i
        else:
            sample = self.inx[i]
        return sample




class Data:
    def __init__(self,data):
        self.starting = 0
        self.dataset = data
        self.shuffle()
        self.powered_degree, self.degree = self.cal_powered_degree2()
        self.sampler = AliasRandom(self.powered_degree)
        self.length = np.shape(data)[0]

    def cal_powered_degree2(self):
        X = self.dataset
        unique, counts = np.unique(X, return_counts=True)
        powered_degree = np.power(counts,3/4)
        powered_degree = powered_degree/powered_degree.sum()
    #        print(unique[0:10])
    #        print(counts[0:10])
        return powered_degree, counts

    def shuffle(self):
        np.random.shuffle(self.dataset)
    

    def gen_mini_batch(self, batch_size, k):
        ending = self.starting + batch_size
        if ending >= self.length :
            ending = self.length
        source_list = self.dataset[self.starting:ending,0]
        target_list = self.dataset[self.starting:ending,1]
        target_list_neg = [0]*len(source_list)*k
        for i in range(len(source_list)*k):
            target_list_neg[i] = self.sampler.sample()
        source_list_neg = np.repeat(source_list,k)
        target_list_neg = np.asarray(target_list_neg)
        source_list_final = np.concatenate((source_list,source_list_neg))
        target_list_final = np.concatenate((target_list,target_list_neg))
        labels = [1] * len(source_list) + [0] *  len(source_list) * k
        self.starting = ending
        return source_list_final, target_list_final, labels