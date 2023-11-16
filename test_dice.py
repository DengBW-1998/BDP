import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
import deeprobust.graph.defense as dr
from Dice.dice import DICE
from sklearn.model_selection import train_test_split
from ProGNN.train import pro_GNN_Hyper,proGNNTrain
from EdgeAttention.edge_attention import *
import warnings
warnings.filterwarnings("ignore")

#the meaning of each hyperparamaters intruduced in BBAI.py
perturbations = -1 #Perturbation number
dim = 64 
window_size = 5 
n_node_pairs = 100000 
seed=0
threshold = 5 #Implicit relationship threshold
rate=-1
batch_size=64
train_model = 'netmf'

ptb_rate=20
dataset='dblp'
defense='bdp'
simFunc='euc'
dp=False
data=open("dice20_euc.txt",'w+')
thj=0.05

if dataset=='dblp':
    perturbations=3600
    rate=1
    thx=2.054
if dataset=='wiki':
    perturbations=3800
    rate=2
    thx=2.576

if ptb_rate==5:
    perturbations=perturbations//4

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
adj = standardize(adj)
#adj_nn = standardize(adj_nn)
emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
time_start=time.time()

labels = np.zeros((u+v))
labels[u:] = 1

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

def get_train_val_test(idx, train_size, val_size, test_size, stratify):

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test
    
def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
        labels = sparse_mx_to_torch_sparse_tensor(labels)
    else:
        labels = torch.LongTensor(np.array(labels))
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    return adj, features, labels

    
idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

device = 'cpu'
surrogate = dr.GCN(nfeat=32, nclass=2,
            nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

model = DICE(surrogate, nnodes=adj.shape[0],
    attack_structure=True, attack_features=False, device=device).to(device)
    
model.attack(adj, labels, n_perturbations=perturbations)
time_end=time.time()


adj_matrix_flipped = sp.csr_matrix(model.modified_adj)
adj_matrix_flipped[:u,:u]=0
adj_matrix_flipped[u:,u:]=0
    
if defense=='proGNN' or defense=='prognn':   
    adj_matrix_flipped=proGNNTrain(adj_matrix_flipped,u,v,emb0_u,emb0_v)
    #print(type(adj_matrix_flipped)) #torch
    adj_matrix_flipped=adj_matrix_flipped.detach().numpy()
    adj_matrix_flipped[adj_matrix_flipped>=0.5]=1
    adj_matrix_flipped[adj_matrix_flipped<0.5]=0
    adj_matrix_flipped=sp.csr_matrix(adj_matrix_flipped)

if defense=='bdp':
    adj_matrix_flipped,emb0_u,emb0_v = GCNSimDefense(emb0_u, emb0_v, u, v, adj_matrix_flipped[:u, u:], thx, thj,simFunc)
    adj_matrix_flipped[:u,:u] = adj_nn.todense()[:u,:u]
    adj_matrix_flipped[u:,u:] = adj_nn.todense()[u:,u:]
    adj_matrix_flipped = ImpAddEdges(emb0_u, emb0_v, u, v,adj_matrix_flipped,thj,simFunc)
    adj_matrix_flipped[:u, :u] = 0
    adj_matrix_flipped[u:, u:] = 0
    adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

if defense=='stable' or defense=='STABLE':
    adj_matrix_flipped=stable(np.row_stack((emb0_u,emb0_v)),adj_matrix_flipped)

for _ in range(5): 
    u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
    v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
    node_pairs = np.column_stack((u_node_pairs,v_node_pairs)) 
        
    if train_model=='netmf':
        embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u,u:]@adj_matrix_flipped[u:,:u], window_size, dim)
        embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:,:u]@adj_matrix_flipped[:u,u:], window_size, dim)
        embedding_imp = np.row_stack((embedding_u,embedding_v))
        embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
        embedding = (embedding_imp+embedding_exp)/2
    if train_model=='bgnn':
        bgnn = BGNNAdversarial(u,v,batch_size,adj_matrix_flipped[:u,u:],adj_matrix_flipped[u:,:u],emb0_u,emb0_v, dim_u,dim_v, dataset)
        embedding = bgnn.adversarial_learning()
    
    if dp:
        embedding=edges_train(adj_matrix_flipped[:u,u:],u,v,embedding)
        
    auc_score = evaluate_embedding_link_prediction(
        adj_matrix=adj_matrix_flipped, 
        node_pairs=node_pairs, 
        embedding_matrix=embedding
    )
    print('dice auc:{}'.format(auc_score))
    print('dice auc:{}'.format(auc_score),file=data)

print(simFunc)
print(simFunc,file=data)
print(defense)
print(defense,file=data)
print('dice') 
print('dice',file=data) 
print(dataset)
print(dataset,file=data)
print(time_end-time_start)
print(time_end-time_start,file=data)     
data.close()