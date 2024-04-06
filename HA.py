from deeprobust.graph.data import Dataset
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
from sklearn.model_selection import train_test_split
from ProGNN.train import pro_GNN_Hyper,proGNNTrain
from EdgeAttention.edge_attention import *

n_flips = -1 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
rate=-1
n_node_pairs = 300000 #Number of test edges
threshold = 5 #Implicit relationship threshold
batch_size=64
train_model='netmf'

ptb_rate=5
dataset='dblp'
defense='bdp'
simFunc='jac'
dp=False
thj=0.05
filename='HA'+str(ptb_rate)+'_'+defense+'_'+str(dp)+'_'+dataset+'_'+simFunc+'.txt'
data=open(filename,'w+')

if dataset=='dblp':
    n_flips=3600
    rate=1
    thx=2.054
if dataset=='wiki':
    n_flips=3800
    rate=2
    thx=2.576

if ptb_rate==5:
    n_flips=n_flips//4

n_candidates=10*n_flips #Number of candidate perturbed edges

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)  
adj = standardize(adj)

labels = np.zeros((u+v))
labels[u:] = 1

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

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

device = 'cpu'

adj = adj.todense()

idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled = np.union1d(idx_val, idx_test)

    
def compute_lambda(adj, idx_train, idx_test):
    num_all = adj.sum().item() / 2
    train_train = adj[idx_train][:, idx_train].sum().item() / 2
    test_test = adj[idx_test][:, idx_test].sum().item() / 2
    train_test = num_all - train_train - test_test
    return train_train / num_all, train_test / num_all, test_test / num_all

def heuristic_attack(adj, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2):
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_unlabeled)
    degree = adj.sum(dim=1).to('cpu')
    canditate_train_idx = idx_train[degree[idx_train] < (int(degree.mean()) + 1)]
    candidate_test_idx = idx_test[degree[idx_test] < (int(degree.mean()) + 1)]
    #     candidate_test_idx = idx_test
    perturbed_adj = adj.clone()
    cnt = 0
    train_ratio = lambda_1 / (lambda_1 + lambda_2)
    n_train = int(n_perturbations * train_ratio)
    n_test = n_perturbations - n_train
    while cnt < n_train:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(canditate_train_idx, 1)
        if labels[node_1] != labels[node_2] and adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1

    cnt = 0
    while cnt < n_test:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(candidate_test_idx, 1)
        if labels[node_1] != labels[node_2] and perturbed_adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1
    return perturbed_adj


# Generate perturbations
lambda_1, lambda_2, lambda_3 = compute_lambda(adj, idx_train, idx_unlabeled)
time_start=time.time()
adj_matrix_flipped = heuristic_attack(torch.from_numpy(adj).float(), n_flips, idx_train, idx_unlabeled, lambda_1, lambda_2)
time_end=time.time()
adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

if defense == 'proGNN' or defense == 'prognn':
    adj_matrix_flipped = proGNNTrain(adj_matrix_flipped, u, v, emb0_u, emb0_v)
    # print(type(adj_matrix_flipped)) #torch
    adj_matrix_flipped = adj_matrix_flipped.detach().numpy()
    adj_matrix_flipped[adj_matrix_flipped >= 0.5] = 1
    adj_matrix_flipped[adj_matrix_flipped < 0.5] = 0
    adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

if defense == 'bdp':
    adj_matrix_flipped, emb0_u, emb0_v = GCNSimDefense(emb0_u, emb0_v, u, v, adj_matrix_flipped[:u, u:], thx, thj,
                                                       simFunc)
    adj_matrix_flipped[:u, :u] = adj_nn.todense()[:u, :u]
    adj_matrix_flipped[u:, u:] = adj_nn.todense()[u:, u:]
    adj_matrix_flipped = ImpAddEdges(emb0_u, emb0_v, u, v, adj_matrix_flipped, thj, simFunc)
    adj_matrix_flipped[:u, :u] = 0
    adj_matrix_flipped[u:, u:] = 0
    adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

if defense == 'stable' or defense == 'STABLE':
    adj_matrix_flipped = stable(np.row_stack((emb0_u, emb0_v)), adj_matrix_flipped)


for _ in range(5):
    u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
    v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
    node_pairs = np.column_stack((u_node_pairs,v_node_pairs))

    adj_matrix_flipped[:u,:u]=0
    adj_matrix_flipped[u:,u:]=0

    if train_model=='netmf':
        embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u,u:]@adj_matrix_flipped[u:,:u], window_size, dim)
        embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:,:u]@adj_matrix_flipped[:u,u:], window_size, dim)
        embedding_imp = np.row_stack((embedding_u,embedding_v))
        embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
        embedding = (embedding_imp+embedding_exp)/2

    if dp:
        embedding=edges_train(adj_matrix_flipped[:u,u:],u,v,embedding)

    auc_score = evaluate_embedding_link_prediction(
        adj_matrix=adj_matrix_flipped,
        node_pairs=node_pairs,
        embedding_matrix=embedding
    )
    print('HA auc:{:.5f}'.format(auc_score))
    print('HA auc:{:.5f}'.format(auc_score), file=data)

print(train_model)
print(train_model, file=data)
print('HA') 
print('HA', file=data)
print(dataset)
print(dataset, file=data)
print(time_end-time_start)
print(time_end - time_start, file=data)
data.close()

