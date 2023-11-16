import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
import scipy.sparse as sp
from ProGNN.train import pro_GNN_Hyper,proGNNTrain
from EdgeAttention.edge_attention import *

import warnings
warnings.filterwarnings("ignore")

#the meaning of each hyperparamaters intruduced in BBAI.py
n_flips = 1 
n_candidates=10*n_flips 
window_size=5
n_node_pairs = 100000
dim = 64 #Embedding vector dimension
seed=0
iteration=1
batch_size=64
rate=1
threshold = 5 #Implicit relationship threshold
train_model = 'netmf'

ptb_rate=20
dataset='dblp'
defense='bdp'
dp=False
simFunc='euc'
data=open("cln_euc.txt",'w+')
thj=0.05

if dataset=='dblp':
    rate=1
    thx=2.054
if dataset=='wiki':
    rate=2
    thx=2.576
    

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
adj = standardize(adj)
adj_nn = standardize(adj_nn)

emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
#print(type(emb0_u)) #np

#print(emb0_u[:10])
time_start=time.time()
adj_matrix_flipped=adj

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

for ite in range(iteration):
    print("\n",file=data)
    ### Link Prediction
    for _ in range(5):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))
        print(_)
        print(_,file=data)

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
        print('cln auc:{}'.format(auc_score))
        print('cln auc:{}'.format(auc_score),file=data)

print(simFunc)
print(simFunc,file=data)
print(defense)
print(defense,file=data)
print(dataset)
print(dataset,file=data)
time_end=time.time()  
print(time_end-time_start)
print(time_end-time_start,file=data) 

data.close()        