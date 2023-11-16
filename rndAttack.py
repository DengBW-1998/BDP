import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
from ProGNN.train import pro_GNN_Hyper,proGNNTrain
from EdgeAttention.edge_attention import *

import warnings
warnings.filterwarnings("ignore")

#the meaning of each hyperparamaters intruduced in BBAI.py
n_flips = -1 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
rate=-1
n_node_pairs = 100000 #Number of test edges
iteration= 1 #Iteration rounds
threshold = 5 #Implicit relationship threshold
batch_size=64
train_model = 'netmf'

ptb_rate=20
dataset='dblp'
defense='bdp'
simFunc='euc'
dp=False
data=open("rnd20_euc.txt",'w+')
thj=0.05

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

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
adj = standardize(adj)
adj_nn = standardize(adj_nn)

emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
seed=0
np.random.seed(seed)
time_start=time.time()

candidates = generate_candidates_addition(adj_matrix=adj, n_candidates=n_candidates,u=u,v=v,seed=seed)
rnd_flips = random.sample(list(candidates.copy()), n_flips)
flips = np.array(rnd_flips)

adj_matrix_flipped = flip_candidates(adj, flips)
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

### Link Prediction
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
    
    if dp:
        embedding=edges_train(adj_matrix_flipped[:u,u:],u,v,embedding)
            
    auc_score = evaluate_embedding_link_prediction(
        adj_matrix=adj_matrix_flipped, 
        node_pairs=node_pairs, 
        embedding_matrix=embedding
    )
    print('rnd auc:{}'.format(auc_score))
    print('rnd auc:{}'.format(auc_score),file=data)
    
    adj=adj_matrix_flipped.copy()
    adj=standardize(adj)

time_end=time.time()
print(simFunc)
print(simFunc,file=data)
print(defense)
print(defense,file=data)
print(dataset)
print(dataset,file=data)
print(time_end-time_start)
print(time_end-time_start,file=data)  

data.close()        