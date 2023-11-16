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

n_flips = -1 #number of perturbations
dim = 64 #embedding vector dimension
window_size = 5 #window size
n_node_pairs = 100000 #number of test edges
iteration= 2 #iteration rounds
threshold = 5 #implicit relationship threshold
batch_size=64
rate=-1 #dataset retention rate
train_model = 'netmf' #training model

ptb_rate=20 #perturbation rate,5 or 20 is valid
dataset='dblp'
defense='bdp' #adversarial defense algorithm
dp=False #the tag of supergat(dp)
simFunc='euc' #similarity function
data=open("BBAI20_euc.txt",'w+')
thj=0.05  #0.001 0.002 0.01 0.02 0.05 0.1 0.2 0.5
thx=0 #3.090 ,2.878 ,2.576 ,2.054 ,1.645 ,1.282 ,0.842 ,0.000

if dataset=='dblp':
    n_flips=3600//iteration
    rate=1
    thx=2.054
if dataset=='wiki':
    n_flips=3800//iteration
    rate=2
    thx=2.576

if ptb_rate==5:
    n_flips=n_flips//4
n_candidates=10*n_flips #number of candidate perturbed edges

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
adj = standardize(adj)
adj_nn = standardize(adj_nn)

emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
seed=0
np.random.seed(seed)
time_start=time.time()

for ite in range(iteration):

    candidates_nn = generate_candidates_addition(adj_matrix=adj_nn, n_candidates=n_candidates,u=u,v=v,seed=seed)
    if ite==0:
        flips,vals_est, vecs_org = perturbation_top_flips(adj_matrix=adj_nn, candidates=candidates_nn, n_flips=n_flips, dim=dim, window_size=window_size)
    else:
        flips,vals_est, vecs_org = increment_perturbation_top_flips(
        adj_matrix=adj_nn,
        candidates=candidates_nn,
        n_flips=n_flips,
        dim=dim,
        window_size=window_size,
        vals_org=vals_org,
        vecs_org=vecs_org,
        flips_org=flips
        )
    adj_matrix_flipped = flip_candidates(adj_nn, flips)
    if ite==iteration-1:
        if defense=='proGNN' or defense=='prognn':
            adj_matrix_flipped=proGNNTrain(adj_matrix_flipped,u,v,emb0_u,emb0_v)
            #print(type(adj_matrix_flipped)) #torch
            adj_matrix_flipped=adj_matrix_flipped.detach().numpy()
            adj_matrix_flipped[adj_matrix_flipped>=0.5]=1
            adj_matrix_flipped[adj_matrix_flipped<0.5]=0
            adj_matrix_flipped=sp.csr_matrix(adj_matrix_flipped)

        if defense=='bdp':
            adj_matrix_flipped,emb0_uu,emb0_vv = GCNSimDefense(emb0_u, emb0_v, u, v, adj_matrix_flipped[:u, u:], thx, thj,simFunc)
            adj_matrix_flipped[:u,:u] = adj_nn.todense()[:u,:u]
            adj_matrix_flipped[u:,u:] = adj_nn.todense()[u:,u:]
            adj_matrix_flipped = ImpAddEdges(emb0_uu, emb0_vv, u, v,adj_matrix_flipped,thj,simFunc)
            adj_matrix_flipped[:u, :u] = 0
            adj_matrix_flipped[u:, u:] = 0
            adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

        if defense == 'stable' or defense == 'STABLE':
            adj_matrix_flipped = stable(np.row_stack((emb0_u, emb0_v)), adj_matrix_flipped)

    ### Link Prediction
    for _ in range(5):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))

        adj_matrix_flipped[:u,:u]=0
        adj_matrix_flipped[u:,u:]=0

        if ite==iteration-1 :
            print(ite)
            print(ite,file=data)

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
            print('UBAI auc:{}'.format(auc_score))
            print('UBAI auc:{}'.format(auc_score),file=data)

        adj_nn=adjsp_2_adjnn(adj_matrix_flipped,u,v,threshold)
        adj_nn=standardize(adj_nn)

    vals_org=vals_est.copy()

time_end=time.time()
'''
print(thj)
print(thx)
print(thj,file=data)
print(thx,file=data)
'''
print(simFunc)
print(simFunc,file=data)
print(defense)
print(defense,file=data)
print(dataset)
print(dataset,file=data)
print(time_end-time_start)
print(time_end-time_start,file=data)
#print(threshold) 
data.close()        