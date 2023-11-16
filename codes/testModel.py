import sys
import numpy as np
from sklearn import preprocessing
from deeprobust.graph.data import Dataset
from codes.data_utils import DataUtils
from codes.graph_utils import GraphUtils
import random
import math
import os
import scipy.sparse as sp

class BineModel(object):
    def __init__(self,dataset):   
        if dataset=='wiki':
            self.train_data=r'data/wiki/rating_train.dat'
            self.test_data=r'data/wiki/rating_test.dat'
            self.model_name='wiki'
            self.vectors_u=r'data/wiki/vectors_u.dat'
            self.vectors_v=r'data/wiki/vectors_v.dat'
            self.case_train=r'data/wiki/case_train.dat'
            self.case_test=r'data/wiki/case_test.dat' 
            self.dataset='wiki'
        
        if dataset=='dblp':
            self.train_data=r'data/dblp/rating_train.dat'
            self.test_data=r'data/dblp/rating_test.dat'
            self.model_name='dblp'
            self.vectors_u=r'data/dblp/vectors_u.dat'
            self.vectors_v=r'data/dblp/vectors_v.dat'
            self.dataset='dblp'

def adj_2_adjnn(adj,u,v,th):
    n=u+v
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    adj_uv=np.array(adj_uv)
    adj_vu=np.array(adj_vu)

    adj_uv=sp.csr_matrix(adj_uv)
    adj_vu=sp.csr_matrix(adj_vu)
    adj=sp.csr_matrix(adj)
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
    
    
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv 
    return adj_nn

def adjsp_2_adjnn(adj,u,v,th):
    n=u+v
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
        
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv   
    return adj_nn

def getAdj(th,dataset,rate):
    
    if dataset== 'dblp' or dataset== 'wiki':
        bine=BineModel(dataset)

        model_path = os.path.join('./', bine.model_name)
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)

        dul = DataUtils(model_path)
        gul = GraphUtils(model_path)
        gul.construct_training_graph(bine.train_data)

        #(1/rate) represents the proportion of nodes in the selected subgraph
        gul.u_nodes = gul.u_nodes//rate
        gul.v_nodes = gul.v_nodes//rate
        gul.n_nodes = gul.n_nodes//rate
        u,v,n=gul.u_nodes,gul.v_nodes,gul.n_nodes
        adj=[[0]*n for _ in range(n)] 

        for i in range(len(gul.edge_list)):
            u_index = (int)(gul.edge_list[i][0][1:])
            v_index = (int)(gul.edge_list[i][1][1:])
            if u>u_index and v>v_index:
                adj[u_index][v_index+u]=1
                adj[v_index+u][u_index]=1
        adj=np.array(adj)
      
        adj_nn=adj_2_adjnn(adj,u,v,th)
        adj=sp.csr_matrix(adj)
        return adj_nn,adj,u,v

def GCNSimDefense(emb0_u, emb0_v, u, v, Bu, thx=2.326, thj=0.05,simFunc='jac'):
    #simFunc ['jac','cos','prs','ham']
    n=u+v
    adj=np.zeros((n,n))
    Bu=np.array(Bu.todense())

    if simFunc=='jac' or simFunc=='ham':
        avg=np.average(emb0_u)
        s=np.std(emb0_u)
        emb0_u=(emb0_u-avg)/s
        emb0_u-=thx
        emb0_u[emb0_u>0]=(int)(1)
        emb0_u[emb0_u<=0]=(int)(0)
        avg=np.average(emb0_v)
        s=np.std(emb0_v)
        emb0_v=(emb0_v-avg)/s
        emb0_v-=thx
        emb0_v[emb0_v>0]=(int)(1)
        emb0_v[emb0_v<=0]=(int)(0)
    
    for i in range(u):
        for j in range(v):
            edge=Bu[i,j]
            if(edge==1):
                xu=emb0_u[i,:]
                xv=emb0_v[j,:]
                p=sim(xu,xv,simFunc)
                if(p<thj):
                    Bu[i,j]=0
    
    #print(Bu)
    adj[:u,u:]=Bu
    adj[u:,:u]=Bu.T
    return adj,emb0_u,emb0_v

def cosSim(X,Y):
    if (X.shape[0]==Y.shape[0]):
        d1 = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    else:
        d1=0
    return d1

def prsSim(X, Y):
    if Y.shape[0]!=128:
        return 0
    n = len(X)
    sum1 = sum(X)
    sum2 = sum(Y)
    sum1_pow = sum([pow(v, 2.0) for v in X])
    sum2_pow = sum([pow(v, 2.0) for v in Y])
    p_sum = sum(np.multiply(X, Y))
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    if den == 0:
        return 0.0
    return num / den

def hamSim(s1,s2):
    if(s2.shape[0]!=128):
        res=0
    else:
        res=sum(s1_ == s2_ for s1_, s2_ in zip(s1, s2))/len(s1)
    return res

def jacSim(x1, x2):
    fenmu = np.sum(np.logical_or(x1, x2).astype(int))
    fenzi = np.sum(np.logical_and(x1, x2).astype(int))
    if fenmu == 0:
        p = 0
    else:
        p = fenzi / fenmu
    return p

def eucSim(x1,x2):
    dis = np.sqrt(sum(np.power((x1 - x2), 2)))
    return 1.0/(dis+1.0)

def sim(xu,xv,simFunc):
    if simFunc == 'jac':
        p = jacSim(xu, xv)
    if simFunc == 'cos':
        p = cosSim(xu, xv)
    if simFunc == 'ham':
        p = hamSim(xu, xv)
    if simFunc == 'prs':
        p = prsSim(xu, xv)
    if simFunc == 'euc':
        p = eucSim(xu, xv)
    return p

def ImpAddEdges(emb0_u, emb0_v, u, v,adj,thj,simFunc):

    for i in range(0,u):
        for j in range(0,u):
            if(adj[i,j]>0):
                xu = emb0_u[i, :]
                xv = emb0_u[j, :]
                p=sim(xu,xv,simFunc)
                for x in range(u,u+v):

                    if(adj[i,x]>0 and adj[j,x]==0):
                        if p>=thj:
                            x1=emb0_u[j,:]
                            x2=emb0_v[x-u:]
                            p2=sim(x1,x2,simFunc)
                            try:
                                if p2 >= thj:
                                    adj[i, x] = (int)(1)
                            except:
                                pass
                                
                    if (adj[j, x] > 0 and adj[i, x] == 0):
                        if p>=thj:
                            x1 = emb0_u[i, :]
                            x2 = emb0_v[x - u:]
                            p2=sim(x1,x2,simFunc)
                            try:
                                if p2 >= thj:
                                    adj[i, x] = (int)(1)
                            except:
                                pass

    for i in range(u,u+v):
        for j in range(u,u+v):
            if (adj[i, j] > 0):
                xu = emb0_v[i-u, :]
                xv = emb0_v[j-u, :]
                p=sim(xu,xv,simFunc)
                for x in range(u):
                    
                    if (adj[x,i] > 0 and adj[x,j] == 0):
                        if p>=thj:
                            x1=emb0_u[x,:]
                            x2=emb0_v[j-u:]
                            p2=sim(x1,x2,simFunc)
                            try:
                                if p2 >= thj:
                                    adj[i, x] = (int)(1)
                            except:
                                pass
                    if (adj[x,j] > 0 and adj[x,i] == 0):
                        if p>=thj:
                            x1 = emb0_u[x, :]
                            x2 = emb0_v[i - u:]
                            p2=sim(x1,x2,simFunc)
                            try:
                                if p2 >= thj:
                                    adj[i, x] = (int)(1)
                            except:
                                pass
    return adj