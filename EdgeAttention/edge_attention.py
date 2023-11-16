import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgeTrain(nn.Module):
    def __init__(self,embedding0,embedding1):
        super(EdgeTrain, self).__init__()
        self.embedding0=nn.Parameter(embedding0)
        self.embedding1=nn.Parameter(embedding1)
    
    def forward(self):
        res=F.sigmoid(torch.dot(self.embedding0,self.embedding1))
        return res
        
def edge_train(embedding,node_pairs,cnt,label):
    for i in range(cnt):
        u=node_pairs[i,0]
        v=node_pairs[i,1]
        a=torch.tensor(embedding[u])
        b=torch.tensor(embedding[v]) 
        
        net=EdgeTrain(a,b)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        if(label==0):
            label=torch.dot(torch.tensor([0.]),torch.tensor([0.])).double()
        else:
            label=torch.dot(torch.tensor([1.]),torch.tensor([1.])).double()
        for i in range(20):
            out=net().double()
            loss=F.binary_cross_entropy(out,label)
            optimizer.zero_grad()  
            loss.backward()         
            optimizer.step()
        
    return embedding

def edges_train(Bu,u,v,embedding):
    Bu=np.array(Bu.todense())
    #训练边条数
    train_cnt=int(sum(sum(Bu))*0.8)
    #print(sum(sum(Bu)))
    #负采样数
    neg_cnt=train_cnt//2
    
    train_pairs=[]
    k=0
    for i in range(u):
        for j in range(v):
            if(Bu[i,j]==1):
                train_pairs.append([i,u+j])
                k+=1
                if(k>=train_cnt):
                    break
        if(k>=train_cnt):
            break  
    train_pairs=np.array(train_pairs)   
    #print(train_pairs.shape)
    embedding=edge_train(embedding,train_pairs,train_cnt,1)
    
    u_node_pairs = np.random.randint(0, u-1, [neg_cnt, 1])
    v_node_pairs = np.random.randint(u, u+v-2, [neg_cnt, 1])
    neg_pairs = np.column_stack((u_node_pairs,v_node_pairs)) 
    #print(neg_pairs.shape)
    embedding=edge_train(embedding,neg_pairs,neg_cnt,0)
    return embedding
    