import time
import argparse
import numpy as np
import torch
from ProGNN.prognn import ProGNN
from ProGNN.gcn import GCN
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
'''
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
'''
class pro_GNN_Hyper:
    def __init__(self):
        self.debug=False, 
        self.lr=0.01,
        self.seed=15,
        self.weight_decay=5e-4,
        self.hidden=16,
        self.dropout=0.5,
        self.dataset='cora',
        self.ptb_rate=0.05,
        self.epochs=400, 
        self.alpha=5e-4, 
        self.beta=1.5, 
        self.gamma=1, 
        self.lambda_=0, 
        self.phi=0, 
        self.inner_steps=2, 
        self.outer_steps=1, 
        self.lr_adj=0.01, 
        self.symmetric='store_true'
        
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
    
def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    labels = torch.LongTensor(np.array(labels))
    features = torch.FloatTensor(np.array(features))
    adj = torch.FloatTensor(adj.todense())

    return adj, features, labels
        
def proGNNTrain(adj,u,v,emb0_u,emb0_v):
    args = pro_GNN_Hyper()
    args.cuda = False
    device = torch.device("cuda" if args.cuda else "cpu")
    
    labels = np.zeros((u+v))
    labels[u:] = 1

    val_size = 0.1
    test_size = 0.8
    train_size = 1 - test_size - val_size
    
    idx = np.arange(adj.shape[0])
    idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
    idx_unlabeled = np.union1d(idx_val, idx_test)

    avg=np.average(emb0_u)
    s=np.std(emb0_u)
    emb0_u=(emb0_u-avg)/s
    emb0_u-=2.326
    emb0_u[emb0_u>0]=(int)(1)
    emb0_u[emb0_u<=0]=(int)(0)    
    avg=np.average(emb0_v)
    s=np.std(emb0_v)
    emb0_v=(emb0_v-avg)/s
    emb0_v-=2.326
    emb0_v[emb0_v>0]=(int)(1)
    emb0_v[emb0_v<=0]=(int)(0)
    features = np.row_stack((emb0_u,emb0_v))
    
    args.debug=False
    args.lr=0.01
    args.weight_decay=5e-4
    args.hidden=16
    args.dropout=0.5
    args.dataset='cora'
    args.ptb_rate=0
    args.epochs=20 #20收敛
    args.alpha=5e-4
    args.beta=1.5 
    args.gamma=1
    args.lambda_=0
    args.phi=0
    args.inner_steps=0 #训练模型设置为2，否则置为0
    args.outer_steps=1
    args.lr_adj=0.01
    args.symmetric='store_true'
    args.seed=15
    
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    perturbed_adj = adj

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''
    print(features.shape[1])
    print(args.hidden)
    print(labels.max().item() + 1)
    print(args.dropout)
    '''
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=(int)(labels.max().item()) + 1,
                dropout=args.dropout, device=device)

    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False)
    prognn = ProGNN(model, args, device)
    '''
    print(features[:10])
    print(perturbed_adj[:10])
    print(labels[:10])
    print(idx_train[:10])
    print(idx_val[:10])
    '''
    return prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)
    

