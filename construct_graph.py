import torch
import numpy as np
import gnn
import glob
from config import *
import os

def to_Graph(properties):
    x_h = torch.zeros(nfibers, n_h).float().cuda()
    x_g = torch.tensor(properties).float().cuda()

    # all classes are observable by all fibers now
    num_h = x_h.size(0)
    num_g = x_g.size(0)
    e_h = []
    e_g = []
    for i in range(num_h):
        for j in range(num_g):
            e_h.append(i)
            e_g.append(j)
    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor([e_h,e_g],dtype=torch.long)

    u=torch.tensor([np.zeros(n_u)]).float()
    data = gnn.BipartiteData(edge_index.cuda(),x_h.cuda(),x_g.cuda(),edge_attr.cuda(),u.cuda())
    return data

# This function varies from problem to problem. 
# The following example only works for the PFS problem
# def to_Graph(indices,properties):
#     # properties: galaxy properties useful for g nodes
#     # indices: pre-calculated connectivity
#     properties = np.array(properties)
#     edge_attr = []
#     e_h = [] # start from h nodes
#     e_g = [] # end at g nodes

#     # Graph Connectivity Related to the Problem
#     for i,index in enumerate(indices):
#         for j in range(len(index)):
#             if index[j]<2394: 
#                 e_h.append(index[j])
#                 e_g.append(k)
#                 edge_attr.append(np.zeros(n_x)) # Edge initialization

#     edge_attr = torch.tensor(edge_attr).float()
#     edge_index = torch.tensor([e_h,e_g],dtype=torch.long)

#     x_h = torch.zeros(2394,n_h).float()
#     x_g = torch.tensor(properties[reachable]).float()
#     u=torch.tensor([np.zeros(n_u)]).float()
#     data = gnn.BipartiteData(edge_index.cuda(),x_h.cuda(),x_g.cuda(),edge_attr.cuda(),u.cuda())
#     return data

if __name__ == '__main__':
    names = glob.glob('pairs-%s/pair-*'%case) # pre-calculated connectivity
    utils = np.loadtxt('initial_info/utils-%s.txt'%case) # pre-calculated galaxy properties

    if not os.path.exists('graphs-%s/'%case):
        os.system('mkdir graphs-%s'%case)
    for k in range(len(names)):
        indices = np.loadtxt(names[k],dtype=int)
        # graph = to_Graph(indices,utils)
        graph = to_Graph(utils)
        torch.save(graph,names[k].replace('pairs-%s/pair'%case,'graphs-%s/graph'%case).replace('txt','pt'))
