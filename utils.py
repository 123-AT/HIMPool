import os
import torch
import psutil
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from torch import Tensor
from torch_scatter import scatter_add
from torch_sparse import coalesce,spspmm
from torch_geometric.utils import remove_self_loops, spmm, cumsum, scatter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from Neighbor_Augment import TwoHopNeighborhood,NHopNeighborhood

def get_memory_info():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # 转换为MB

def get_gpu_memory_info():
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为MB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # 转换为MB
        return gpu_mem_allocated, gpu_mem_reserved
    else:
        return None, None

class Generate_Sub_Emb(nn.Module):
    def __init__(self,center_hop):
        super(Generate_Sub_Emb, self).__init__()
        self.center_hop=center_hop
        self.neighbor_augment=TwoHopNeighborhood()

    def forward(self,x,edge_index,edge_attr=None):
        num_nodes=x.size(0)

        hop_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if self.center_hop>=2:
            hop_data=self.neighbor_augment(hop_data)
        edge_index=hop_data.edge_index
        edge_attr=hop_data.edge_attr

        if edge_attr==None:
            edge_attr=torch.ones((edge_index.size(1),),dtype=torch.float,device=edge_index.device)
        row,col=edge_index[0],edge_index[1]
        deg=scatter(edge_attr,row,dim=0,dim_size=num_nodes,reduce='sum')
        deg_inv_sqrt=deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        norm = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

        adj=SparseTensor(row=row,col=col,value=norm,sparse_sizes=(num_nodes,num_nodes))
        sub_embedding=spmm(adj,x)

        return sub_embedding

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

def compute_indices(batch):
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
    shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    cum_num_nodes = num_nodes.cumsum(dim=0)
    return num_nodes, shift_cum_num_nodes, cum_num_nodes

def generate_mask(batch,new_batch):
    mask=torch.zeros((batch.size(0),new_batch.size(0)),dtype=torch.float,device=batch.device)

    num_nodes, shift_cum_num_nodes, cum_num_nodes = compute_indices(batch)
    drop_num_nodes, drop_shift_cum_num_nodes, drop_cum_num_nodes = compute_indices(new_batch)
    #mask=torch.zeros((num_nodes,drop_num_nodes),dtype=torch.float,device=batch.device)
    for (idx_i,idx_j),(drop_idx_i,drop_idx_j) in zip(zip(shift_cum_num_nodes, cum_num_nodes),
                                                     zip(drop_shift_cum_num_nodes, drop_cum_num_nodes)
                                                     ):
        mask[idx_i:idx_j,drop_idx_i:drop_idx_j]=1.0
    return mask

def filter_adj_matirx(A,idx):
    """
    只保留选择节点和未选择节点之间的邻接关系，进而聚合信息
    :param A: 分配矩阵
    :param idx: 选择节点下标
    :return:
    """
    mask=torch.ones_like(A,dtype=torch.bool)
    mask[idx,:]=0
    return A*mask.float()

def Gumbel_Softmax(logits,tau,hard=False):
    """
    hard方式使用直通估计器（Straight-Through Estimator）:
    这种方法在前向传播时输出离散化的结果（硬样本），但在反向传播时，将离散化
    结果视为连续的概率分布（软样本），以便梯度可以正常传播。
    """
    gumbels=-torch.empty_like(logits).exponential_().log()
    gumbels=(logits+gumbels)/tau
    y_soft=gumbels.softmax(dim=-1)
    if hard:
        index=y_soft.max(dim=-1,keepdim=True)[1]
        y_hard=torch.zeros_like(logits).scatter_(-1,index,1.0)
        # y_soft.detach()将y_soft从计算图中分离，使其梯度在反向传播中不会被计算
        ret=y_hard-y_soft.detach()+y_soft
    else:
        ret=y_soft
    return ret


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
