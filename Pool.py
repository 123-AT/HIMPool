import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch_geometric.utils import add_remaining_self_loops, dense_to_sparse

from sparse_softmax import Sparsemax
from Local_Estimator import Adaptive_Local_MI,Average_Local_MI
from Global_Estimator import Global_MI
from utils import Generate_Sub_Emb,topk,compute_indices
from Neighbor_Augment import TwoHopNeighborhood



class SAMGIPool(nn.Module):
    def __init__(self,hidden_channels,ratio,sample_times,mi_loss_type,l_pool_method,struct_improve,lamb,fast_SL,sl_hop,negative_slop,center_hop):
        super(SAMGIPool, self).__init__()
        self.hidden_channels=hidden_channels
        self.ratio=ratio
        self.sample_times = sample_times
        self.mi_loss_type = mi_loss_type

        self.center_hop = center_hop

        self.negative_slop=negative_slop

        # init 局部互信息估计方法
        self.l_pool_method=l_pool_method.lower()
        if self.l_pool_method=='adaptive':
            self.local_mi_estimator=Adaptive_Local_MI(in_channels=hidden_channels*2,hidden_channels=hidden_channels,sample_times=sample_times,
                                                      mi_loss_type=mi_loss_type,negative_slop=negative_slop,center_hop=center_hop)
        elif self.l_pool_method=='average':
            self.local_mi_estimator=Average_Local_MI(in_channels=hidden_channels*2,hidden_channels=hidden_channels,sample_times=sample_times,
                                                     mi_loss_type=mi_loss_type,negative_slop=negative_slop)
        else:
            raise ValueError(f'Unknown Local Pool Method {self.l_pool_method}')

        self.generate_sub_emb=Generate_Sub_Emb(center_hop=center_hop)
        self.global_mi_estimator=Global_MI(in_channels=hidden_channels*3,hidden_channels=hidden_channels,sample_times=sample_times,
                                           mi_loss_type=mi_loss_type,center_hop=center_hop,negative_slop=negative_slop)

        # 结构学习
        self.struct_improve=struct_improve
        self.lamb=lamb

        self.fast_sl=fast_SL
        self.sl_hop=sl_hop
        self.neighbor_augment=TwoHopNeighborhood()

        self.sparse_attention=Sparsemax()

        self.struct_att=nn.Parameter(torch.Tensor(1,hidden_channels*2))
        nn.init.xavier_uniform_(self.struct_att.data)

        self.view_att=nn.Parameter(torch.Tensor(2,2))
        nn.init.xavier_uniform_(self.view_att.data)
        self.view_bias=nn.Parameter(torch.Tensor(2))
        nn.init.zeros_(self.view_bias.data)


    def forward(self,graph_embedding,x,edge_index,edge_attr,batch=None):
        if batch==None:
            batch=edge_index.new_zeros(x.size(0))
        num_node = x.size(0)
        sub_emb=self.generate_sub_emb(x,edge_index,edge_attr)
        if self.l_pool_method=='adaptive':
            scores1,loc_loss=self.local_mi_estimator(x,edge_index,batch)
        else:
            scores1,loc_loss=self.local_mi_estimator(sub_emb,x,batch)
        scores2,glo_loss=self.global_mi_estimator(x,edge_index,batch)

        scores_cat=torch.cat([scores1,scores2],dim=1)
        score_weights=torch.sigmoid(torch.matmul(scores_cat,self.view_att)+self.view_bias)
        score_weights=torch.softmax(score_weights,dim=1)
        scores=torch.sigmoid(torch.sum(scores_cat * score_weights, dim=1))

        perm=topk(scores,self.ratio,batch)
        original_x=x
        x=x[perm]*scores[perm].view(-1,1)
        batch=batch[perm]
        #induced_edge_index,induced_edge_attr=filter_adj(edge_index,edge_attr,perm,num_nodes=num_node)

        if self.struct_improve is not True:
            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            hop_data = self.neighbor_augment(hop_data)
            i_edge_index=hop_data.edge_index
            i_edge_attr=hop_data.edge_attr
            if i_edge_attr==None:
                i_edge_attr=torch.ones((i_edge_index.size(1),),dtype=torch.float32,device=i_edge_index.device)
            induced_edge_index, induced_edge_attr = filter_adj(i_edge_index, i_edge_attr, perm, num_nodes=num_node)
            return x, induced_edge_index, induced_edge_attr, batch, perm, loc_loss, glo_loss

        if self.fast_sl is True:
            if edge_attr==None:
                edge_attr=torch.ones((edge_index.size(1),),dtype=torch.float32,device=edge_index.device)

            # 邻域增强
            hop_data=Data(x=original_x,edge_index=edge_index,edge_attr=edge_attr)
            for _ in range(self.sl_hop-1):
                hop_data=self.neighbor_augment(hop_data)
            hop_edge_index=hop_data.edge_index
            hop_edge_attr=hop_data.edge_attr
            new_edge_index,new_edge_attr=filter_adj(hop_edge_index,hop_edge_attr,perm,num_nodes=num_node)
            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))

            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.struct_att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb

            adj = torch.zeros((num_node, num_node), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            new_edge_attr = self.sparse_attention(weights, row)
            adj[row, col] = new_edge_attr
            new_edge_index,new_edge_attr=dense_to_sparse(adj)
            del adj
            torch.cuda.empty_cache()
        else:
            if edge_attr==None:
                induced_edge_attr=torch.ones((induced_edge_index.size(1),),dtype=torch.float32,device=edge_index.device)
            num_nodes, shift_cum_num_nodes, cum_num_nodes=compute_indices(batch)
            adj=torch.zeros((x.size(0),x.size(0)),dtype=torch.float32,device=x.device)
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0

            new_edge_index,_=dense_to_sparse(adj)
            row,col=new_edge_index
            weights=(torch.cat([x[row],x[col]],dim=1)*self.struct_att).sum(dim=-1)
            weights=F.leaky_relu(weights,self.negative_slop)
            adj[row,col]=weights

            induced_row,induced_col=induced_edge_index
            adj[induced_row,induced_col]+=induced_edge_attr*self.lamb

            weights=adj[row,col]
            new_edge_attr=self.sparse_attention(weights,row)
            adj[row,col]=new_edge_attr
            new_edge_index,new_edge_attr=dense_to_sparse(adj)

            del adj
            torch.cuda.empty_cache()
        return x, new_edge_index, new_edge_attr, batch, perm, loc_loss, glo_loss















