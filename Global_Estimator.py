import torch
import torch.nn as nn
from Estimator import MI_Estimator
from MI_Loss import MIObjective
from utils import Generate_Sub_Emb
from torch_geometric.nn import global_mean_pool,global_max_pool
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
class Global_MI(nn.Module):
    def __init__(self,in_channels,hidden_channels,sample_times,mi_loss_type,center_hop,negative_slop=0.2):
        super(Global_MI, self).__init__()
        self.estimator=MI_Estimator(in_channels,hidden_channels,negative_slope=negative_slop)
        self.sample_times=sample_times
        self.generate_sub_emb=Generate_Sub_Emb(center_hop=center_hop)
        # 定义互信息损失类别
        self.loss=MIObjective(mi_loss_type)
        self.act=nn.PReLU()
    def forward(self,x,edge_index,batch):
        num_nodes=x.size(0)
        new_edge_index,_=add_self_loops(edge_index,num_nodes=num_nodes)

        pos_sub_embs=F.relu(self.generate_sub_emb(x,new_edge_index))
        graph_embeddings=F.relu(torch.concat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
        pred_xy=self.estimator(pos_sub_embs,graph_embeddings[batch])

        pred_x_y_list=[]
        for i in range(self.sample_times):
            idx=torch.randperm(num_nodes)
            neg_sub_embs=self.act(self.generate_sub_emb(x[idx],new_edge_index))
            pred_x_y_i=self.estimator(neg_sub_embs,graph_embeddings[batch])
            pred_x_y_list.append(pred_x_y_i)

        pred_x_y=torch.squeeze(torch.stack(pred_x_y_list,dim=1),dim=-1)
        loss=self.loss(pred_xy,pred_x_y)

        return pred_xy,loss
# 子结构的采样需要改变一下   打乱特征，节点和图结构不变，然后进行互信息估计