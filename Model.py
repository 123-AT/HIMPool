import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from Pool import SAMGIPool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv,GATConv,GINConv,GraphSAGE


# 使用三层的MLP作为分类器
class Classifier(nn.Module):
    # 图分类的分类器
    def __init__(self,in_channels,hidden_channels,out_channels,dropout=0.7):
        super(Classifier, self).__init__()
        self.lin1=nn.Linear(in_channels,hidden_channels)
        self.lin2=nn.Linear(hidden_channels,hidden_channels//2)
        self.lin3=nn.Linear(hidden_channels//2,out_channels)

        self.act=nn.ReLU()
        self.dropout=dropout

    def forward(self,x):
        x=self.act(self.lin1(x))
        x=F.dropout(x,p=self.dropout,training=self.training)

        x=self.act(self.lin2(x))
        x=F.dropout(x,p=self.dropout,training=self.training)

        output=F.log_softmax(self.lin3(x),dim=-1)
        return output


class SAMGINet(nn.Module):
    def __init__(self,args):
        super(SAMGINet, self).__init__()
        # 模型参数
        self.in_channels=args.num_features
        self.hidden_channels=args.hidden_channels
        self.out_channels=args.num_classes
        self.dropout=args.dropout
        self.negative_slop=args.negative_slop
        # 池化参数
        self.ratios=args.ratios
        self.depth=len(self.ratios)
        self.sample_times=args.sample_times
        # 中心节点
        self.center_hop=args.center_hop
        self.sl_hop=args.struct_learn_hop
        # 方法选择
        self.mi_loss_type=args.mi_loss_type
        self.local_pool_method=args.local_pool_method
        # 结构学习参数
        self.struct_improve=args.struct_improve
        self.lamb=args.lamb
        self.fast_sl=args.fast_struct_learning

        self.Conv_Name=args.Conv_Name
        if self.Conv_Name == 'GCN':
            self.Conv_layer = GCNConv
        elif self.Conv_Name == 'GAT':
            self.Conv_layer = GATConv
        elif self.Conv_Name == 'GraphSAGE':
            self.Conv_layer = GraphSAGE
        elif self.Conv_Name == 'GIN':
            self.Conv_Name = GINConv
        else:
            raise ValueError(f"Unsupported convolution layer: {self.Conv_Name}")

        self.Encoder=nn.ModuleList()
        if self.Conv_Name=='GIN':
            self.Encoder.append(self.Conv_layer(nn.Linear(self.in_channels,self.hidden_channels)))
        else:
            self.Encoder.append(self.Conv_layer(self.in_channels,self.hidden_channels))

        self.Pools=nn.ModuleList()
        for i in range(self.depth):
            self.Pools.append(
                SAMGIPool(self.hidden_channels,self.ratios[i],self.sample_times,
                         self.mi_loss_type,self.local_pool_method,self.struct_improve,
                         self.lamb,self.fast_sl,self.sl_hop,self.negative_slop,self.center_hop)
            )
            self.Encoder.append(self.Conv_layer(self.hidden_channels,self.hidden_channels))
        self.classifier=Classifier(in_channels=self.hidden_channels*2,hidden_channels=self.hidden_channels,out_channels=self.out_channels,dropout=self.dropout)

    def forward(self,data):
        edge_index,x,batch,edge_attr=data.edge_index,data.x,data.batch,data.edge_attr
        x=self.Encoder[0](x,edge_index)
        x=F.relu(x)

        cur_embedding=F.relu(torch.concat([global_mean_pool(x,batch),global_max_pool(x,batch)],dim=1))
        graph_embedding=torch.zeros_like(cur_embedding,device=cur_embedding.device,dtype=cur_embedding.dtype)

        local_loss_list=[]
        global_loss_list=[]

        for pool,conv in zip(self.Pools,self.Encoder[1:]):
            x,edge_index,edge_attr,batch,perm,loc_loss,glo_loss=pool(cur_embedding,x,edge_index,edge_attr,batch)

            cur_embedding=F.relu(torch.concat([global_mean_pool(x,batch),global_max_pool(x,batch)],dim=1))
            graph_embedding = graph_embedding + cur_embedding

            x=conv(x,edge_index,edge_attr)
            x=F.relu(x)

            local_loss_list.append(loc_loss)
            global_loss_list.append(glo_loss)

        loss1=torch.stack(local_loss_list,dim=0).mean()
        loss2=torch.stack(global_loss_list,dim=0).mean()
        out=self.classifier(graph_embedding)
        return out,loss1,loss2













