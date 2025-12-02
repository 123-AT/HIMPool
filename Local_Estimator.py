import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, add_remaining_self_loops
from torch_geometric.utils import structured_negative_sampling_feasible
from torch_geometric.utils import structured_negative_sampling, batched_negative_sampling,negative_sampling
from torch_sparse import coalesce, spspmm
from torch_scatter import scatter_add
from Estimator import MI_Estimator
from MI_Loss import MIObjective
from Neighbor_Augment import TwoHopNeighborhood, NHopNeighborhood
# 全部都统一一下方便操作
class Average_Local_MI(nn.Module):
    def __init__(self, in_channels, hidden_channels, sample_times, mi_loss_type, negative_slop=0.2):
        super(Average_Local_MI, self).__init__()
        self.estimator = MI_Estimator(in_channels, hidden_channels, negative_slope=negative_slop)
        self.sample_times = sample_times
        self.loss = MIObjective(mi_loss_type)

    def forward(self, sub_emb, x, batch):
        # sub_emb的center_hop和全局互信息估计一致
        num_nodes = x.shape[0]
        pred_xy = self.estimator(x, sub_emb)

        pred_x_y_list = []
        for i in range(self.sample_times):
            idx = torch.randperm(num_nodes)
            pred_x_y = self.estimator(x, sub_emb[idx])
            pred_x_y_list.append(pred_x_y)

        pred_x_y = torch.squeeze(torch.stack(pred_x_y_list, dim=1), dim=-1)
        loss = self.loss(pred_xy, pred_x_y)
        return pred_xy, loss

class Adaptive_Att(nn.Module):
    def __init__(self, hidden_channels):
        super(Adaptive_Att, self).__init__()
        self.hidden_channels = hidden_channels
        self.att_weight = nn.Parameter(torch.Tensor(1, self.hidden_channels * 2))
        nn.init.xavier_uniform_(self.att_weight.data)

    def forward(self, edge_index, x):
        row, col = edge_index
        mi_att = torch.matmul(torch.cat([x[row], x[col]], dim=1), self.att_weight.transpose(0, 1))
        mi_att = torch.sigmoid(mi_att)
        return mi_att


class Adaptive_Local_MI(nn.Module):
    def __init__(self, in_channels, hidden_channels, sample_times, mi_loss_type, negative_slop=0.2, center_hop=1):
        super(Adaptive_Local_MI, self).__init__()
        self.estimator = MI_Estimator(in_channels, hidden_channels, negative_slope=negative_slop)

        self.sample_times = sample_times
        self.loss = MIObjective(mi_loss_type)
        self.adap_attr = Adaptive_Att(hidden_channels)

        self.center_hop = center_hop
        self.neighbor_augment = TwoHopNeighborhood()

    def forward(self, x, edge_index, batch=None):
        # 自环的问题
        if self.center_hop >= 2:
            hop_data = Data(x=x, edge_index=edge_index, edge_attr=None)
            hop_data = self.neighbor_augment(hop_data)
            edge_index = hop_data.edge_index
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        pred_xy = self.estimator(x[row], x[col])
        att_mi = self.adap_attr(edge_index, x)

        pred_x_y_list = []
        for i in range(self.sample_times):
            # t=time.time()
            # negative_row,negative_col=batched_negative_sampling(edge_index,batch)
            # print(time.time()-t,negative_row.size(),row.size())
            # pred_x_y=self.estimator(x[negative_row], x[negative_col])
            # pred_x_y_list.append(pred_x_y)
            # pass
            if structured_negative_sampling_feasible(edge_index):
                row, _, negative_col = structured_negative_sampling(edge_index, num_nodes=x.size(0))
                pred_x_y = self.estimator(x[row], x[negative_col])
                pred_x_y_list.append(pred_x_y)
            # else:
            #     #打乱目标节点
            # perm = torch.randperm(col.size(0))
            # negative_col = col[perm]
            # pred_x_y = self.estimator(x[row], x[negative_col])
            # pred_x_y_list.append(pred_x_y)


        pred_x_y = torch.squeeze(torch.stack(pred_x_y_list, dim=1), dim=-1)
        loss = self.loss(pred_xy, pred_x_y)
        # pos_scores=torch.log(pred_xy)
        # # neg_scores=torch.mean(torch.log(1-pred_x_y),dim=-1,keepdim=True)
        # # loss=-torch.mean(pos_scores+neg_scores)
        # # row,_=edge_index
        scores = torch.sigmoid(scatter_add(pred_xy * att_mi, row, dim=0))

        return scores, loss

# 互信息估计得到的值为0
