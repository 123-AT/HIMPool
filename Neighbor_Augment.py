import torch
from torch_sparse import coalesce,spspmm


class NHopNeighborhood(object):
    def __init__(self, num_hops):
        self.num_hops = num_hops

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        original_value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)
        original_edge_index = edge_index

        # 初始化第1跳邻域
        current_edge_index = original_edge_index
        current_value = original_value

        # 扩展到 num_hops 跳邻域
        for _ in range(self.num_hops - 1):
            current_edge_index, current_value = spspmm(current_edge_index, current_value, original_edge_index, original_value, n, n, n, True)
            edge_index = torch.cat([edge_index, current_edge_index], dim=1)

            if edge_attr is not None:
                expanded_value = current_value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
                expanded_value = expanded_value.expand(-1, *list(edge_attr.size())[1:])
                edge_attr = torch.cat([edge_attr, expanded_value], dim=0)

        # 合并边集
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min')
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(num_hops={})'.format(self.__class__.__name__, self.num_hops)

class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min')
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
