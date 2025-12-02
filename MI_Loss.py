import torch
import torch.nn as nn
import torch.nn.functional as F


class MIObjective(nn.Module):
    def __init__(self,mi_loss_type):
        super(MIObjective, self).__init__()
        self.mi_loss_type=mi_loss_type.lower()

    def DV_loss(self,T_xy,T_x_y_marg):
        loss=T_xy.mean()-torch.log(torch.exp(T_x_y_marg).mean())
        return -loss

    def NWJ_loss(self,T_xy,T_x_y_marg):
        loss=T_xy.mean()-torch.exp(T_x_y_marg-1).mean()
        return -loss

    def JSD_loss(self,T_xy,T_x_y_marg):
        pos_scores=F.softplus(-T_xy)
        neg_scores=torch.mean(F.softplus(T_x_y_marg),dim=-1,keepdim=True)
        loss=torch.mean(pos_scores+neg_scores)
        return loss

    def GAN_loss(self,T_xy,T_x_y_marg):
        # pos_scores=torch.log(T_xy)
        # neg_scores=torch.mean(torch.log(1-T_x_y_marg),dim=-1,keepdim=True)
        # loss=-torch.mean(pos_scores+neg_scores)
        labels_xy = torch.ones_like(T_xy)
        labels_x_y_marg = torch.zeros_like(T_x_y_marg)
        bce_loss = nn.BCELoss(reduction='mean')

        joint_loss = bce_loss(T_xy, labels_xy)
        marginal_loss = bce_loss(T_x_y_marg, labels_x_y_marg)
        loss = (joint_loss + marginal_loss)/2.0
        return loss
    def forward(self,T_xy,T_x_y_marg):
        if self.mi_loss_type=='nwj':
            return self.NWJ_loss(T_xy,T_x_y_marg)
        elif self.mi_loss_type=='dv':
            return self.DV_loss(T_xy,T_x_y_marg)
        elif self.mi_loss_type=='jsd':
            return self.JSD_loss(T_xy,T_x_y_marg)
        elif self.mi_loss_type=='gan':
            return self.GAN_loss(T_xy,T_x_y_marg)
        else:
            raise ValueError(f'Unknown method{self.method}')


"""
在图结构中衡量节点与其邻域的相关性，以选择关键节点时，使用 GAN-like 互信息估计方法。
理由如下：
敏感性高：GAN-like 互信息估计对输入数据的相关性变化更为敏感。这种敏感性在图结构中有助于更准确地捕捉节点与其邻域之间的相关性，从而更好地识别关键节点。
判别器机制：GAN-like 方法使用判别器来区分不同分布的样本，能够更有效地学习和捕捉复杂的关系模式。在图结构中，节点及其邻域可能具有复杂的相关性模式，GAN-like 方法的判别器机制能够更好地捕捉这些模式。
更高的区分能力：由于 GAN-like 方法在捕捉分布差异方面具有更高的区分能力，它能够更好地区分关键节点和非关键节点，从而提升关键节点选择的准确性。
"""
if __name__ == '__main__':
    # 当变量越相关，loss应该越大  因为此时表明estimator完全无法区分征服样本
    x=torch.rand((1,128))
    y=torch.rand((10,128))
    mi_loss_type_list=['nwj','dv','jsd','gan']
    for mi_type in mi_loss_type_list:
        mi=MIObjective(mi_type)
        loss=mi(x,y)
        print(f"MI type: {mi_type}, Loss: {loss.item()}")

    for mi_type in mi_loss_type_list:
        mi=MIObjective(mi_type)
        loss=mi(x,x)
        print(f"MI type: {mi_type}, Loss: {loss.item()}")




