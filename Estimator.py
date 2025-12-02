import torch
import torch.nn as nn
import torch.nn.functional as F

class Bilinear_Estimator(nn.Module):
    def __init__(self,hidden_channels1,hidden_channels2, negative_slope=0.2):
        super(Bilinear_Estimator, self).__init__()
        self.dis=nn.Bilinear(in1_features=hidden_channels1,in2_features=hidden_channels2,out_features=1)
        self.act=nn.Sigmoid()
        for module in self.modules():
            self.init_weights(module)

    def init_weights(self,module):
        if isinstance(module,nn.Bilinear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self,x,y):
        scores=self.act(self.dis(x,y))
        return scores

class MI_Estimator(nn.Module):
    def __init__(self,in_channels,hidden_channels,negative_slope=0.2):
        super(MI_Estimator, self).__init__()
        self.estimator=nn.Sequential(
            nn.Linear(in_features=in_channels,out_features=hidden_channels),
            nn.LeakyReLU(negative_slope=negative_slope,inplace=True),

            nn.Linear(in_features=hidden_channels,out_features=hidden_channels//2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Linear(in_features=hidden_channels//2,out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x,y):
        """
        input:征服样本对
        :return: 分数 (nx1)
        """
        embedding=torch.concat([x,y],dim=1)
        scores=self.estimator(embedding)
        return scores
#tensor([4760, 4871], device='cuda:0')

if __name__ == '__main__':
    x=torch.rand((10,128))
    y=torch.rand((10,128))
    estimator=MI_Estimator(128,128)
    scores=estimator(x,y)
    print(scores.shape)
    print(scores)