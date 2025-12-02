import os
import argparse
from Trainer import Trainer
from utils import setup_seed
import numpy as np
#torch.autograd.set_detect_anomaly(True)MUTAG
# 生物分子数据集  MUTAG NCI1和NCI109 Mutagenicity
# 蛋白质数据集    DD PROTEINS ENZYMES
# 社交网络数据集   COLLAB  IMDB-B 和 IMDB-M
parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=356,help='random seed')
parser.add_argument('--dataset',type=str,default='COLLAB',help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device',type=str,default='cuda:0',help='cuda device')
parser.add_argument('--max_epochs',type=int,default=1000,help='maximum number of epoch')
parser.add_argument('--patience',type=int,default=50,help='patience for early stopping')

# 损失参数
parser.add_argument('--alpha',type=float,default=0.1,help='局部互信息权重参数')
parser.add_argument('--beta',type=float,default=0.1,help='全局互信息权重参数')

# 互信息参数
parser.add_argument('--center_hop',type=int,default=1,help='局部互信息估计节点邻域数')
parser.add_argument('--local_pool_method',type=str,default='average',help='局部互信息估计方法  average 或者  adaptive')
parser.add_argument('--sample_times',type=int,default=1,help='负采样次数')
parser.add_argument('--mi_loss_type',type=str,default='GAN',help='互信息优化目标函数JSD/NWJ/DV/GAN')

# 训练相关参数
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
parser.add_argument('--negative_slop',type=float,default=0.001,help='dropout rate')
parser.add_argument('--batch_size',type=int,default=512,help='batch size')
parser.add_argument('--hidden_channels',type=int,default=256,help='hidden layer dims')
parser.add_argument('--ratios', type=float, nargs='+', default=[0.8,0.8,0.6],help='Pooling ratios.')

# 结构学习参数
parser.add_argument('--struct_improve',type=bool,default=False,help='是否进行图结构学习对粗图进行结构增强')
parser.add_argument('--fast_struct_learning',type=bool,default=True,help='是否进行快速图结构学习')
parser.add_argument('--lamb',type=float,default=1.5,help='图结构残差连接')
parser.add_argument('--struct_learn_hop',type=int,default=2,help='快速结构学习邻域增强阶数')
parser.add_argument('--Conv_Name',type=str,default='GCN',help='GCN/GAT/GraphSAGE/GIN')

if __name__ == '__main__':
    acc_list=[]
    for i in range(10):
        print('Times for Train: {:04d}'.format(i))
        args = parser.parse_args()
        #args.seed=356
        #setup_seed(args.seed)
        print(args)
        trainer=Trainer(args)
        acc=trainer.train()
        acc_list.append(acc)
        print()
    acc_array = np.array(acc_list)*100


    mean = np.mean(acc_array)
    std = np.std(acc_array)

    print("Result:")
    print(acc_array)
    print('mean: {:.6f}'.format(mean),'std: {:.6f}'.format(std))

    result_dir = os.path.join('Result', args.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存实验结果到文件
    result_file = os.path.join(result_dir, f'results_{args.seed}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("\nArguments:\n")
        f.write(str(args))
        f.write("\n\nIndividual Accuracies (%):\n")
        for i, acc in enumerate(acc_array):
            f.write(f"Run {i}: {acc:.6f}\n")
        f.write("\nSummary:\n")
        f.write(f"Mean Accuracy: {mean:.6f}%\n")
        f.write(f"Std Deviation: {std:.6f}%\n")
    
    print(f"Results saved to {result_file}")