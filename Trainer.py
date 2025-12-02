import time
import os
import numpy as np
import torch
import math
import torch.nn as nn
from Model import Classifier,SAMGINet
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
#import argparse
#torch.autograd.set_detect_anomaly(True)
#Multi-Granularity InfoMax Pooling (MGIP)
def get_weight(epoch, initial_weight, decay_rate):
    return initial_weight * math.exp(-decay_rate * epoch)

class Trainer(nn.Module):
    def __init__(self,args):
        super(Trainer, self).__init__()
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        # 加载数据集
        if args.dataset == 'IMDB-MULTI' or args.dataset == 'IMDB-BINARY' or args.dataset == 'COLLAB':
            self.dataset = TUDataset('data/', name=args.dataset)
            max_degree = 0
            for g in self.dataset:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            self.dataset.transform = OneHotDegree(max_degree)
            args.num_classes = self.dataset.num_classes
            args.num_features = self.dataset.num_features
        else:
            self.dataset = TUDataset('data/', name=args.dataset, use_node_attr=True)
            args.num_classes = self.dataset.num_classes
            args.num_features = self.dataset.num_features

        self.args=args
        self.in_channels=self.args.num_features
        self.hidden_channels=self.args.hidden_channels
        self.out_channels=self.args.num_classes

        self.device=self.args.device
        self.max_epoch=args.max_epochs
        self.patience=self.args.patience

        # 设置训练集/验证集/测试集
        self.num_training=int(len(self.dataset)*0.8)
        self.num_val=int(len(self.dataset)*0.1)
        self.num_test=len(self.dataset)-self.num_training-self.num_val

        training_set,val_set,test_set=random_split(self.dataset,[self.num_training,self.num_val,self.num_test])

        self.train_loader=DataLoader(training_set,batch_size=args.batch_size,shuffle=True)
        self.val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False)
        self.test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

        # 这个位置语法有没有问题  验证一下是否最后保留的模型是最优的
        self.model=None
        self.optimizer=None
        self.criterion=nn.NLLLoss()

        self.path=os.path.join('model_param',self.args.dataset)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.best_model= f"best_model_{self.args.dataset}_{self.args.local_pool_method}.pth"

    def set_model(self):
        self.model=SAMGINet(self.args).to(device=self.device)

    def save_model(self,model_name):
        path=os.path.join(self.path,model_name)
        torch.save(self.model.state_dict(),path)

    def load_model(self,model_name):
        path=os.path.join(self.path,model_name)
        self.set_model()
        self.model.load_state_dict(torch.load(path))

    def set_optim(self):
        self.optimizer=torch.optim.Adam(params=self.model.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)

    def train_one_epoch(self,data_loader):
        self.model.train()
        total_loss=[]
        correct=0
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()
            data=data.to(self.device)
            out,local_loss,global_loss=self.model(data)
            cls_loss=self.criterion(out,data.y)

            loss=1.2*cls_loss+ self.args.alpha*local_loss +self.args.beta*global_loss
            loss.backward()

            self.optimizer.step()
            pred=torch.argmax(out,dim=1)==data.y.reshape(-1)
            correct+=pred.sum().item()
            total_loss.append([
                loss.item(),
                cls_loss.item(),
                local_loss.item(),
                global_loss.item()])

        acc=correct/len(data_loader.dataset)
        total_loss=np.array(total_loss)
        avg_losses=total_loss.mean(axis=0)
        return avg_losses,acc,total_loss

    def train(self):
        self.set_model()
        self.set_optim()
        best_val_acc=0.0
        best_test_acc=0.0
        best_val_loss=1e6
        patience_count=0
        best_epoch=0
        t = time.time()
        for epoch in range(self.max_epoch):
            t_epoch=time.time()
            self.epoch=epoch+1
            train_avg_loss,train_acc,train_total_loss=self.train_one_epoch(self.train_loader)
            val_avg_loss,val_acc,val_total_loss=self.evaluate(self.val_loader)
            #test_avg_loss, test_acc, test_total_loss=self.evaluate(self.test_loader)

            if best_val_loss>val_avg_loss[1]:
                best_val_loss=val_avg_loss[1]
                best_val_acc=val_acc
                best_epoch=epoch+1
                patience_count=0
                self.save_model(self.best_model)
            else:
                patience_count+=1

            if patience_count>self.patience:
                print("Early Stopping")
                break
            print('Epoch: {:04d}'.format(epoch+1),'train loss: {:.6f}'.format(train_avg_loss[0]),
                  'cls loss: {:.6f}'.format(train_avg_loss[1]),'loc_loss: {:.6f}'.format(train_avg_loss[2]),
                  'glo_loss: {:.6f}'.format(train_avg_loss[3]),'train acc: {:.6f}'.format(train_acc),
                  'val loss: {:.6f}'.format(val_avg_loss[0]),'val cls loss: {:.6f}'.format(val_avg_loss[1]),
                  'val loc_loss: {:.6f}'.format(val_avg_loss[2]),'val glo_loss: {:.6f}'.format(val_avg_loss[3]),
                  'val acc: {:.6f}'.format(val_acc),'time: {:.6f}'.format(time.time()-t_epoch))

        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

        test_acc,test_loss=self.test()
        print(f'Test set results:best_epoch={best_epoch} Test Loss= {test_loss}, Test Acc= {test_acc}')

        val_avg_loss, val_acc, val_total_loss = self.evaluate(self.val_loader)
        print('val loss: {:.6f}'.format(val_avg_loss[0]), 'val cls loss: {:.6f}'.format(val_avg_loss[1]),
        'val loc_loss: {:.6f}'.format(val_avg_loss[2]), 'val glo_loss: {:.6f}'.format(val_avg_loss[3]),
        'val acc: {:.6f}'.format(val_acc), 'time: {:.6f}'.format(time.time() - t_epoch))
        return test_acc

    def evaluate(self,data_loader):
        self.model.eval()
        total_loss=[]
        correct = 0
        for data in data_loader:
            data=data.to(self.device)
            with torch.no_grad():
                out,l_loss,g_loss=self.model(data)
                cls_loss=self.criterion(out,data.y)
                loss=cls_loss + self.args.alpha*l_loss + self.args.beta*g_loss
                pred=torch.argmax(out,dim=1)==data.y.reshape(-1)
                correct+=pred.sum().item()

            total_loss.append([
                loss.item(),
                cls_loss.item(),
                l_loss.item(),
                g_loss.item()])
        acc=correct/len(data_loader.dataset)
        total_loss=np.array(total_loss)
        avg_losses=total_loss.mean(axis=0)
        return avg_losses,acc,total_loss

    # 模型名字测试一下结果
    def test(self):
        self.load_model(self.best_model)
        test_losses,acc,_=self.evaluate(self.test_loader)
        return acc,test_losses

