import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models import *
from dataset import *
from torch.autograd import Variable



def main():

    # ======================== step 1/5 构建Dataset ==============================
    DATA_DIR = os.path.join(os.getcwd(), "..", "data", "mix_data.csv")

    # 构建Dataset实例
    train_data = DataSetV2(DATA_DIR, mode="train")
    valid_data = DataSetV2(DATA_DIR, mode="valid")

    # 构建DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, num_workers=16, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=512, num_workers=16, drop_last=False)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    # ======================== step 2/5 构建模型 ==============================
    model = CDNet(embedding_index=[0, 1, 2, 3, 4],
                  embedding_size=[92106, 93, 101587, 2000, 16727], dense_feature_num=802, cross_layer_num=2,
                  deep_layer=[256, 128, 32])
    model = model.to(cpu_device)

    # ======================== step 3/5 构建损失函数 ==============================
    criterion = nn.BCELoss()

    # ======================== step 4/5 构建优化器 ==============================
    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)

    # ======================== step 5/5 训练并且验证模型 ==============================
    loss_rec = {"train": [], "valid": []}
    iter_num = 0

    epoch_num = 100
    for epoch in range(epoch_num):
        print('starit epoch [{}/{}]'.format(epoch + 1, 5))
        model.train()
        loss_sigmal = []
        for sparse_feature, dense_feature, label in train_loader:
            # sparse_feature, dense_feature, label = sparse_feature.to(gpu_device), dense_feature.to(gpu_device), label.to(gpu_device)
            iter_num += 1
            pctr = model(sparse_feature, dense_feature)
            pctr = pctr.to(torch.float32)
            label = label.to(torch.float32)
            loss = criterion(pctr, label)
            iter_loss = loss.item()
            loss_sigmal.append(iter_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch {}/{}, total_iter is {}, train loss is {:.2f}".format(epoch + 1, epoch_num, iter_num, iter_loss))

        train_loss = np.mean(loss_sigmal)
        loss_rec['train'].append(train_loss)

        model.eval()
        loss_sigmal2 = []
    
        for sparse_feature, dense_feature, label in valid_loader:
            iter_num += 1
            # sparse_feature, dense_feature, label = sparse_feature.to(gpu_device), dense_feature.to(gpu_device), label.to(gpu_device)
            pctr = model(sparse_feature, dense_feature)
            pctr = pctr.to(torch.float32)
            label = label.to(torch.float32)
            loss = criterion(pctr, label)
            iter_loss = loss.item()
            loss_sigmal2.append(iter_loss)
            print("epoch {}/{}, total_iter is {}, valid loss is {:.2f}".format(epoch + 1, epoch_num, iter_num, iter_loss))
        
        valid_loss = np.mean(loss_sigmal2)
        loss_rec['valid'].append(valid_loss)


if __name__ == '__main__':
    main()
