#/usr/bin/env python

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from model import Net
from dataset import get_dataloader
import numpy as np
import os.path as osp
import os
from matplotlib import pyplot as plt

def evaluate(model, data_loader):
    mse=0
    mae=0
    output_all=[]
    gt_all=[]
    for i,(feature, target) in enumerate(data_loader):
        target = Variable(target.type(torch.FloatTensor))
        feature = Variable(feature.type(torch.FloatTensor))
        
        output = model(feature)
        
        target = target.data.numpy()
        output = output.data.numpy()
        
        error = target.T - output.T

        mse += np.sum(np.square(error))
        mae += np.sum(np.fabs(error))
        gt_all += target.tolist()
        output_all += output.tolist()

    mse = mse/len(gt_all)
    mae = mae/len(gt_all)
    print("mae: %.4f; mse: %.4f"%(mae, mse))

    return mae, mse, gt_all, output_all

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MG-Prediction')
    return parser

def main():
    args=parse_args()
    args.model_path = "./models/best.pth"
    args.batch_size = 256
    args.data_path = "./data/data.npy"

    checkpoint = torch.load(args.model_path)
    args.input_dim = checkpoint["input_dim"]
    args.hidden_nums = checkpoint["hidden_nums"]
    model = Net(args.input_dim, args.hidden_nums, 1)
    model.load_state_dict(checkpoint['state_dict'])

    data = np.load(args.data_path)
    data_loader = get_dataloader(data, train=False, batch_size=args.batch_size, input_dim=args.input_dim)
    mae, mse, gt_all, output_all = evaluate(model, data_loader)
    
    data_range = data_loader.dataset.data_range
    x = [i for i in range(data_range[0], data_range[1])]


    plt.figure()
    plt.title("MG-eval")
    plt.plot(x, gt_all, color='coral',label = "gt")
    plt.plot(x, output_all, color='green', label = "predict")
    plt.xlabel("time", fontsize=13)
    plt.ylabel("value", fontsize=13)
    plt.legend(loc='best')#显示在最好的位置
    plt.show()

if __name__ == "__main__":
    main()


        