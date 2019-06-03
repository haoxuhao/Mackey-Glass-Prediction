#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from model import Net
from dataset import get_dataloader
import numpy as np
import os.path as osp
import os

def train(epoch, model, opt, criterion, data, args):
    train_loader = get_dataloader(data, train=True, batch_size=args.batch_size, input_dim=args.input_dim)
    total_loss = 0
    model.train()
    for i,(feature, target) in enumerate(train_loader):
        opt.zero_grad()
        target = Variable(target.type(torch.FloatTensor))
        feature = Variable(feature.type(torch.FloatTensor))

        output = model(feature)
        
        loss = criterion(output.squeeze(1), target)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if i%args.print_freq == 0:
            #print(output)
            print("[%d/%d] curr loss: %4f; total loss: %4f"%(i, epoch, loss.item(), total_loss))
    print("epoch: %d; total train loss: %4f"%(epoch, total_loss))

def val(epoch, model, criterion, data, args):
    val_loader = get_dataloader(data, train=False, batch_size=args.batch_size, input_dim=args.input_dim)
    total_loss = 0
    for i,(feature, target) in enumerate(val_loader):
        target = Variable(target.type(torch.FloatTensor))
        feature = Variable(feature.type(torch.FloatTensor))
        
        output = model(feature)
        loss = criterion(output.squeeze(1), target)
        total_loss += loss.item()
        if i%args.print_freq == 0:
            print("[%d/%d] total val loss: %4f"%(i, len(val_loader), total_loss))
        
    print("epoch: %d; total val loss: %4f"%(epoch, total_loss))
    return total_loss

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MG-Prediction')
    return parser
def main():
    args=parse_args()
    args.lr = 0.001
    args.max_epochs = 60
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.print_freq = 100
    args.start_epoch = 0
    args.data_path = "./data/data.npy"
    args.model_save_dir = "./models"
    args.batch_size = 16
    args.input_dim = 5
    min_loss = 1e6

    if not osp.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    data = np.load(args.data_path)
    model = Net(args.input_dim, [64,16], 1)
    opt = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    criterion = nn.MSELoss(size_average=False)
    
    for epoch in range(args.start_epoch, args.max_epochs):
        train(epoch, model, opt, criterion, data, args)
        val_loss = val(epoch, model, criterion, data, args)
        if val_loss < min_loss:
            min_loss = val_loss
            #save model
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': min_loss,
            'optimizer' : opt.state_dict(),
            }
            torch.save(state, osp.join(args.model_save_dir, "best.pth"))
            print("save best model; best loss: %4f"%min_loss)


if __name__ =="__main__":
    main()






