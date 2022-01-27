# @Time : 2022/1/27 19:19 
# @Author : PeinuanQin
# @File : utils.py
import torch.utils.data as Data
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c
import cv2
import seaborn as sns


def train(model
          ,model_name
          ,train_set
          ,testset
          ,optimizer
          ,scheduler
          ,device
          ,save_path = None
          ,EPOCH = 200
          ,batch_size=128
          ,resume=None
          ,checkpoint_file_path=None):
    print("saving path is: %s" % save_path)
    writer = SummaryWriter(save_path)
    print("training model: %s" % model_name)
    model_name = model_name
    loss_func = nn.CrossEntropyLoss()
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    best_acc = 0.7

    if resume:
        assert checkpoint_file_path is not None, "please ensure input the checkpoint file"
        model.load_state_dict(torch.load(checkpoint_file_path))
    else:
        resume = 0
    for epoch in range(resume,EPOCH):
        model.train()
        epoch_loss = 0
        avg_loss = 0
        for step,(x,y) in tqdm(enumerate(train_loader)):
            input_x = x.to(device)
            input_y = y.to(device)
            output = model(input_x)
            loss = loss_func(output,input_y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = epoch_loss / (step+1)
        scheduler.step()
        accuracy = test(model,testset,device,128)
        print('[epoch:%d] Loss: %.4f    |    Acc: %.2f%%' % (epoch + 1, avg_loss, 100. * accuracy))
        writer.add_scalar("val_accuracy", 100. * accuracy, epoch + 1)
        writer.add_scalar("avg_loss", avg_loss, epoch + 1)
        if accuracy > best_acc:
            best_acc = accuracy
            filename = '%s.%03d.pt' % (model_name,epoch)
            print("saving models ... to %s " % os.path.join(save_path,filename))
            torch.save(model.state_dict(), os.path.join(save_path,filename))

    print("model: %s ,best acc: %.2f%% " % (model_name, 100. * best_acc))

def test(model,test_set,device,BATCH_SIZE = 128):
    model.eval()
    dataloader = Data.DataLoader(test_set, batch_size=BATCH_SIZE)
    correct = 0
    with torch.no_grad():
        for step,(x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    accuracy = correct / len(test_set)
    return accuracy

