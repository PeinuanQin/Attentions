# @Time : 2022/1/27 19:17 
# @Author : PeinuanQin
# @File : run.py
import sys
sys.path.append("../../")

from utils import train
import torch
from SENet.model.SEResNet import SEResNet
import random
from torchvision import transforms
import numpy as np
from torchvision.datasets import CIFAR10
from torch.optim import lr_scheduler

def trainSENet(depth
                   ,model_name
                   ,save_model_path
                   , resume=None
                   , checkpoint_file_path=None
                   , classes=10):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SEResNet(depth,classes).to(device)

    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_set = CIFAR10(root='../../datasets', train=True, download=False, transform=transform_train)
    test_set = CIFAR10(root='../../datasets', train=False, download=False, transform=transform_test)


    print(sum(param.numel() for param in model.parameters()))
    print(*[param.numel() for param in model.parameters()])
    optimizer = torch.optim.SGD(model.parameters()
                                , lr=0.1
                                , momentum=0.9,
                                weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer
                                         , milestones=[50, 100, 150]
                                         , gamma=0.1
                                         , last_epoch=-1)
    train(model
          , model_name
          , train_set
          , test_set
          , optimizer
          , scheduler
          , device
          , save_model_path
          , 200
          , 128
          , resume
          , checkpoint_file_path)


if __name__ == '__main__':

    trainSENet(20
               ,"resnet20_SE"
               , "./trained_model/resnet20_SE"
               , None
               , None
               ,10)
