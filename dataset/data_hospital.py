'''以下是医院数据集的读取文件,如果要使用这份文件，需要将本文件名改为data.py'''
from __future__ import print_function
import random
import numpy as np
import torch
import os
def seed_torch(seed=42):
    random.seed(seed) # Python的随机性
    os.environ["PYTHONHASHSEED"] = str(seed) # 设置python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed) # numpy的随机性
    torch.manual_seed(seed) # torch的CPu随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed) # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPu.torch的GPu随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be Falsetorch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.deterministic = True # 选择确定性算法
seed=42
seed_torch(seed)

# import torch
import torchvision
import torchvision.transforms as transforms
# import numpy as np 
from torchvision import transforms, datasets
# import os
crop_size = 32
padding = 4

def prepare_cifar100_train_dataset(data_dir, dataset='cifar100', batch_size=16, 
                                    shuffle=True, num_workers=4):

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(crop_size, padding),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
    #     #                      std=[0.2673, 0.2564, 0.2762]),
    #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                          std=[0.2023, 0.1994, 0.2010]),
    # ])
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = datasets.ImageFolder("./hospital_data/train",
                                         transform=data_transform["train"])
#     train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, 
#                                                  download=True, transform=train_transform)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers)
    return train_loader

def prepare_cifar100_test_dataset(data_dir, dataset='cifar100', batch_size=16, 
                                    shuffle=False, num_workers=4):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
    #         #                      std=[0.2673, 0.2564, 0.2762]),
    #         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                              std=[0.2023, 0.1994, 0.2010]),
    #     ])

#     testset = torchvision.datasets.CIFAR100(root=data_dir,
#                                                train=False,
#                                                download=True,
#                                                transform=transform_test) 
    testset = datasets.ImageFolder("./hospital_data/test",
                                            transform=data_transform["val"])
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)  
    return test_loader
