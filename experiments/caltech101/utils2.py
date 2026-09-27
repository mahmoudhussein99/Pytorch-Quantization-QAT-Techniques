import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import gc
import joblib
import cv2
import random
import pretrainedmodels
from imutils import paths
from tqdm import tqdm
from sklearn import preprocessing

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.labels = labels
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.images[index][:]
        if self.transforms:
            data = self.transforms(data)
        if self.labels is not None:
            return (data, self.labels[index])
        else:
            return data
def data_generator(args):

    # Data
    print('==> Preparing data..')
    #caltech101
    if(args.dataset=='caltech101'):
        custom_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5458, 0.5288, 0.5022],std=[0.3137, 0.3078, 0.3206]),
            ])

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
          #  transforms.RandomCrop(32, padding=4),
           # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5458, 0.5288, 0.5022],std=[0.3137, 0.3078, 0.3206]),
        ])

        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5458, 0.5288, 0.5022],std=[0.3137, 0.3078, 0.3206]),
        ])

        objectCatgsPath="./101_ObjectCategories"
        image_paths = list(paths.list_images(objectCatgsPath))

        data = []
        labels = []
        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)
        lb = preprocessing.LabelEncoder()
        labels = lb.fit_transform(labels)
        print(f"Total Number of Classes: {len(lb.classes_)}")
        set_CALTECH101=CustomDataset(data,labels,transform_train)
        #same split as "How well do sparse ImageNet Models transfer"
        trainset,testset = torch.utils.data.random_split(set_CALTECH101,[3030,5647])
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    
    elif(args.dataset=='food101'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5450, 0.4435, 0.3436],std=[0.2695, 0.2719, 0.2766]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5450, 0.4435, 0.3436],std=[0.2695, 0.2719, 0.2766]),
        ])
        trainset = torchvision.datasets.Food101(
           root='./data_food101', split='train', download=True, transform=transform_train)

        testset = torchvision.datasets.Food101(
           root='./data_food101', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader

    elif(args.dataset=='stanfordCars'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4709, 0.4603, 0.4550],std=[0.2891, 0.2881, 0.2967]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4709, 0.4603, 0.4550],std=[0.2891, 0.2881, 0.2967]),
        ])
        trainset = torchvision.datasets.StanfordCars(
           root='./data_stanford', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.StanfordCars(
           root='./data_stanford', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    
    elif(args.dataset=='dtd'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5274, 0.4703, 0.4235],std=[0.2618, 0.2506, 0.2593]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5274, 0.4703, 0.4235],std=[0.2618, 0.2506, 0.2593]),
        ])
        trainset = torchvision.datasets.DTD(
           root='./data_dtd', split='train', download=True, transform=transform_train)
        valset = torchvision.datasets.DTD(
           root='./data_dtd', split='val', download=True, transform=transform_train)
        trainset= torch.utils.data.ConcatDataset([trainset,valset])
        testset = torchvision.datasets.DTD(
           root='./data_dtd', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif(args.dataset=='cifar10'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
        ])
        trainset = torchvision.datasets.CIFAR10(
           root='./data_cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
           root='./data_cifar10', train=False, download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif(args.dataset=='cifar100'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761]),
        ])
        trainset = torchvision.datasets.CIFAR100(
           root='./data_cifar100', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
           root='./data_cifar100', train=False, download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif(args.dataset=='fgvc'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4799, 0.5111, 0.5343],std=[0.2171, 0.2103, 0.2426]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4799, 0.5111, 0.5343],std=[0.2171, 0.2103, 0.2426]),
        ])
        trainset = torchvision.datasets.FGVCAircraft(
           root='./data_fgvc', split='trainval', download=True, transform=transform_train)
        testset = torchvision.datasets.FGVCAircraft(
           root='./data_fgvc', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif(args.dataset=='oxfordFlowers'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4343, 0.3828, 0.2954],std=[0.2907, 0.2424, 0.2704]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4343, 0.3828, 0.2954],std=[0.2907, 0.2424, 0.2704]),
        ])
        trainset = torchvision.datasets.Flowers102(
           root='./data_oxfordFlowers', split='train', download=True, transform=transform_train)
        valset = torchvision.datasets.Flowers102(
           root='./data_oxfordFlowers', split='val', download=True, transform=transform_train)
        trainset= torch.utils.data.ConcatDataset([trainset,valset])
        testset = torchvision.datasets.Flowers102(
           root='./data_oxfordFlowers', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif(args.dataset=='oxfordPets'):

        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4784, 0.4461, 0.3959],std=[0.2628, 0.2573, 0.2654]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4784, 0.4461, 0.3959],std=[0.2628, 0.2573, 0.2654]),
        ])
        trainset = torchvision.datasets.Flowers102(
           root='./data_oxfordPets', split='trainval', download=True, transform=transform_train)
        testset = torchvision.datasets.Flowers102(
           root='./data_oxfordPets', split='test', download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader

