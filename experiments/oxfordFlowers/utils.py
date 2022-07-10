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
    #custom_transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    #objectCatgsPath="./101_ObjectCategories"
    #image_paths = list(paths.list_images(objectCatgsPath))
    #data = []
    #labels = []
    #for img_path in tqdm(image_paths):
    #    label = img_path.split(os.path.sep)[-2]
    #    if label == "BACKGROUND_Google":
    #        continue
    #    img = cv2.imread(img_path)
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                     
    #    data.append(img)
    #    labels.append(label)                                            
    #data = np.array(data)
    #labels = np.array(labels)
    #lb = preprocessing.LabelEncoder()
    #labels = lb.fit_transform(labels)
    #print(f"Total Number of Classes: {len(lb.classes_)}")
    #set_CALTECH101=CustomDataset(data,labels,custom_transform)
    #same split as "How well do sparse ImageNet Models transfer"
    #trainset_caltech101,testset_caltech101=torch.utils.data.random_split(set_CALTECH101, [3030, 5647])
    #trainset,testset = torch.utils.data.random_split(set_CALTECH101,[3030,5647])
    
    trainset = torchvision.datasets.Flowers102(
        root='./data', split='train', download=True, transform=transform_train)
    valset = torchvision.datasets.Flowers102(
        root='./data', split='val', download=True, transform=transform_train)
    trainset= torch.utils.data.ConcatData([trainset,valset])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.Flowers102(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
