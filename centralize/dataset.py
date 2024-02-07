import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv
import wandb

from torch.utils.data import Subset, DataLoader, ConcatDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transf = transforms.Compose([
                            transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                            transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                            transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])

def photo_transform(data):
    transf_data = transf(data)
    transf_data.domain_id = 1
    return transf_data

def art_transform(data):
    transf_data = transf(data)
    transf_data.domain_id = 2
    return transf_data

def cartoon_transform(data):
    transf_data = transf(data)
    transf_data.domain_id = 3
    return transf_data

def sketch_transform(data):
    transf_data = transf(data)
    transf_data.domain_id = 4
    return transf_data

def prepare_pacs(args):
    dir_photo = '../data/pacs_v1.0/photo/'
    dir_art = '../data/pacs_v1.0/art_painting/'
    dir_cartoon = '../data/pacs_v1.0/cartoon/'
    dir_sketch = '../data/pacs_v1.0/sketch/'

    photo_dataset = ImageFolder(dir_photo, transform=photo_transform)
    art_dataset = ImageFolder(dir_art, transform=art_transform)
    cartoon_dataset = ImageFolder(dir_cartoon, transform=cartoon_transform)
    sketch_dataset = ImageFolder(dir_sketch, transform=sketch_transform)
    
    return [photo_dataset, art_dataset, cartoon_dataset, sketch_dataset]

def prepare_officehome(args):
    dir_art = '../data/OfficeHomeDataset_10072016/Art'
    dir_clipart = '../data/OfficeHomeDataset_10072016/Clipart'
    dir_product = '../data/OfficeHomeDataset_10072016/Product'
    dir_realworld = '../data/OfficeHomeDataset_10072016/RealWorld'
    
    art_dataset = ImageFolder(dir_art, transform=photo_transform)
    clipart_dataset = ImageFolder(dir_clipart, transform=art_transform)
    product_dataset = ImageFolder(dir_product, transform=cartoon_transform)
    realworld_dataset = ImageFolder(dir_realworld, transform=sketch_transform)
    
    return [art_dataset, clipart_dataset, product_dataset, realworld_dataset]