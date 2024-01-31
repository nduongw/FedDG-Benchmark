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

from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

dir_photo = '../data/pacs_v1.0/photo/'
dir_art = '../data/pacs_v1.0/art_painting/'
dir_cartoon = '../data/pacs_v1.0/cartoon/'
dir_sketch = '../data/pacs_v1.0/sketch/'

photo_dataset = ImageFolder(dir_photo, transform=photo_transform)
art_dataset = ImageFolder(dir_art, transform=art_transform)
cartoon_dataset = ImageFolder(dir_cartoon, transform=cartoon_transform)
sketch_dataset = ImageFolder(dir_sketch, transform=sketch_transform)

def train(args, model, train_loader, test_loader, criterion, optimizer):
    model.to(device)
    stored_label = []
    max_accuracy = 0.0

    for epoch in range(args.num_epoch):
        if args.method == 'conststyle':
            for conststyle in model.conststyle:
                conststyle.clear_memory()

        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels, domains = inputs[0].to(device), inputs[1].to(device), labels
            optimizer.zero_grad()

            stored_label.extend(labels.detach().cpu())
            if args.method == 'conststyle':
                if epoch == 0:
                    outputs = model(inputs, domains, store_style=True)
                else:
                    outputs = model(inputs, domains, const_style=True, store_style=True)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if args.method == 'conststyle':
            if epoch % 10 == 0:
                for conststyle in model.conststyle:
                    conststyle.cal_mean_std(2)

        print(f"Epoch {epoch+1}/{args.num_epoch}, Train Loss: {running_loss/len(train_loader)}")

        model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                if args.method == 'conststyle':
                    outputs = model(inputs, domains, const_style=True, test=True)

                outputs = model(inputs)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        # Calculate test accuracy
        test_accuracy = correct_predictions / total_samples
        if test_accuracy >= max_accuracy:
            max_accuracy = test_accuracy

        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    print(f"Training finished | Max Accuracy: {max_accuracy}")
    if not os.path.exists(f'{args.method}_{args.train_domains}_{args.test_domains}'):
        os.makedirs(f'{args.method}_{args.train_domains}_{args.test_domains}')
    with open(f'{args.method}_{args.train_domains}_{args.test_domains}/acc.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Acc', max_accuracy])

def main(args):
    if os.path.exists(f'{args.method}_{args.train_domains}_{args.test_domains}/acc.csv'):
        os.remove(f'{args.method}_{args.train_domains}_{args.test_domains}/acc.csv')

    train_dataset = []
    len_dataset = []
    idx = 1
    if 'p' in args.train_domains:
        train_dataset.append(photo_dataset)
        len_dataset.append(torch.full((len(photo_dataset), 1), idx))
        idx += 1
    if 'a' in args.train_domains:
        train_dataset.append(art_dataset)
        len_dataset.append(torch.full((len(art_dataset), 1), idx))
        idx += 1
    if 'c' in args.train_domains:
        train_dataset.append(cartoon_dataset)
        len_dataset.append(torch.full((len(cartoon_dataset), 1), idx))
        idx += 1
    if 's' in args.train_domains:
        train_dataset.append(sketch_dataset)
        len_dataset.append(torch.full((len(sketch_dataset), 1), idx))
        idx += 1
    concated_train_dataset = ConcatDataset(train_dataset)
    concated_train_domain = torch.vstack(len_dataset)
    train_loader = DataLoader(list(zip(concated_train_dataset, concated_train_domain)), batch_size=32, shuffle=True, num_workers=8)

    if 'p' in args.train_domains:
        test_loader = DataLoader(photo_dataset, batch_size=32, shuffle=False, num_workers=8)
    elif 'a' in args.train_domains:
        test_loader = DataLoader(art_dataset, batch_size=32, shuffle=False, num_workers=8)
    elif 'c' in args.train_domains:
        test_loader = DataLoader(cartoon_dataset, batch_size=32, shuffle=False, num_workers=8)
    elif 's' in args.train_domains:
        test_loader = DataLoader(sketch_dataset, batch_size=32, shuffle=False, num_workers=8)

    if args.method == 'conststyle':
        model = ConstStyleModel()
    elif args.method == 'mixstyle':
        model = MixStyleModel()
    elif args.method == 'dsu':
        model = DSUModel(uncertainty=0.5)
    elif args.method == 'csu':
        model = CSUModel(['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], 0.5, 0.1)

    model.model.fc = torch.nn.Linear(model.model.fc.in_features, 7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=1e-4, weight_decay=1e-5)
    train(args, model, train_loader, test_loader, criterion, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_domains', type=str)
    parser.add_argument('--test_domains', type=str)
    parser.add_argument('--method', type=str, choices=['csu', 'dsu', 'mixstyle', 'conststyle'])
    parser.add_argument('--num_epoch', type=int, default=50)

    args = parser.parse_args()
    main(args)
