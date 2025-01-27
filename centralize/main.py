import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import numpy as np
import argparse
import os
import csv
import wandb
import tqdm

from torch.utils.data import Subset, DataLoader, ConcatDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataset import *
from style import *
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--train_domains', type=str)
parser.add_argument('--test_domains', type=str)
parser.add_argument('--method', type=str, choices=['csu', 'dsu', 'mixstyle', 'conststyle', 'baseline', 'conststyle-bn'])
parser.add_argument('--dataset', type=str, choices=['pacs', 'officehome'])
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--style_idx', type=int, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--update_interval', type=int, default=10)
parser.add_argument('--option', type=str, default='')
args = parser.parse_args()

gr_name = 'PACS' if args.dataset == 'pacs' else 'OfficeHome'
if args.wandb:
    job_type = f'{args.method}_{args.option}'
    tracker = wandb.init(
        project = 'CentralizedDG',
        entity = 'aiotlab',
        config = args,
        group = f'{gr_name}',
        name = f'train={args.train_domains}_test={args.test_domains}'+
            f'_method={args.method}'+
            f'_style={args.style_idx}',
        job_type = job_type
    )
    args.tracker = tracker

set_seed(args.seed)
    

def train(args, model, train_loader, test_in_domain_loader, test_out_domain_loader, criterion, optimizer):
    model.to(device)
    stored_label = []
    max_accuracy = 0.0

    for epoch in tqdm.tqdm(range(args.num_epoch)):
        if args.method == 'conststyle' or args.method == 'conststyle-bn':
            for conststyle in model.conststyle:
                conststyle.clear_memory()
        elif args.method == 'mixstyle':
            model.mixstyle.clear_memory()
        elif args.method == 'dsu':
            model.pertubration1.clear_memory()
            model.pertubration2.clear_memory()
        elif args.method == 'csu':
            model.pertubration1.clear_memory()
            model.pertubration2.clear_memory()

        model.train()
        running_loss = 0.0

        print(f'Training w epoch {epoch+1}')
        for inputs, domains_val in tqdm.tqdm(train_loader):
            inputs, labels, domains = inputs[0].to(device), inputs[1].to(device), domains_val
            optimizer.zero_grad()

            stored_label.extend(labels.detach().cpu().numpy())
            if args.method == 'conststyle' or args.method == 'conststyle-bn':
                if epoch == 0:
                    outputs = model(inputs, domains, store_feats=True)
                else:
                    outputs = model(inputs, domains, const_style=True, store_feats=True)
            else:
                outputs = model(inputs, domains=domains, store_feats=True)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            #training more using sampling features
            # if args.method == 'conststyle' or args.method == 'conststyle-bn':
            #     if epoch > 0:
            #         optimizer.zero_grad()
            #         outputs = model(inputs, domains, const_style=True, store_style=True, sampling=True)
            #         loss = criterion(outputs, labels)
            #         loss.backward()
            #         optimizer.step()

            # running_loss += loss.item()

        if args.method == 'conststyle' or args.method == 'conststyle-bn':
            if epoch % args.update_interval == 0:
                print('Update cluster')
                for idx, conststyle in enumerate(model.conststyle):
                        conststyle.cal_mean_std(idx, args, epoch)

        print(f"Epoch {epoch+1}/{args.num_epoch}, Train Loss: {running_loss/len(train_loader)}")

        model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            print(f'Test in-domain data')
            count = 0
            for inputs, labels in test_in_domain_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                if args.method == 'conststyle' or args.method == 'conststyle-bn':
                    if epoch == 0:
                        outputs = model(inputs, domains)
                    else:
                        outputs = model(inputs, domains, const_style=True)
                else:
                    outputs = model(inputs, domains)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            # Calculate test accuracy
            test_accuracy = correct_predictions / total_samples
            print(f"ID Accuracy: {test_accuracy * 100:.2f}%")
            if args.wandb:
                args.tracker.log({
                    'ID Accuracy': test_accuracy
                }, step=epoch)
            
            correct_predictions = 0
            total_samples = 0
            
            print(f'Test out-of-domain data')
            count = 0
            for inputs, labels in test_out_domain_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                if args.method == 'conststyle' or args.method == 'conststyle-bn':
                    if epoch > 0:
                        domains = torch.full((len(labels), 1), 4)
                        outputs = model(inputs, domains, const_style=True, store_feats=True)
                    else:
                        outputs = model(inputs, domains)
                else:
                    domains = torch.full((len(labels), 1), 4)
                    outputs = model(inputs, domains, store_feats=True)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            # Calculate test accuracy
            test_accuracy = correct_predictions / total_samples
            if test_accuracy >= max_accuracy:
                max_accuracy = test_accuracy

            print(f"OD Accuracy: {test_accuracy * 100:.2f}%")
            if args.wandb:
                args.tracker.log({
                    'OD Accuracy': test_accuracy
                }, step=epoch)
            
        # if epoch > 0:
        #     model.plot_style(args, epoch)

    print(f"Training finished | Max Accuracy: {max_accuracy}")
    if args.wandb:
        args.tracker.log({
                    'Max OD Accuracy': max_accuracy
                })

    save_path = f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}'

    with open(f'{save_path}/acc.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Acc', max_accuracy])

def main(args):
    print('Create folder to store results...')
    save_path = f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if os.path.exists(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}/acc.csv'):
        os.remove(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}/acc.csv')
    print('Created')
        
    train_dataset = []
    test_dataset = []
    len_dataset = []
    idx = 1
    
    print(f'Setting up data for dataset {args.dataset} with test domain {args.test_domains}...')
    if args.dataset == 'pacs':
        dataset_list = prepare_pacs(args)
        if 'p' in args.train_domains:
            total_data = len(dataset_list[0])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_p, test_dataset_p = random_split(dataset_list[0], [train_len, test_len])
            train_dataset.append(train_dataset_p)
            test_dataset.append(test_dataset_p)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 'a' in args.train_domains:
            total_data = len(dataset_list[1])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_a, test_dataset_a = random_split(dataset_list[1], [train_len, test_len])
            train_dataset.append(train_dataset_a)
            test_dataset.append(test_dataset_a)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 'c' in args.train_domains:
            total_data = len(dataset_list[2])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_c, test_dataset_c = random_split(dataset_list[2], [train_len, test_len])
            train_dataset.append(train_dataset_c)
            test_dataset.append(test_dataset_c)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 's' in args.train_domains:
            total_data = len(dataset_list[3])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_s, test_dataset_s = random_split(dataset_list[3], [train_len, test_len])
            train_dataset.append(train_dataset_s)
            test_dataset.append(test_dataset_s)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        
        if 'p' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[0], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 'a' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[1], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 'c' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[2], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 's' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[3], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        
    elif args.dataset == 'officehome':
        dataset_list = prepare_officehome(args)
        dataset_list = prepare_pacs(args)
        if 'a' in args.train_domains:
            total_data = len(dataset_list[0])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_p, test_dataset_p = random_split(dataset_list[0], [train_len, test_len])
            train_dataset.append(train_dataset_p)
            test_dataset.append(test_dataset_p)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 'c' in args.train_domains:
            total_data = len(dataset_list[1])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_a, test_dataset_a = random_split(dataset_list[1], [train_len, test_len])
            train_dataset.append(train_dataset_a)
            test_dataset.append(test_dataset_a)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 'p' in args.train_domains:
            total_data = len(dataset_list[2])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_c, test_dataset_c = random_split(dataset_list[2], [train_len, test_len])
            train_dataset.append(train_dataset_c)
            test_dataset.append(test_dataset_c)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        if 'r' in args.train_domains:
            total_data = len(dataset_list[3])
            train_len = int(total_data * 0.8)
            test_len = total_data - train_len
            train_dataset_s, test_dataset_s = random_split(dataset_list[3], [train_len, test_len])
            train_dataset.append(train_dataset_s)
            test_dataset.append(test_dataset_s)
            len_dataset.append(torch.full((train_len, 1), idx))
            idx += 1
        
        if 'a' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[0], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 'c' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[1], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 'p' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[2], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
        elif 'r' == args.test_domains:
            test_out_domain_loader = DataLoader(dataset_list[3], batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
    
        
    concated_train_dataset = ConcatDataset(train_dataset)
    concated_test_dataset = ConcatDataset(test_dataset)
    concated_train_domain = torch.vstack(len_dataset)
    train_loader = DataLoader(list(zip(concated_train_dataset, concated_train_domain)), batch_size=64, shuffle=True, num_workers=8, worker_init_fn=seed_worker)
    test_in_domain_loader = DataLoader(concated_test_dataset, batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker)

    if args.method == 'conststyle':
        model = ConstStyleModel()
    if args.method == 'conststyle-bn':
        model = ConstStyleModel2()
    elif args.method == 'mixstyle':
        model = MixStyleModel()
    elif args.method == 'dsu':
        model = DSUModel(1, uncertainty=0.5)
    elif args.method == 'csu':
        model = CSUModel(['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], 0.5, 0.1)
    elif args.method == 'baseline':
        model = BaselineModel()

    if args.dataset == 'pacs':
        model.model.fc = torch.nn.Linear(model.model.fc.in_features, 7)
    elif args.dataset == 'officehome':
        model.model.fc = torch.nn.Linear(model.model.fc.in_features, 65)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=1e-4, weight_decay=1e-5)
    print(f'Setup done')
    train(args, model, train_loader, test_in_domain_loader, test_out_domain_loader, criterion, optimizer)

if __name__ == "__main__":
    main(args)
