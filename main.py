import time
import datetime
import gc
import argparse
import torch
import torch.cuda
from src.server import *
from src.client import *
import src.datasets as my_datasets
# from dataclasses import dataclass
from src.splitter import *
from src.utils import *
from src.dataset_bundle import *
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset

import wandb
from wandb_env import WANDB_ENTITY, WANDB_PROJECT
"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""
def main(args):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "nmalloc"
    
    #load hyperparam
    hparam = vars(args)
    config_file = args.config_file
    with open(config_file) as fh:
        config = json.load(fh)
    hparam.update(config)
    wandb_project = WANDB_PROJECT
    running_name = f"iid={hparam['dataset']['iid']}-nclient={hparam['global']['num_clients']}-nround={hparam['server']['num_rounds']}-seed={hparam['global']['seed']}_same_space"
    # setup WanDB
    wandb.init(project=wandb_project,
                entity=WANDB_ENTITY,
                group=hparam['global']['dataset'],
                name=running_name,
                job_type=f"{hparam['server']['algorithm']}-{hparam['client']['algorithm']}",
                config=hparam)
    wandb.run.log_code()
    
    config['id'] = wandb.run.id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Current available device: {device}')
    seed = hparam['global']['seed']
    set_seed(seed)
    
    data_path = hparam['global']['data_path']
    if not os.path.exists(data_path + "opt_dict/"): os.makedirs(data_path + f"opt_dict/{running_name}")
    if not os.path.exists(data_path + "sch_dict/"): os.makedirs(data_path + f"sch_dict/{running_name}/")
    if not os.path.exists(data_path + "models/"): os.makedirs(data_path + f"models/{running_name}")

    # optimizer preprocess
    if hparam['client']['optimizer'] == 'torch.optim.SGD':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'momentum': hparam['momentum'], 'weight_decay': hparam['weight_decay']}
    elif hparam['client']['optimizer'] == 'torch.optim.Adam' or hparam['optimizer'] == 'torch.optim.AdamW':
        hparam['optimizer_config'] = hparam['client']['optimizer_config']

    # initialize data
    if hparam['global']['dataset'].lower() == 'pacs':
        dataset = my_datasets.PACS(version='1.0', root_dir=hparam['dataset']['dataset_path'], download=True)
    elif hparam['global']['dataset'].lower() == 'officehome':
        dataset = my_datasets.OfficeHome(version='1.0', root_dir=hparam['dataset']['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
    elif hparam['global']['dataset'].lower() == 'femnist':
        dataset = my_datasets.FEMNIST(version='1.0', root_dir=hparam['dataset']['dataset_path'], download=True)
    elif hparam['global']['dataset'].lower() == 'celeba':
        dataset = get_dataset(dataset="celebA", root_dir=hparam['dataset']['dataset_path'], download=True)
    else:
        dataset = get_dataset(dataset=hparam['global']["dataset"].lower(), root_dir=hparam['dataset']['dataset_path'], download=True)
    # if server_config['algorithm'] == "FedDG":
    #     # make it easier to hash fourier transformation
    #     indices = torch.arange(len(dataset)).reshape(-1,1)
    #     new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    #     dataset._metadata_array = new_metadata_array
    if hparam['client']['algorithm'] == "FedSR":
        ds_bundle = eval(hparam['global']["dataset"])(dataset, hparam['global']["feature_dimension"], probabilistic=True)
    elif hparam['client']['algorithm'] == "ProposalClient":
        ds_bundle = eval(f"{hparam['global']['dataset']}Proposal")(dataset, hparam['global']["feature_dimension"], hparam, probabilistic=False)
    else:
        if hparam['global']['dataset'].lower() == 'py150' or hparam['global']['dataset'].lower() == 'civilcomments':
            ds_bundle = eval(hparam['global']["dataset"])(dataset, probabilistic=False)
        else:
            ds_bundle = eval(hparam['global']["dataset"])(dataset, hparam['global']["feature_dimension"], probabilistic=False)
            
    if hparam['client']['algorithm'] == "FedDG":
        if hparam['global']["dataset"].lower() == "iwildcam":
            dataset = my_datasets.FourierIwildCam(root_dir=hparam, download=True)
            total_subset = dataset.get_subset('train', transform=ds_bundle.test_transform)
        elif hparam['global']["dataset"].lower() == "pacs":
            dataset = my_datasets.FourierPACS(root_dir=hparam, download=True, split_scheme=hparam['global']["split_scheme"])
            total_subset = dataset.get_subset('train', transform=ds_bundle.test_transform)
        else:
            raise NotImplementedError
    else:
        total_subset = dataset.get_subset('train', transform=ds_bundle.train_transform)

    testloader = {}
    for split in dataset.split_names:
        if split != 'train':
            ds = dataset.get_subset(split, transform=ds_bundle.test_transform)
            dl = get_eval_loader(loader='standard', dataset=ds, batch_size=hparam['global']["batch_size"])
            testloader[split] = dl

    
    sampler = RandomSampler(total_subset, replacement=True)
    global_dataloader = DataLoader(total_subset, batch_size=hparam['global']["batch_size"], sampler=sampler)
    # # DS
    # out_test_dataset, test_train = RandomSplitter(ratio=0.5, seed=seed).split(out_test_dataset)
    # out_test_dataset.transform = ds_bundle.test_transform
    # out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset, batch_size=global_config["batch_size"])
    # if global_config['cheat']:
    #     total_subset = concat_subset(total_subset, test_train)
    # training_datasets = [total_subset]
    # print(len(total_subset), len(in_validation_dataset), len(lodo_validation_dataset), len(in_test_dataset), len(out_test_dataset))
    num_shards = hparam['global']['num_clients']
    if num_shards == 1:
        training_datasets = [total_subset]
    elif num_shards > 1:
        training_datasets = NonIIDSplitter(num_shards=num_shards, iid=hparam['dataset']['iid'], seed=seed).split(dataset.get_subset('train'), ds_bundle.groupby_fields, transform=ds_bundle.train_transform)
    else:
        raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))

    # initialize client
    clients = []
    for k in tqdm(range(hparam['global']["num_clients"]), leave=False):
        client = eval(hparam['client']["algorithm"])(k, device, training_datasets[k], ds_bundle, hparam['client'])
        clients.append(client)
    print(f"successfully initialize all clients!")

    # initialize server (model should be initialized in the server. )
    central_server = eval(hparam['server']["algorithm"])(seed, config['id'], device, ds_bundle, hparam['server'])
    if hparam['server']['algorithm'] == "FedDG":
        central_server.set_amploader(global_dataloader)
    if hparam['global']['start_epoch'] == 0:
        central_server.setup_model(None, 0)
    else:
        central_server.setup_model(hparam['resume_file'], hparam['start_epoch'])
    central_server.register_clients(clients)
    central_server.register_testloader(testloader)
    # do federated learning
    central_server.fit()
    
    # bye!
    print("...done all learning process!\n...exit program!")
    time.sleep(3)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedDG Benchmark')
    parser.add_argument('--config_file', help='config file', default="config.json")
    parser.add_argument('--running_name', help='name of wandb run')
    
    args = parser.parse_args()
    main(args)

