import copy
import logging
import time
import pickle
from multiprocessing import pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *

logger = logging.getLogger(__name__)


class FedAvg(object):
    def __init__(self, seed, exp_id, device, ds_bundle, server_config):
        self.seed = seed
        self.id = exp_id
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.server_config = server_config
        self.mp_flag = server_config['mp']
        self.num_rounds = server_config['num_rounds']
        self.fraction = server_config['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None
    
    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        pbar = tqdm(self.clients)
        for client in pbar:
            pbar.set_description('Registering clients...')
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        print('Done\n')
        
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)
    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            print(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            print(message)
            del message

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(3)}] Select clients...!"
        logging.debug(message)
        print(message)
        del message

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        print(f'Selected clientss : {sampled_client_indices}')
        return sampled_client_indices
    

    def update_clients(self, sampled_client_indices, round):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(round)
            client_size = len(self.clients[selected_index])
            return client_size

        message = f"[Round: {str(self._round).zfill(3)}] Start updating selected {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        print(message)
        selected_total_size = 0
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(update_single_client, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            for idx in tqdm(sampled_client_indices, leave=False):
                client_size = update_single_client(idx)
                selected_total_size += client_size
        message = f"[Round: {str(self._round).zfill(3)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        logging.debug(message)
        print(message)
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        
        message = f"[Round: {str(self._round).zfill(3)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        logging.debug(message)
        del message
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(evaluate_single_client, sampled_client_indices)
        else:
            for idx in tqdm(sampled_client_indices):
                self.clients[idx].client_evaluate()
            

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        print(message)
        del message

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(3)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        logging.debug(message)
        print(message)
        del message
    

    def train_federated_model(self, round):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, round)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def evaluate_global_model(self, dataloader, global_round, name):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()

        with torch.no_grad():
            y_pred = None
            y_true = None
            stored_data = {
                'data': [],
                'label': [],
            }
            for idx, batch in enumerate(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                    stored_data['data'].append(features.detach().cpu().numpy())
                    stored_data['label'].append(labels.detach().cpu().numpy())
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
            np.savez(f'./results/data_round{global_round}_{name}.npz',data=stored_data)
                
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            if self.device == "cuda": torch.cuda.empty_cache()
        return metric

    def fit(self, running_name):
        """
        Description: Execute the whole process of the federated learning.
        """
        key_metric = []
        max_test_acc = 0.0
        for r in tqdm(range(self.num_rounds), desc='Training round'):
            test_acc_list = []
            print("Round {}".format(r+1))
            key_metric.append([])
            self._round += 1
            self.train_federated_model(r)
            message = f"{str(self._round).zfill(3)} \t "
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader, r, name)
                print(f"Hold out {name} results: TestAcc: {metric[0]['acc_avg']}")
                test_acc_list.append(metric[0]['acc_avg'])
                for value in metric[0].values():
                    message += f"{value:05.4} "
                message += f"\t"
                key_metric[-1].append(list(metric[0].values())[-1])
            logging.info(message)
            wandb.log({'OOD Valid Acc': test_acc_list[0]}, step=r+1)
            wandb.log({'OOD Test Acc': test_acc_list[1]}, step=r+1)
            wandb.log({'ID Valid Acc': test_acc_list[2]}, step=r+1)
            wandb.log({'ID Test Acc': test_acc_list[3]}, step=r+1)
            if max_test_acc < test_acc_list[1]:
                print(f'OOD Test Accuracy increase from {max_test_acc} to {test_acc_list[1]} --> Saving model')
                self.save_model(r, running_name)
                max_test_acc = test_acc_list[1]
        key_metric = np.array(key_metric)
        in_max_idx, lodo_max_idx, _, _ = np.argmax(key_metric, axis=0)
        print(f"{key_metric[in_max_idx][2]:05.4} \t {key_metric[in_max_idx][3]:05.4} \t {key_metric[lodo_max_idx][3]:05.4}")
        self.transmit_model()

    def save_model(self, round, running_name):
        path = f"{self.server_config['data_path']}models/{running_name}/{self.ds_bundle.name}_bestmodel_round{round}.pth"
        torch.save(self.model.state_dict(), path)


class FedDG(FedAvg):
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
            client.set_amploader(self.amploader)
        super().register_clients(clients)
            
    def set_amploader(self, amp_dataset):
        self.amploader = amp_dataset


class FedADGServer(FedAvg):
    def __init__(self, seed, exp_id, device, ds_bundle, server_config):
        super().__init__(seed, exp_id, device, ds_bundle, server_config)
        self.gen_input_size = server_config['gen_input_size']

    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._generator = GeneDistrNet(num_labels=self.ds_bundle.n_classes, input_size=self.gen_input_size, hidden_size=self._featurizer.n_outputs)
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.generator = nn.DataParallel(self._generator)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier), copy.deepcopy(self._generator))

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict(), self._generator.state_dict())

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_model(self.model.state_dict(), self._generator.state_dict())
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            del message

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message

        averaged_weights = OrderedDict()
        averaged_generator_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            local_generator_weights = self.clients[idx].generator.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]                 
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]         
            for key in self.generator.state_dict().keys():
                if it == 0:
                    averaged_generator_weights[key] = coefficients[it] * local_generator_weights[key]
                    
                else:
                    averaged_generator_weights[key] += coefficients[it] * local_generator_weights[key]
        self.model.load_state_dict(averaged_weights)
        self.generator.load_state_dict(averaged_generator_weights)
        message = f"[Round: {str(self._round).zfill(3)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        logging.debug(message)
        del message


class FedGMA(FedAvg):
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to('cpu')
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to('cpu')
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()
                # if it == 0:
                #     averaged_weights[key] = coefficients[it] * local_weights[key]
                # else:
                #     averaged_weights[key] += coefficients[it] * local_weights[key]
        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()
        # print(sign_delta[key])
        mask = abs_sign_delta.ge(self.server_config['mask_threshold'])
        # print("--mid--")
        # print(mask)
        # print("-------")
        final_mask = mask + (0-mask) * abs_sign_delta
        averaged_weights = last_weights + self.server_config['step_size'] * final_mask * sum_delta 
        self.model.load_state_dict(averaged_weights)


class ScaffoldServer(FedAvg):
    def __init__(self, seed, exp_id, device, ds_bundle, server_config):
        super().__init__(seed, exp_id, device, ds_bundle, server_config)
        self.c = None

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
                client.c_global = copy.deepcopy(self.c)

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].c_global = copy.deepcopy(self.c)
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            del message
    
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if it == 0:
                c_local = self.clients[idx].c_local
            else:
                c_local += self.clients[idx].c_local
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
    
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.c = c_local / len(sampled_client_indices)
        self.model.load_state_dict(averaged_weights)
        message = f"[Round: {str(self._round).zfill(3)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        logging.debug(message)
        del message


class ProposalServer(FedAvg):
    def __init__(self, seed, exp_id, device, ds_bundle, server_config):
        super().__init__(seed, exp_id, device, ds_bundle, server_config)
        self.global_centroid = None
    
    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self.model = self.ds_bundle.model
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)
    
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        pbar = tqdm(self.clients)
        for client in pbar:
            pbar.set_description('Registering clients...')
            client.setup_model(copy.deepcopy(self.model))
        print('Done\n')
    
    def update_clients(self, sampled_client_indices, round):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(round)
            client_size = len(self.clients[selected_index])
            return client_size

        message = f"[Round: {str(self._round).zfill(3)}] Start updating selected {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        print(message)
        selected_total_size = 0
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(update_single_client, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            for idx in tqdm(sampled_client_indices, leave=False):
                client_size = update_single_client(idx)
                selected_total_size += client_size
        message = f"[Round: {str(self._round).zfill(3)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        logging.debug(message)
        print(message)
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        
        message = f"[Round: {str(self._round).zfill(3)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        logging.debug(message)
        del message
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(evaluate_single_client, sampled_client_indices)
        else:
            for idx in tqdm(sampled_client_indices):
                self.clients[idx].client_evaluate()
            

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        print(message)
        del message

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(3)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        logging.debug(message)
        print(message)
        del message
    

    def train_federated_model(self, round):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, round)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
                client.m = self.global_centroid

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            print(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].m = self.global_centroid
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            print(message)
            del message
    
    def evaluate_global_model(self, dataloader, global_round, name):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        # import pdb
        # pdb.set_trace()
        self.model.to(self.device)
        stored_data = {
                'data': [],
                'label': [],
            }
        with torch.no_grad():
            y_pred = None
            y_pred2 = None
            y_pred3 = None
            y_true = None
            for idx, batch in enumerate(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self.model.featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    prediction, u_pred, c_pred = self.model(data)
                    features = self.model.featurizer(data)
                    stored_data['data'].append(features.detach().cpu().numpy())
                    stored_data['label'].append(labels.detach().cpu().numpy())
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                    prediction2 = torch.argmax(u_pred, dim=-1)
                    prediction3 = torch.argmax(c_pred, dim=-1)
                    # import pdb
                    # pdb.set_trace()
                if y_pred is None:
                    y_pred = prediction
                    y_pred2 = prediction2
                    y_pred3 = prediction3
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_pred2 = torch.cat((y_pred2, prediction2))
                    y_pred3 = torch.cat((y_pred3, prediction3))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
            # import pdb
            # pdb.set_trace()
            np.savez(f'./results/data_round{global_round}_{name}.npz',data=stored_data)
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            metric2 = self.ds_bundle.dataset.eval(y_pred2.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            metric3 = self.ds_bundle.dataset.eval(y_pred3.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric, metric2, metric3

    def fit(self, running_name):
        """
        Description: Execute the whole process of the federated learning.
        """
        key_metric = []
        for r in tqdm(range(self.num_rounds), desc='Training round'):
            test_acc_list = []
            print("Round {}".format(r+1))
            key_metric.append([])
            self._round += 1
            self.train_federated_model(r)
            message = f"{str(self._round).zfill(3)} \t "
            for name, dataloader in self.test_dataloader.items():
                metric, metric2, metric3 = self.evaluate_global_model(dataloader, r, name)
                print(f"Hold out {name} results: Overall TestAcc: {metric[0]['acc_avg']}")
                print(f"Hold out {name} results: Uncertainty TestAcc: {metric2[0]['acc_avg']}")
                print(f"Hold out {name} results: Classification TestAcc: {metric3[0]['acc_avg']}")
                test_acc_list.append(metric[0]['acc_avg'])
                for value in metric[0].values():
                    message += f"{value:05.4} "
                message += f"\t"
                key_metric[-1].append(list(metric[0].values())[-1])
            logging.info(message)
            wandb.log({'OOD Valid Acc': test_acc_list[0]}, step=r+1)
            wandb.log({'OOD Test Acc': test_acc_list[1]}, step=r+1)
            wandb.log({'ID Valid Acc': test_acc_list[2]}, step=r+1)
            wandb.log({'ID Test Acc': test_acc_list[3]}, step=r+1)
            self.save_model(r, running_name)
        key_metric = np.array(key_metric)
        # import pdb
        # pdb.set_trace()
        in_max_idx, lodo_max_idx, _, _ = np.argmax(key_metric, axis=0)
        print(f"{key_metric[in_max_idx][2]:05.4} \t {key_metric[in_max_idx][3]:05.4} \t {key_metric[lodo_max_idx][3]:05.4}")
        self.transmit_model()
    
    

    def save_model(self, num_epoch, running_name):
        path = f"{self.server_config['data_path']}models/{running_name}/{self.ds_bundle.name}_{self.clients[0].name}_{self.id}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        print(message)
        del message

        averaged_weights = OrderedDict()
        averaged_centroid = torch.zeros_like(self.clients[0].model.m).to(self.device)
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            averaged_centroid += self.clients[idx].model.m / len(sampled_client_indices)
            for key in self.model.state_dict().keys():
                if key == 'W':
                    if it == 0:
                        averaged_weights[key] = local_weights[key] / len(sampled_client_indices)
                    else:
                        averaged_weights[key] += local_weights[key] / len(sampled_client_indices)
                else:
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)
        self.global_centroid = averaged_centroid

        message = f"[Round: {str(self._round).zfill(3)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        logging.debug(message)
        print(message)
        del message