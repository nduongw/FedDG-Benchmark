import logging
import os
import copy
import time
import random
import math
from torch import optim
from urllib.parse import _NetlocResultMixinBytes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_train_loader
from wilds.common.utils import split_into_groups
from src.models import Discriminator
import multiprocessing as mp
from src.utils import *
logger = logging.getLogger(__name__)



class ERM(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        """Client object is initiated by the center server."""
        self.seed = seed
        self.id = client_id
        self.device = device
        self.featurizer = None
        self.classifier = None
        self.model = None
        self.dataset = dataset # Wrapper of WildSubset.
        self.ds_bundle = ds_bundle
        self.client_config = client_config
        self.n_groups_per_batch = client_config['n_groups_per_batch']
        self.local_epochs = self.client_config['local_epochs']
        self.batch_size = self.client_config["batch_size"]
        self.optimizer_name = self.client_config['optimizer']
        self.optim_config = self.client_config['optimizer_config']
        self.dataloader = get_train_loader(self.loader_type, self.dataset, batch_size=self.batch_size, uniform_over_groups=None, grouper=self.ds_bundle.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.saved_optimizer = False
        self.opt_dict_path = "/local/scratch/a/bai116/opt_dict/client_{}.pt".format(self.id)
        if os.path.exists(self.opt_dict_path): os.remove(self.opt_dict_path)

    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))

    @property
    def loader_type(self):
        return 'standard'
    

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)
    
    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
        
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        del self.optimizer
        if self.device == "cuda": torch.cuda.empty_cache()

    def fit(self):
        """Update local model using local dataset."""
        self.init_train()
        for e in range(self.local_epochs):
            for batch in self.dataloader:
                results = self.process_batch(batch)
                self.step(results)
        self.end_train()

    def evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            metric = {}
            y_pred = None
            y_true = None
            for batch in self.dataloader:
                results = self.process_batch(batch)
                if y_pred is None:
                    y_pred = results['y_pred']
                    y_true = results['y_true']
                else:
                    y_pred = torch.cat((y_pred, results['y_pred']))
                    y_true = torch.cat((y_true, results['y_true']))

            metric_new = self.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"))
            print(metric_new)
            for key, value in metric_new.items():
                if key not in metric.keys():
                    metric[key] = value
                else:
                    metric[key] += value
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric
    
    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        outputs = self.model(x)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def step(self, results):
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        
        if objective.grad_fn is None:
            # print('jump')
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def name(self):
        return self.__class__.__name__
    
    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataset)


class IRM(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.penalty_weight = client_config['penalty_weight']
        self.penalty_anneal_iters = client_config['penalty_anneal_iters']
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.update_count = 0
    
    @property
    def loader_type(self):
        return 'group'  
    
    def step(self, results):
        unique_groups, group_indices, _ = split_into_groups(results['g'])
        n_groups_per_batch = unique_groups.numel()
        avg_loss = 0.
        penalty = 0.
        # torch.save(results['y_pred'], "pred.pt")
        # torch.save(results['y_true'], "true.pt")
        for i_group in group_indices: # Each element of group_indices is a list of indices
            # print(i_group)
            group_losses, _ = self.ds_bundle.loss.compute_flattened(results['y_pred'][i_group] * self.scale, results['y_true'][i_group], return_dict=False)
            if group_losses.numel()>0:
                avg_loss += group_losses.mean()
            penalty += self.irm_penalty(group_losses)
        avg_loss /= n_groups_per_batch
        penalty /= n_groups_per_batch
        if self.update_count >= self.penalty_anneal_iters:
            penalty_weight = self.penalty_weight
        else:
            penalty_weight = self.update_count / self.penalty_anneal_iters
        penalty_weight = 0.
        # print(self.update_count, penalty_weight)
        objective = avg_loss + penalty * penalty_weight
        # print(avg_loss, penalty, objective)
        # wprint(avg_loss, penalty)
        if self.update_count == self.penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = eval(self.optimizer_name)(params, **self.optim_config)
        if objective.grad_fn is None:
            pass
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_count += 1

    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result
    
    
class Fish(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.meta_lr = client_config["meta_lr"]
    
    @property
    def loader_type(self):
        return 'group'
    
    def fit(self):
        self.init_train()
        for e in range(self.local_epochs):
            for batch in self.dataloader:
                self.step(batch)
            if self.device == "cuda": torch.cuda.empty_cache()            
        self.end_train()
    
    def step(self, batch):
        param_dict = ParamDict(copy.deepcopy(self.model.state_dict()))
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        unique_groups, group_indices, _ = split_into_groups(g)
        for i_group in group_indices: # Each element of group_indices is a list of indices
            # print(i_group)
            group_loss = self.ds_bundle.loss.compute(self.model(x[i_group]), y_true[i_group], return_dict=False)
            # print(group_loss)
            # print(group_loss.grad_fn)
            if group_loss.grad_fn is None:
                # print('jump')
                pass
            else:
                group_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        param_dict = param_dict + self.meta_lr * (ParamDict(self.model.state_dict()) - param_dict)
        self.model.load_state_dict(copy.deepcopy(param_dict))


class MMD(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.penalty_weight = client_config["penalty_weight"]        

    @property
    def loader_type(self):
        return 'group'

    
    def penalty(self,x,y):
        def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
            if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
                x = x.view(-1, x.size(-1))
                y = y.view(-1, y.size(-1))
            def my_cdist(x1, x2):
                x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
                x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
                res = torch.addmm(x2_norm.transpose(-2, -1),
                                x1,
                                x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
                return res.clamp_min_(1e-30)
            D = my_cdist(x, y)
            K = torch.zeros_like(D)

            for g in gamma:
                K.add_(torch.exp(D.mul(-g)))
            return K

        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # forward pass
        x, y_true, metadata = batch
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }
        x = x.to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)
        y_pred = outputs[: len(y_true)]
        results['features'] = features
        results['y_pred'] = y_pred
        return results

    def step(self, results):
        features = results.pop('features')
        unique_groups, group_indices, _ = split_into_groups(results['g'])
        n_groups_per_batch = unique_groups.numel()
        penalty = torch.zeros(1, device=self.device)

        for i_group in range(n_groups_per_batch): # Each element of group_indices is a list of indices
            for j_group in range(i_group+1, n_groups_per_batch):
                penalty += self.penalty(features[group_indices[i_group]], features[group_indices[j_group]])
            if n_groups_per_batch > 1:
                penalty /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty
        else:
            penalty = 0.
        avg_loss = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        objective =  avg_loss + penalty * self.penalty_weight
        if objective.grad_fn is None:
            pass
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



class Coral(MMD):
    def penalty(self,x,y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff


class GroupDRO(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.group_weights_step_size = client_config['groupdro_eta']
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_g = self.ds_bundle.grouper.metadata_to_group(self.dataset.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.group_weights = self.group_weights.to(self.device)

    def step(self, results):
        loss = torch.zeros_like(self.group_weights)
        unique_groups, group_indices, _ = split_into_groups(results['g'])
        for group_idx, i_group in zip(unique_groups, group_indices):
            group_losses = self.ds_bundle.loss.compute(results['y_pred'][i_group], results['y_true'][i_group], return_dict=False).mean()
            loss[group_idx] = group_losses
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*loss.data)
        self.group_weights = (self.group_weights/(self.group_weights.sum()))       
        objective = self.group_weights @ loss
        if objective.grad_fn is None:
            # print('jump')
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()


class Mixup(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.alpha = client_config['alpha']
    
    @property
    def loader_type(self):
        return 'group'

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        _, group_indices, _ = split_into_groups(g)
        lam = np.random.beta(self.alpha, self.alpha)

        outputs = self.model(lam * x[group_indices[0]] + (1 - lam) * x[group_indices[1]])
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'lam': lam
        }
        return results
    
    def step(self, results):
        
        _, group_indices, _ = split_into_groups(results['g'])
        objective = results['lam'] * self.ds_bundle.loss.compute(results['y_pred'], results['y_true'][group_indices[0]], return_dict=False).mean() + (1 - results['lam']) * self.ds_bundle.loss.compute(results['y_pred'], results['y_true'][group_indices[1]], return_dict=False).mean()
        if objective.grad_fn is None:
            # print('jump')
            pass
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class FourierMixup(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self.ratio = self.client_config['ratio']

    def set_global_dataloader(self, dataloader):
        self.global_dataloader = iter(dataloader) # list of indices of dataset

    @property
    def loader_type(self):
        return 'standard'

    def process_batch(self, batch):
        x, y_true, [metadata, amp, pha] = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        amp = amp.to(self.device)
        pha = pha.to(self.device)
        lmda = random.uniform(0, 1)
        _,_,[_, sampled_amp, _] = next(self.global_dataloader)
        sampled_amp = sampled_amp.to(self.device)
        new_amp = self._amp_spectrum_swap(amp, sampled_amp, L=lmda, ratio=self.ratio)
        fft_local_ = new_amp * torch.exp(1j * pha)
        new_x = torch.real(torch.fft.ifft2(fft_local_))

        _, group_indices, _ = split_into_groups(g)

        outputs = self.model(new_x)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata
        }
        return results


    @staticmethod
    def _amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
        a_local = torch.fft.fftshift(amp_local, dim=(-2, -1))
        a_trg = torch.fft.fftshift(amp_target, dim=(-2, -1))

        _, _, h, w = a_local.shape
        b = int(min(h,w) * L)
        c_h = int(h/2)
        c_w = int(w/2)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1

        a_local[:, :,h1:h2,w1:w2] = a_local[:, :,h1:h2,w1:w2] * ratio + a_trg[:, :,h1:h2,w1:w2] * (1 - ratio)
        a_local = torch.fft.ifftshift(a_local, dim=(-2, -1))
        return a_local


class FedADG(ERM):
    def __init__(self, seed, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, client_id, device, dataset, ds_bundle, client_config)
        self._generator = None
        self._discriminator = None
        self.alpha = self.client_config['alpha']
    
    def setup_model(self, featurizer, classifier, generator):
        super().setup_model(featurizer, classifier)
        self._generator = generator
        self.generator = nn.DataParallel(self._generator)
        self._discriminator = Discriminator(self.featurizer.n_outputs)
        self.discriminator = nn.DataParallel(self._discriminator)

    def update_model(self, model_dict, generator_dict):
        super().update_model(model_dict)
        self._generator.load_state_dict(generator_dict)
        
    def init_train(self):
        super().init_train()
        self.generator.train()
        self.generator.to(self.device)
        self.discriminator.train()
        self.discriminator.to(self.device)
        self.gen_optimizer_name = self.client_config['gen_optimizer_name']
        self.gen_optimizer_config = self.client_config['gen_optimizer_config']
        self.disc_optimizer_name = self.client_config['disc_optimizer_name']
        self.disc_optim_config = self.client_config['disc_optim_config']
        # self.opti_encoder = optim.SGD(self.featurizer.parameters(), args.lr1, args.momentum,weight_decay=args.weight_dec)

        self.criterion = LabelSmoothingLoss(0.2, lbl_set_size=5)
        self.disc_optimizer = eval(self.disc_optimizer_name)(self.discriminator.parameters(), **self.disc_optim_config)
        self.gen_optimizer = eval(self.gen_optimizer_name)(self.generator.parameters(), **self.gen_optim_config)
        # scheduler to update learning rate
        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=20, threshold=1e-4, min_lr=1e-7)
        self.optimizer_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=500, after_scheduler=afsche_task)
        afsche_gen = optim.lr_scheduler.ReduceLROnPlateau(self.gen_optimizer, factor=0.2, patience=20,threshold=1e-4, min_lr=1e-7)
        self.gen_optimizer_scheduler = GradualWarmupScheduler(self.gen_optimizer, total_epoch=500, after_scheduler=afsche_gen)
        afsche_dis = optim.lr_scheduler.ReduceLROnPlateau(self.disc_optimizer, factor=0.2, patience=20, threshold=1e-4, min_lr=1e-7)
        self.disc_optimizer_scheduler = GradualWarmupScheduler(self.disc_optimizer, total_epoch=500, after_scheduler=afsche_dis)
        
    def end_train(self):
        self.generator.to("cpu")
        self.discriminator.to("cpu")
        super().end_train()
        

    def fit(self):
        """Update local model using local dataset."""
        self.init_train()
        for e in range(self.first_local_epochs):
            for batch in self.dataloader:
                x, y_true = batch
                x = x.to(self.args.device)
                y_true = y_true.to(self.device)
                self.optimizer.zero_grad()
                feature = self.featurizer(x)
                y = self.classifier(feature)
                loss = self.criterion(y, y_true)
                loss.backward()
                self.optimizer.step()
            metric = self.evaluate()
            self.optimizer_scheduler.step(e, 1. - metric['acc']) #TODO: metric?
                
        for e in range(self.second_local_epochs):
            for t, batch in enumerate(self.dataloader):
                self.process_batch(batch)
            metric = self.evaluate()
            # update learning rate
            self.disc_optimizer_scheduler.step(e + 1, 1. - metric['acc'])
            self.gen_optimizer_scheduler.step(e + 1, 1. - metric['acc'])
            self.optimizer_scheduler.step(e + 5 + 1, 1. - metric['acc'])
        self.end_train()

    def process_batch(self, batch):
        self.featurizer.train()
        self.classifier.train()
        self.discriminator.eval()
        self.generator.eval()
        
        x, y = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        y_onehot = torch.zeros(y_true.size(0), self.dataset.n_classes).to(self.device)
        y_onehot.scatter_(1, y.view(-1, 1), 0.6).to(self.device)
        randomn = torch.rand(y_true.size(0), self.ds_bundle.input_size).to(self.device)

        # training feature extractor and classifier
        self.optimizer.zero_grad()
        feature = self.featurizer(x)
        y = self.classifier(feature)
        loss = self.criterion(y, y_true)
        loss_enc = torch.mean(torch.pow(1 - self.discriminator(y_onehot, feature), 2))
        loss_cla = self.alpha * loss + (1 - self.alpha) * loss_enc
        loss_cla.backward()
        self.optimizer.step()
    
        # training discriminator
        self.featurizer.eval()
        self.discriminator.train()
        self.disc_optimizer.zero_grad()
        feature = self.featurizer(x).detach()
        gen_feature = self.generator(y=y_onehot,x=randomn,device=self.device).detach()
        loss_discriminator = -torch.mean(torch.pow(self.discriminator(y_onehot, gen_feature), 2) + torch.pow(1-self.discriminator(y_onehot, feature),2))
        loss_discriminator.backward()
        self.disc_optimizer.step()
        self.discriminator.eval()

        # training distribution generator
        self.generator.train()
        self.gen_optimizer.zero_grad()
        gen_feature = self.generator(y=y_onehot,x=randomn,device=self.device).detach()
        loss_gene = torch.mean(torch.pow(1-self.discriminator(y_onehot, gen_feature), 2))
        loss_gene.backward()
        self.gen_optimizer.step()
    
    @property
    def loader_type(self):
        return 'standard'


class ScaffoldClient(ERM):
    def setup_model(self, featurizer, classifier):
        super().setup_model(featurizer, classifier)
        self.c_local = None

    def fit(self):
        """Update local model using local dataset."""
        global_model = ParamDict(self.model.state_dict())
        self.init_train()
        lr = self.optimizer.param_groups[0]['lr']
        for e in range(self.local_epochs):
            # for batch in tqdm(self.dataloader):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                self.step(results)
        
        self.end_train()
        self.model.to('cpu')
        local_model = ParamDict(self.model.state_dict())
        if self.c_local is None:
            self.c_local = (global_model - local_model) / (self.local_epochs * lr)
        else:
            self.c_local = self.c_local - self.c_global + (global_model - local_model) / (self.local_epochs * lr)
    
    def step(self, results):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            param_dict = ParamDict(copy.deepcopy(self.model).to('cpu').state_dict())
            if self.c_local is not None:
                param_dict = param_dict - self.optimizer.param_groups[0]['lr'] * (self.c_global - self.c_local)
            self.model.load_state_dict(copy.deepcopy(param_dict))


class FedProx(ERM):
    def __init__(self, seed, exp_id, client_id, device, dataset, ds_bundle, client_config):
        super().__init__(seed, exp_id, client_id, device, dataset, ds_bundle, client_config)
        self.mu = self.client_config['mu']
    def prox(self):
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        return proximal_term
    
    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        self.scheduler = eval(self.scheduler_name)(self.optimizer, **self.scheduler_config)
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))
    
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer, self.global_model
    
    def step(self, results):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean() + self.mu / 2 * self.prox()
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()
