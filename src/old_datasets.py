import os
from typing import Callable, Optional
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torchvision.datasets import MNIST, ImageFolder
from wilds.datasets.wilds_dataset import WILDSSubset
from torchvision.transforms.functional import rotate
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from .splitter import *
from sklearn.metrics import f1_score
from wilds.common.grouper import CombinatorialGrouper

    
class WildsDataset(object):
    def __init__(self, data_config):
        self.data_config = data_config
        self.root_dir = self.data_config["data_path"]
        self.num_shards = self.data_config["num_shards"]
        self.iid = self.data_config["iid"]
        self._classes = None
        self._training_datasets = None
        self._out_test_dataset = None
        self._in_test_dataset = None
        self._lodo_validation_dataset = None
        self._in_validation_dataset = None
        self._split_flag = False
        self.criterion =  self.data_config["criterion"]
        self.input_shape = (3, 224, 224)
        self.dataset = get_dataset(dataset=self.name, root_dir=self.root_dir, download=True)
        self.dataset.groupby_id = self.domain_field
        self.dataset.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.groupby_fields)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = int(self.data_config['seed'])

    def _split_dataset(self):
        self._training_dataset = self.dataset.get_subset('train')
        self._training_datasets = NonIIDSplitter(num_shards=self.num_shards, iid=self.iid, seed=self.seed).split(self._training_dataset, self.domain_field, transform=self.train_transform)
        # for dataset in self._training_datasets:
        #     dataset.groupby_fields = self.groupby_fields
        self._in_validation_dataset = self.dataset.get_subset('id_val', transform=self.test_transform)
        # self._in_validation_dataset.groupby_fields = self.groupby_fields
        self._lodo_validation_dataset = self.dataset.get_subset('val', transform=self.test_transform)
        # self._lodo_validation_dataset.groupby_fields = self.groupby_fields
        self._in_test_dataset = self.dataset.get_subset('id_test',transform = self.test_transform)
        # self._in_test_dataset.groupby_fields = self.groupby_fields
        self._out_test_dataset = self.dataset.get_subset('test', transform=self.test_transform)
        self._split_flag = True

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.input_shape[1],self.input_shape[2])),
            transforms.RandomResizedCrop(self.input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def eval(self, y, labels):
        # classification with accuracy by default.
        loss = self.criterion()(y, labels).item()
        predicted = y.argmax(dim=1, keepdim=True)
        correct = predicted.eq(labels.view_as(predicted)).sum().item()
        return {"loss": loss, "acc": correct/len(y)}
    
    @property
    def domain_field(self):
        return None

    # doesn't require re-implemented.    
    @property
    def num_classes(self):
        return self.dataset.n_classes

    @property
    def in_channel(self):
        return self.input_shape[0]

    @property
    def lodo_validation_dataset(self):
        if not self._split_flag:
            self._split_dataset()
        return self._lodo_validation_dataset

    @property
    def out_test_dataset(self):
        if not self._split_flag:
            self._split_dataset()
        return self._out_test_dataset

    @property
    def in_test_dataset(self):
        if not self._split_flag:
            self._split_dataset()
        return self._in_test_dataset

    # @property
    # def out_validation_dataset(self):
    #     if not self._split_flag:
    #         self._split_dataset()
    #     return self._out_validation_dataset

    @property
    def in_validation_dataset(self):
        if not self._split_flag:
            self._split_dataset()
        return self._in_validation_dataset
    
    @property
    def training_dataset(self):
        if not self._split_flag:
            self._split_dataset()
        return self._training_dataset

    @property
    def training_datasets(self):
        if not self._split_flag:
            self._split_dataset()
        return self._training_datasets

    @property
    def name(self):
        return self.__class__.__name__.lower()
        
    @staticmethod
    def transform(datasets, transform):
        for _, dataset in datasets.items():
            dataset.transform = transform

    @property
    def domains(self):
        if self._domains is None:
            _domain = []
            metadata_name = self.dataset.metadata_fields[self.domain_field]
            metadata_array = self.dataset.metadata_array[:, self.domain_field]
            metadata_number = torch.sort(torch.unique(metadata_array))[0]
            for i in metadata_number:
                _domain.append(f"{metadata_name}_{i}")
            self._domains = _domain
        return self._domains
    
    @property
    def groupby_fields(self):
        return [self.dataset.metadata_fields[self.domain_field]]

class IWildCam(WildsDataset):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.input_shape = (3, 448, 448)

    @property
    def domain_field(self):
        return 0

    def eval(self, outputs, y_true):
        loss = self.criterion(outputs, y_true).item()
        y_pred = outputs.argmax(dim=1)
        return {"loss": loss/(len(y_pred)), "accuracy": f1_score(y_pred, y_true, average='micro'), "f1": f1_score(y_pred, y_true, average='macro')}


class DomainNet(WildsDataset):
    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.input_shape[1],self.input_shape[2])),
            transforms.RandomResizedCrop(self.input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
        
    @property
    def test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.input_shape[1],self.input_shape[2])),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
    
    def _split_dataset(self):
        self._training_dataset = self.dataset.get_subset('train')
        self._in_validation_dataset = RandomSplitter(ratio=0.1, seed=self.seed).split(self._training_dataset, transform=self.test_transform)
        self._training_datasets = NonIIDSplitter(num_shards=self.num_shards, iid=self.iid, seed=self.seed).split(self._training_dataset, self.domain_field, transform=self.train_transform)
        self._lodo_validation_dataset = self.dataset.get_subset('val', transform=self.test_transform)
        self._in_test_dataset = self.dataset.get_subset('id_test',transform = self.test_transform)
        self._out_test_dataset = self.dataset.get_subset('test', transform=self.test_transform)
        self._split_flag = True

    @property
    def domain_field(self):
        return None


class Py150(WildsDataset):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.input_shape = (255)
    @property
    def domain_field(self):
        return 0
    
    @property
    def train_transform(self):
        return None
    
    @property
    def test_transform(self):
        return None