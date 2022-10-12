import torch
import torch.nn as nn
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss

from .splitter import *
from .models import ResNet, code_gpt_py

class ObjBundle(object):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = ResNet(self.input_shape)
        self.classifier = torch.nn.Linear(self.featurizer.n_outputs, dataset.n_classes)

    @property
    def _train_transform(self):
        raise NotImplementedError

    @property
    def _test_transform(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    @property
    def _input_shape(self):
        return None
    
    @property
    def _domain_fields(self):
        return None

    @property
    def n_classes(self):
        return self.dataset.n_classes

#### Doesn't require re-implemented by derived classes ####
    @property
    def in_channel(self):
        return self._input_shape[0]

    @property
    def name(self):
        return self.__class__.__name__.lower()


class IWildCam(ObjBundle):
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        

    @property
    def _input_shape(self):
        return (3, 448, 448)
    
    @property
    def _domain_fields(self):
        return ['location',]


class DomainNet(ObjBundle):
    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (3, 224, 224)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
        
    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def _domain_fields(self):
        return ['domain',]


class Py150(ObjBundle):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer, self.classifier = code_gpt_py()

    def _loss(self):
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    @property
    def _domain_fields(self):
        return ["repo",]
    
    @property
    def _train_transform(self):
        return None
    
    @property
    def _test_transform(self):
        return None
    
    @property
    def _input_shape(self):
        return (255)
