import torch
import torch.nn as nn
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE

from .splitter import *
from .models import ResNet, code_gpt_py, Classifier, DistilBertFeaturizer, CNN, UEModel
from transformers import DistilBertTokenizerFast

class ObjBundle(object):
    def __init__(self, dataset, feature_dimension, probabilistic=False) -> None:
        self.dataset = dataset
        self.feature_dimension = feature_dimension
        self.probabilistic = probabilistic
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = ResNet(self.input_shape, self.feature_dimension, probabilistic=probabilistic)
        self.classifier = Classifier(self.featurizer.n_outputs, self.n_classes, is_nonlinear=True)

    @property
    def is_classification(self):
        return True

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


class PACS(ObjBundle):
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
    
class PACSProposal(ObjBundle):
    def __init__(self, dataset, feature_dimension, hparam, probabilistic=False) -> None:
        super().__init__(dataset, feature_dimension, probabilistic)
        self.params = hparam
        self.model = UEModel(self.input_shape, self.n_classes, hparam, self.feature_dimension, probabilistic)
        
    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss())
    
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
    def __init__(self, dataset, probabilistic=False) -> None:
        self.dataset = dataset
        self.probabilistic = probabilistic
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer, self.classifier = code_gpt_py(self.probabilistic)
        if self.probabilistic:
            self.featurizer.init_probablistic()

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


class CivilComments(ObjBundle):
    def __init__(self, dataset, probabilistic=False) -> None:
        self.dataset = dataset
        self.probabilistic = probabilistic
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = DistilBertFeaturizer.from_pretrained('distilbert-base-uncased')
        if self.probabilistic:
            self.featurizer.init_probablistic()
        self.classifier = nn.Linear(self.featurizer.d_out, self.n_classes)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    def _loss(self):
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    @property
    def _domain_fields(self):
        return ['black', 'y']
    
    def transform(self, text):
        if isinstance(text, float):
            print(text)
            text = str(text)
        try: 
            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            )
        except ValueError:
            print(text)
            print(type(text))
        except AssertionError:
            print(text)
            print(type(text))
        else:
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
            x = torch.squeeze(x, dim=0)  # First shape dim is always 1
            return x
    @property
    def _train_transform(self):
        return self.transform
    
    @property
    def _test_transform(self):
        return self.transform
    
    @property
    def _input_shape(self):
        return (255)


class FEMNIST(ObjBundle):
    def __init__(self, dataset, feature_dimension=None, probabilistic=False) -> None:
        self.dataset = dataset
        self.probabilistic = probabilistic
        self.feature_dimension = feature_dimension
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = CNN(self.input_shape, probabilistic=probabilistic)
        self.classifier = Classifier(self.featurizer.n_outputs, dataset.n_classes)


    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (1, 28, 28)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.ToTensor()
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor()
        ])

    @property
    def _domain_fields(self):
        return ['domain',]


class Poverty(ObjBundle):
    @property
    def is_classification(self):
        return False

    @property
    def _train_transform(self):
        return transforms.Compose([])

    @property
    def _test_transform(self):
        return transforms.Compose([])

    def _loss(self):
        return MSE(name='loss')

    @property
    def _input_shape(self):
        return (8, 224, 224)
    
    @property
    def _domain_fields(self):
        return ['country',]

    def _oracle_training_set(self):
        return False

    @property
    def n_classes(self):
        return 1

class OfficeHome(PACS):
    def __init__(self, dataset, feature_dimension, probabilistic=False) -> None:
        super().__init__(dataset, feature_dimension, probabilistic)
