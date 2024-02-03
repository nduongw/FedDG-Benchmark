import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from style import MixStyle, DistributionUncertainty, ConstantStyle, CorrelatedDistributionUncertainty

class ConstStyleModel(nn.Module):
    def __init__(self, num_style=2):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        self.num_style = num_style
        self.conststyle = [ConstantStyle() for i in range(self.num_style)]
        self.mean = []
        self.std = []
        self.const_mean = None
        self.const_std = None
    
    def forward(self, x, domains, const_style=False, store_style=False, test=False):
        x = self.model.conv1(x)
        # x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if store_style:
            self.conststyle[0].store_style(x, domains)
        if const_style:
            x = self.conststyle[0](x, test=test)
        x = self.model.layer2(x)
        if store_style:
            self.conststyle[1].store_style(x, domains)
        if const_style:
            x = self.conststyle[1](x, test=test)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x
    
class ConstStyleModel2(nn.Module):
    def __init__(self, num_style=2):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        self.num_style = num_style
        self.conststyle = [ConstantStyle() for i in range(self.num_style)]
        self.mean = []
        self.std = []
        self.const_mean = None
        self.const_std = None
    
    def forward(self, x, domains, const_style=False, store_style=False, test=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if store_style:
            self.conststyle[0].store_style(x, domains)
        if const_style:
            x = self.conststyle[0](x, test=test)
        x = self.model.layer2(x)
        if store_style:
            self.conststyle[1].store_style(x, domains)
        if const_style:
            x = self.conststyle[1](x, test=test)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x
    
class MixStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.mixstyle(x)
        x = self.model.layer2(x)
        x = self.mixstyle(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x
    
class DSUModel(nn.Module):
    def __init__(self, pertubration, uncertainty):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        
        self.pertubration0 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration1 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration2 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration3 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration4 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration5 = DistributionUncertainty(p=uncertainty) if pertubration else nn.Identity()
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.pertubration0(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.pertubration1(x)
        x = self.model.layer1(x)
        x = self.pertubration2(x)
        x = self.model.layer2(x)
        x = self.pertubration3(x)
        x = self.model.layer3(x)
        x = self.pertubration4(x)
        x = self.model.layer4(x)
        x = self.pertubration5(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x
    
class CSUModel(nn.Module):
    def __init__(self, csustyle_layers, csustyle_p, csustyle_alpha):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        
        self.pertubration_list = csustyle_layers
        self.pertubration0 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer0' in  csustyle_layers else nn.Identity()
        self.pertubration1 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer1' in  csustyle_layers else nn.Identity()
        self.pertubration2 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer2' in  csustyle_layers else nn.Identity()
        self.pertubration3 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer3' in  csustyle_layers else nn.Identity()
        self.pertubration4 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer4' in  csustyle_layers else nn.Identity()
        self.pertubration5 = CorrelatedDistributionUncertainty(p=csustyle_p, alpha=csustyle_alpha) if 'layer5' in  csustyle_layers else nn.Identity()
        
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.pertubration0(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.pertubration1(x)
        x = self.model.layer1(x)
        x = self.pertubration2(x)
        x = self.model.layer2(x)
        x = self.pertubration3(x)
        x = self.model.layer3(x)
        x = self.pertubration4(x)
        x = self.model.layer4(x)
        x = self.pertubration5(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x