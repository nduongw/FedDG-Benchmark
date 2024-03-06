import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from style import MixStyle, DistributionUncertainty, ConstantStyle, CorrelatedDistributionUncertainty
import numpy as np
import os

class ConstStyleModel(nn.Module):
    def __init__(self, num_style=6):
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
    def __init__(self, num_style=4):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model
        self.num_style = num_style
        self.conststyle = [ConstantStyle() for i in range(self.num_style)]
        self.mean = []
        self.std = []
        self.const_mean = None
        self.const_std = None
    
    def plot_style(self, args, epoch):
        for idx, style in enumerate(self.conststyle):
            style.plot_style(args, idx, epoch)
    
    def forward(self, x, domains, const_style=False, store_feats=False, sampling=False):
        x = self.model.conv1(x)
        if store_feats:
            self.conststyle[0].store_style(x, domains)
        if const_style:
            x = self.conststyle[0](x, store_feats, sampling=sampling)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if store_feats:
            self.conststyle[1].store_style(x, domains)
        if const_style:
            x = self.conststyle[1](x, store_feats, sampling=sampling)
        x = self.model.layer1(x)
        if store_feats:
            self.conststyle[2].store_style(x, domains)
        if const_style:
            x = self.conststyle[2](x, store_feats, sampling=sampling)
        x = self.model.layer2(x)
        if store_feats:
            self.conststyle[3].store_style(x, domains)
        if const_style:
            x = self.conststyle[3](x, store_feats, sampling=sampling)
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
        self.num_style = 2
    
    def plot_style(self, args, epoch):
        self.mixstyle.plot_style(args, 0, epoch)
    
    def forward(self, x, domains, store_feats=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.mixstyle(x, domains, store_feats=store_feats)
        x = self.model.layer2(x)
        x = self.mixstyle(x, domains)
        x = self.model.layer3(x)
        x = self.mixstyle(x, domains)
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
        self.domain_list = []
        self.scaled_feats = []
    
    def plot_style(self, args, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=42)
        tsne4 = TSNE(n_components=1, random_state=42)
        
        transformed_mean = tsne3.fit_transform(np.squeeze(mu))
        transformed_std = tsne4.fit_transform(np.squeeze(var))
        
        if args.test_domains == 'p':
            classes = ['art', 'cartoon', 'sketch', 'photo']
        elif args.test_domains == 'a':
            classes = ['photo', 'cartoon', 'sketch', 'art']
        elif args.test_domains == 'c':
            classes = ['photo', 'art', 'sketch', 'cartoon']
        elif args.test_domains == 's':
            classes = ['photo', 'art', 'cartoon', 'sketch']
    
        scatter = plt.scatter(transformed_mean[:, 0], transformed_std[:, 0], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
        
    def clear_memory(self):
        self.domain_list = []
        self.scaled_feats = []
        
    def forward(self, x, domains, store_feats=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if store_feats:
            self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
            self.domain_list.extend([i.item() for i in domains])
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
    
    def plot_style(self, args, epoch):
        self.pertubration2.plot_style(args, 1, epoch)
        self.pertubration3.plot_style(args, 2, epoch)
        
    def forward(self, x, domains, store_feats=False):
        x = self.model.conv1(x)
        x = self.pertubration0(x, domains, store_feats=store_feats)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.pertubration1(x, domains, store_feats=store_feats)
        x = self.model.layer1(x)
        x = self.pertubration2(x, domains, store_feats=store_feats)
        x = self.model.layer2(x)
        x = self.pertubration3(x, domains, store_feats=store_feats)
        x = self.model.layer3(x)
        x = self.pertubration4(x, domains, store_feats=store_feats)
        x = self.model.layer4(x)
        x = self.pertubration5(x, domains, store_feats=store_feats)
        
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
    
    def plot_style(self, args, epoch):
        self.pertubration2.plot_style(args, 1, epoch)
        self.pertubration3.plot_style(args, 2, epoch)
        
    def forward(self, x, domains, store_feats=False):
        x = self.model.conv1(x)
        x = self.pertubration0(x, domains, store_style=store_feats)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.pertubration1(x, domains, store_style=store_feats)
        x = self.model.layer1(x)
        x = self.pertubration2(x, domains, store_style=store_feats)
        x = self.model.layer2(x)
        x = self.pertubration3(x, domains, store_style=store_feats)
        x = self.model.layer3(x)
        x = self.pertubration4(x, domains, store_style=store_feats)
        x = self.model.layer4(x)
        x = self.pertubration5(x, domains, store_style=store_feats)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x