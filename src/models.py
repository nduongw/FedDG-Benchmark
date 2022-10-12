import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from transformers import GPT2LMHeadModel, GPT2Model
from transformers import GPT2Tokenizer
import copy
from collections import OrderedDict
from torch.nn import init


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class GPT2LMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size

    def __call__(self, x):
        outputs = super().__call__(x)
        logits = outputs[0] #[batch_size, seqlen, vocab_size]
        return logits


class GPT2Featurizer(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.n_embd

    def __call__(self, x):
        outputs = super().__call__(x)
        hidden_states = outputs[0] #[batch_size, seqlen, n_embd]
        return hidden_states


class GPT2FeaturizerLMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size
        self.transformer = GPT2Featurizer(config)

    def __call__(self, x):
        hidden_states = self.transformer(x) #[batch_size, seqlen, n_embd]
        logits = self.lm_head(hidden_states) #[batch_size, seqlen, vocab_size]
        return logits


class GeneDistrNet(nn.Module):
    def __init__(self, num_labels, input_size=1024, hidden_size=2048):
        super(GeneDistrNet,self).__init__()
        self.num_labels = num_labels
        self.latent_size = 4096
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size + self.num_labels, self.latent_size)),
            ("relu1", nn.LeakyReLU()),

            ("fc2", nn.Linear(self.latent_size, hidden_size)),
            ("relu2", nn.ReLU()),
        ]))
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.genedistri(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size=1024):
        super(Discriminator,self).__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        self.optimizer = None
        self.projection = nn.Linear(hidden_size+num_labels, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature = torch.cat([feature, y], dim=1)
        feature = self.projection(feature)
        logit = self.features_pro(feature)
        return logit

def code_gpt_py():
    name = 'microsoft/CodeGPT-small-py'
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
    model.resize_token_embeddings(len(tokenizer))
    featurizer = model.transformer
    classifier = model.lm_head
    model = (featurizer, classifier)
    return model
