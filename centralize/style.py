import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
import scipy.stats as st
from scipy.stats import multivariate_normal
import numpy as np
import copy
import os
import math

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    return -estimator.bic(X)

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.domain_list = []
        self.scaled_feats = []

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix
    
    def clear_memory(self):
        self.domain_list = []
        self.scaled_feats = []

    def forward(self, x, domains, store_feats=False):
        if not self.training or not self._activated:
            if store_feats:
                self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
                self.domain_list.extend([i.item() for i in domains])
            
            return x

        if random.random() > self.p:
            if store_feats:
                self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
                self.domain_list.extend([i.item() for i in domains])
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        out = x_normed*sig_mix + mu_mix
        
        if store_feats:
            self.scaled_feats.extend([i.detach().cpu().numpy() for i in out])
            self.domain_list.extend([i.item() for i in domains])
            # print(f'Total features: {len(self.scaled_feats)} | Total labels: {len(self.domain_list)}')
        return x_normed*sig_mix + mu_mix
    
    def plot_style(self, args, idx, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=args.seed)
        tsne4 = TSNE(n_components=1, random_state=args.seed)
        
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
    
class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.domain_list = []
        self.scaled_feats = []

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def clear_memory(self):
        self.domain_list = []
        self.scaled_feats = []

    def forward(self, x, domains, store_feats=False):
        if (not self.training) or (np.random.random()) > self.p:
            if store_feats:
                self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
                self.domain_list.extend([i.item() for i in domains])
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        if store_feats:
            self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
            self.domain_list.extend([i.item() for i in domains])
        return x

    def plot_style(self, args, idx, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=args.seed)
        tsne4 = TSNE(n_components=1, random_state=args.seed)
        
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
    
class ConstantStyle(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mean = []
        self.std = []
        self.eps = eps
        self.const_mean = None
        self.const_std = None
        self.const_cov = None
        self.domain_list = []
        self.scaled_feats = []
        self.factor = 1.0
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain_list = []
        self.scaled_feats = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = var.sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        
        return mu, var
    
    def store_style(self, x, domains):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain_list.extend([i.item() for i in domains])
    
    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std
    
    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t
    
    def cal_mean_std(self, idx, args, epoch):
        domain_list = np.array(self.domain_list)
        #clustering
        mean_list = copy.copy(self.mean)
        std_list = copy.copy(self.std)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        # pca = PCA(n_components=32)
        # pca_data = pca.fit_transform(reshaped_data)
        pca_data = reshaped_data
        # param_grid = {
        #     "n_components": range(1, 7),
        #     "covariance_type": ["full"],
        # }
        # bayes_cluster = GridSearchCV(
        #     GaussianMixture(init_params='k-means++'), param_grid=param_grid, scoring=gmm_bic_score
        # )

        bayes_cluster = BayesianGaussianMixture(n_components=3, covariance_type='full')
        bayes_cluster.fit(pca_data)
        
        labels = bayes_cluster.predict(pca_data)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        cluster_samples = []
        cluster_samples_idx = []
        for val in unique_labels:
            print(f'Get samples belong to cluster {val}')
            samples = [reshaped_data[i] for i in range(len(labels)) if labels[i] == val]
            samples_idx = [i for i in range(len(labels)) if labels[i] == val]
            samples = np.stack(samples)
            print(f'Cluster {val} has {len(samples)} samples')
            cluster_samples.append(samples)
            cluster_samples_idx.append(samples_idx)
        
        log_likelihood_score = []
        for cluster_idx, cluster_sample_idx in enumerate(cluster_samples_idx):
            cluster_sample = [pca_data[i] for i in cluster_sample_idx]
            sample_score = bayes_cluster.score_samples(cluster_sample)
            mean_score = np.mean(sample_score)
            print(f'Mean log likelihood of cluster {cluster_idx} is {mean_score}')
            log_likelihood_score.append(mean_score)

        idx_val = np.argmax(log_likelihood_score)
        print(f'Layer {idx} chooses cluster {unique_labels[idx_val]} with log likelihood score {log_likelihood_score[idx_val]}')
        
        # import pdb
        # pdb.set_trace()
        self.const_mean = torch.from_numpy(bayes_cluster.means_[idx_val])
        # s_test = np.vstack([reshaped_data[i] for i in range(len(labels)) if labels[i] == idx_val])
        # self.const_mean = torch.from_numpy(np.mean(s_test, axis=0)).double()
        # self.const_cov = torch.from_numpy(np.cov(s_test, rowvar=False)).double()
        self.const_cov = torch.from_numpy(bayes_cluster.covariances_[idx_val])

        #plot features
        if args.test_domains == 'p':
            classes = ['art', 'cartoon', 'sketch']
        elif args.test_domains == 'a':
            classes = ['photo', 'cartoon', 'sketch']
        elif args.test_domains == 'c':
            classes = ['photo', 'art', 'sketch']
        elif args.test_domains == 's':
            classes = ['photo', 'art', 'cartoon']
        
        tsne = TSNE(n_components=2, random_state=args.seed)
        plot_data = tsne.fit_transform(reshaped_data)
        
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'training-features{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
        
        classes = ['c1', 'c2', 'c3']
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=labels)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'cluster{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
        
        if args.wandb:
            args.tracker.log({
                f'Mean_domain_{idx}': torch.mean(self.const_mean).item()
            }, step=epoch)
            
            args.tracker.log({
                f'Std_domain_{idx}': torch.mean(self.const_cov).item()
            }, step=epoch)
    
    def plot_style(self, args, idx, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=args.seed)
        transformed_mean = tsne3.fit_transform(np.squeeze(mu))

        tsne4 = TSNE(n_components=1, random_state=args.seed)
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
        
    def forward(self, x, store_style, sampling=False):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        # print(f'Before applying ConstStyle: Mean: {torch.mean(mu.squeeze(), dim=(0,1))} | Std: {torch.mean(sig.squeeze(), dim=(0,1))}')
        if not self.training:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float()
            const_std = const_value[1].float()
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
        else:
            generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix = self.const_cov)
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])
            
            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()
            
            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')
            
        out = x_normed * const_std + const_mean
        
        # mu = out.mean(dim=[2, 3], keepdim=True)
        # var = out.var(dim=[2, 3], keepdim=True)
        # sig = (var + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        # print(f'After applying ConstStyle: Mean: {torch.mean(mu.squeeze(), dim=(0,1))} | Std: {torch.mean(sig.squeeze(), dim=(0,1))}')
        
        if store_style:
            feats = [i.detach().cpu().numpy() for i in out]
            self.scaled_feats.extend(feats)

        return out
    
class CorrelatedDistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.domain_list = []
        self.scaled_feats = []
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def clear_memory(self):
        self.domain_list = []
        self.scaled_feats = []
        
    def forward(self, x, domains, store_style=False):
        if (not self.training) or (np.random.random()) > self.p:
            if store_style:
                self.scaled_feats.extend([i.detach().cpu().numpy() for i in x])
                self.domain_list.extend([i.item() for i in domains])
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                _, mu_eng_vector = torch.linalg.eigh(C*correlation_mu+self.eps*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)

            try:
                _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)

            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)

        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig
        out = x_normed * sig_mix + mu_mix
        
        if store_style:
            self.scaled_feats.extend([i.detach().cpu().numpy() for i in out])
            self.domain_list.extend([i.item() for i in domains])
        return out 
    
    def plot_style(self, args, idx, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=args.seed)
        tsne4 = TSNE(n_components=1, random_state=args.seed)
        
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()