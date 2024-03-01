import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
import scipy.stats as st
import numpy as np
import os

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
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
        self.const_mean_std = None
        self.const_std_std = None
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
        mu, var = mu.detach().squeeze(), var.detach().squeeze()
        
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
    
    def cal_mean_std(self, idx, domain_id, args, epoch):
        domain_list = np.array(self.domain_list)
        
        #for plotting
        plot_mean = [i.detach().cpu().numpy() for i in self.mean]
        plot_std = [i.detach().cpu().numpy() for i in self.std]
        tsne = TSNE(n_components=1, random_state=42)
        transformed_mean = tsne.fit_transform(np.array(plot_mean))
        tsne2 = TSNE(n_components=1, random_state=42)
        transformed_std = tsne2.fit_transform(np.array(plot_std))    
        
        #GMM clustering
        mean_list = np.array(plot_mean)
        std_list = np.array(plot_std)
        stacked_data = np.vstack((mean_list, std_list))
        reshaped_data = stacked_data.reshape((len(plot_mean), 2, -1))

        mean = np.mean(reshaped_data, axis=(1, 2))
        std = np.std(reshaped_data, axis=(1, 2))
        data_list = np.stack((mean, std), axis=1)

        bayes_cluster = BayesianGaussianMixture(n_components=3, covariance_type='full')
        bayes_cluster.fit(data_list)
        
        labels = bayes_cluster.predict(data_list)
        unique_labels, _ = np.unique(labels, return_counts=True)
        
        #get index of cluster which has largest variant
        std_list = []
        for val in unique_labels:
            std_val = torch.stack([self.std[i] for i in labels if i == val])
            std_val = torch.sqrt(torch.mean(std_val ** 2, axis=0))
            std_list.append(sum(std_val).detach().cpu().item())

        idx_val = np.argmax(std_list)
        print(f'Layer {idx} chooses cluster {unique_labels[idx_val]}')
        cluster_mean = [self.mean[i] for i in labels if i == unique_labels[idx_val]]
        cluster_std = [self.std[i] for i in labels if i == unique_labels[idx_val]]
        
        #plot features
        if args.test_domains == 'p':
            classes = ['art', 'cartoon', 'sketch']
        elif args.test_domains == 'a':
            classes = ['photo', 'cartoon', 'sketch']
        elif args.test_domains == 'c':
            classes = ['photo', 'art', 'sketch']
        elif args.test_domains == 's':
            classes = ['photo', 'art', 'cartoon']
        scatter = plt.scatter(transformed_mean[:, 0], transformed_std[:, 0], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'features{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
        # import pdb;pdb.set_trace()
        cluster_mean = torch.stack(cluster_mean)
        cluster_std = torch.stack(cluster_std)
        cluster_samples = torch.stack((cluster_mean, cluster_std), axis=1)
        self.const_mean = torch.mean(cluster_mean, axis=0)
        self.const_std = torch.mean(cluster_std, axis=0)
        
        if args.wandb:
            args.tracker.log({
                f'Mean_domain{domain_id}_{idx}': torch.mean(self.const_mean).item()
            }, step=epoch)
            
            args.tracker.log({
                f'Std_domain{domain_id}_{idx}': torch.mean(self.const_std).item()
            }, step=epoch)
    
    def plot_style(self, args, idx, epoch):
        domain_list = np.array(self.domain_list)
        scaled_feats = np.array(self.scaled_feats)
        
        mu = scaled_feats.mean(axis=(2, 3), keepdims=True)
        var = scaled_feats.var(axis=(2, 3), keepdims=True)
        var = np.sqrt(var)

        tsne3 = TSNE(n_components=1, random_state=42)
        transformed_mean = tsne3.fit_transform(np.squeeze(mu))

        tsne4 = TSNE(n_components=1, random_state=42)
        transformed_std = tsne4.fit_transform(np.squeeze(var))
        
        if args.test_domains == 'p':
            classes = ['art', 'cartoon', 'sketch', 'photo']
        elif args.test_domains == 'a':
            classes = ['photo', 'cartoon', 'sketch', 'art']
        elif args.test_domains == 'c':
            classes = ['photo', 'art', 'sketch', 'cartoon']
        elif args.test_domains == 's':
            classes = ['photo', 'art', 'cartoon', 'sketch']
        
        #kde plot
        kde_data = np.hstack((transformed_mean, transformed_std))
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(kde_data)
        
        # Generate grid points for visualization
        x_min, x_max = kde_data[:, 0].min() - 1, kde_data[:, 0].max() + 1
        y_min, y_max = kde_data[:, 1].min() - 1, kde_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        z = np.exp(kde.score_samples(grid_points))
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.viridis)
        plt.scatter(kde_data[:, 0], kde_data[:, 1], s=5, color='black', alpha=0.5, c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.colorbar(label='Density')
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_kde_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
    
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
        const_mean = torch.reshape(self.const_mean, (1, self.const_mean.shape[0], 1, 1))
        const_std = torch.reshape(self.const_std, (1, self.const_std.shape[0], 1, 1))
    
        out = x_normed * const_std + const_mean
        
        # if sampling:
        #     generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix = self.cov_mat)
        #     #create generator to get value from normal distribution 
        #     # gaussian_generator = torch.distributions.Normal(loc=self.const_mean, scale=self.const_std)
        #     sample_list = []
        #     gen_mean, gen_var = [], []
        #     import pdb;pdb.set_trace()
        #     for i in range(x.shape[0]):
        #         mean, var = generator.sample()
        #         gen_mean.append(mean)
        #         gen_var.append(var)
        #         sample = torch.reshape(sample, (1, sample.shape[0], 1, 1))
        #     gen_mean = torch.vstack(gen_mean)
        #     gen_var = torch.vstack(gen_var)
        #     out = x_normed * gen_var + gen_mean
            # sample_list.append(sample)
            # sample_list = torch.vstack(sample_list)
        # if sampling:
        #     out = x_normed * aug_sig + aug_mu
        
        if store_style:
            self.scaled_feats.extend([i.detach().cpu().numpy() for i in out])

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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.option}', f'style{idx}_features_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()