import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
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

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
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

        return x_normed*sig_mix + mu_mix
    
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

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x
    
class ConstantStyle(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mean = []
        self.std = []
        self.eps = eps
        self.const_mean = None
        self.const_std = None
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
    
    def clustering(self, round):
        mean = torch.vstack(self.mean)
        std = torch.vstack(self.std)
        tsne = TSNE(n_components=1, random_state=42)
        transformed_mean = tsne.fit_transform(mean.detach().cpu().numpy())

        tsne2 = TSNE(n_components=1, random_state=42)
        transformed_std = tsne2.fit_transform(std.detach().cpu().numpy())
        plt.cla()
        plt.clf()
        plt.scatter(transformed_mean[:, 0], transformed_std[:, 0])
        plt.savefig(f'mean_std_round{round}.png')
        
        data = torch.cat((mean, std), dim=1).detach().cpu().numpy()
        # neigh = NearestNeighbors(n_neighbors=2)
        # nbrs = neigh.fit(data)
        # distances, indices = nbrs.kneighbors(data)
        # distances = np.sort(distances, axis=0)
        # distances = distances[:,1]
        # plt.figure(figsize=(20,10))
        # plt.plot(distances)
        dbscan = DBSCAN(eps=5, min_samples=50)
        # dbscan = KMeans(n_clusters=3, n_init=50)
        dbscan.fit(data)
        
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # print(f'Total cluster: {n_clusters}')
        
        sample_each_label = [len(labels[labels == i]) for i in range(n_clusters)]
        largest_cluster = np.argmax(sample_each_label)
        cluster_mean = mean[labels == largest_cluster]
        cluster_std = std[labels == largest_cluster]
        self.const_mean = torch.mean(cluster_mean, axis=0)
        self.const_std = torch.mean(cluster_std, axis=0)
    
    def cal_mean_std(self, idx, domain_id, args, epoch):
        domain_list = np.array(self.domain_list)
        # idx_val = np.where(domain_list == domain_id)[0]
        # cluster_mean = [self.mean[i] for i in idx_val]
        # cluster_std = [self.std[i] for i in idx_val]
        
        #for plotting
        plot_mean = [i.detach().cpu().numpy() for i in self.mean]
        plot_std = [i.detach().cpu().numpy() for i in self.std]
        
        # import pdb
        # pdb.set_trace()
        #GMM clustering
        mean_list = np.array(plot_mean)
        std_list = np.array(plot_std)
        
        data_list = np.concatenate((mean_list, std_list), axis=1)
        
        param_grid = {
            "n_components": 3,
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        
        # grid_search = GridSearchCV(
        #     BayesianGaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        # )
        grid_search = BayesianGaussianMixture(n_components=3, covariance_type='full')
        grid_search.fit(data_list)
        
        labels = grid_search.predict(data_list)
        unique_labels, _ = np.unique(labels, return_counts=True)
        # idx_val = unique_labels[np.argmax(counts)] #idx_val is index of the largest cluster
        
        #get index of cluster which has largest variant
        std_list = []
        for val in unique_labels:
            std_val = torch.stack([self.std[i] for i in labels if i == val])
            std_val = torch.sqrt(torch.mean(std_val ** 2, axis=0))
            std_list.append(sum(std_val).detach().cpu().item())

        idx_val = np.argmax(std_list)
        print(f'Layer {idx} chooses cluster {idx_val}')
        cluster_mean = [self.mean[i] for i in labels if i == idx_val]
        cluster_std = [self.std[i] for i in labels if i == idx_val]
        
        #for plotting
        tsne = TSNE(n_components=1, random_state=42)
        transformed_mean = tsne.fit_transform(np.array(plot_mean))

        tsne2 = TSNE(n_components=1, random_state=42)
        transformed_std = tsne2.fit_transform(np.array(plot_std))
        
        #plot clusters
        plt.close()
        plt.cla()
        plt.clf()
        classes = []
        for val in unique_labels:
            classes.append(f'Cluster {val}')
            
        scatter = plt.scatter(transformed_mean[:, 0], transformed_std[:, 0], c=labels)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'plot_cluster{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        
        #plot features
        plt.cla()
        plt.clf()
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'features{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)

        cluster_mean = torch.stack(cluster_mean)
        cluster_std = torch.stack(cluster_std)
        
        self.const_mean = torch.mean(cluster_mean, axis=0)
        self.const_std = torch.sqrt(torch.mean(cluster_std ** 2, axis=0) / len(cluster_std))
        
        args.tracker.log({
            f'Mean_domain{domain_id}_{idx}': torch.mean(self.const_mean).item()
        }, step=epoch)
        
        args.tracker.log({
            f'Std_domain{domain_id}_{idx}': torch.mean(self.const_std).item()
        }, step=epoch)
    
    def cal_mean_std_ver2(self, idx, domain_id, args, epoch):
        domain_list = np.array(self.domain_list)
        
        plot_mean = [i.detach().cpu().numpy() for i in self.mean]
        plot_std = [i.detach().cpu().numpy() for i in self.std]
        
        mean_list = np.array(plot_mean)
        std_list = np.array(plot_std)
        
        cluster_mean = BayesianGaussianMixture(n_components=3, covariance_type='tied')
        cluster_std = BayesianGaussianMixture(n_components=3, covariance_type='tied', max_iter=500)
        cluster_mean.fit(mean_list)
        cluster_std.fit(std_list)
        
        labels_mean = cluster_mean.predict(mean_list)
        labels_std = cluster_mean.predict(std_list)
        
        unique_labels_mean, _ = np.unique(labels_mean, return_counts=True)
        unique_labels_std, _ = np.unique(labels_std, return_counts=True)
        
        #get index of cluster which has largest variant
        std_list_mean = []
        for val in unique_labels_mean:
            std_mean = torch.stack([self.mean[i] for i in labels_mean if i == val])
            std_mean = torch.sqrt(torch.mean(std_mean ** 2, axis=0))
            std_list_mean.append(sum(std_mean).detach().cpu().item())

        idx_val_mean = np.argmax(std_list_mean)
        print(f'Layer {idx} chooses cluster {idx_val_mean} for mean')
        
        std_list_std = []
        for val in unique_labels_std:
            std_std = torch.stack([self.std[i] for i in labels_std if i == val])
            std_std = torch.sqrt(torch.mean(std_std ** 2, axis=0))
            std_list_std.append(sum(std_std).detach().cpu().item())

        idx_val_std = np.argmax(std_list_std)
        print(f'Layer {idx} chooses cluster {idx_val_std} for std')
        
        cluster_mean = [self.mean[i] for i in labels_mean if i == idx_val_mean]
        cluster_std = [self.std[i] for i in labels_std if i == idx_val_std]
        
        tsne = TSNE(n_components=2, random_state=42)
        transformed_mean = tsne.fit_transform(np.array(plot_mean))

        tsne2 = TSNE(n_components=2, random_state=42)
        transformed_std = tsne2.fit_transform(np.array(plot_std))
        
        #plot clusters
        plt.close()
        plt.cla()
        plt.clf()
        classes_mean = []
        for val in unique_labels_mean:
            classes_mean.append(f'Cluster mean {val}')
            
        scatter = plt.scatter(transformed_mean[:, 0], transformed_mean[:, 1], c=labels_mean)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes_mean)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'cluster_mean{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        
        plt.close()
        plt.cla()
        plt.clf()
        
        classes_std = []
        for val in unique_labels_mean:
            classes_std.append(f'Cluster std {val}')
        
        scatter = plt.scatter(transformed_std[:, 0], transformed_std[:, 1], c=labels_std)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes_std)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'cluster_std{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        
        plt.close()
        plt.cla()
        plt.clf()
        tsne3 = TSNE(n_components=1, random_state=42)
        transformed_mean = tsne3.fit_transform(np.array(plot_mean))

        tsne4 = TSNE(n_components=1, random_state=42)
        transformed_std = tsne4.fit_transform(np.array(plot_std))
        
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
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'features_mean{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        
        # import pdb
        # pdb.set_trace()
        cluster_mean = torch.stack(cluster_mean)
        cluster_std = torch.stack(cluster_std)
        
        self.const_mean = torch.mean(cluster_mean, axis=0)
        self.const_std = torch.mean(cluster_std, axis=0)
        
        args.tracker.log({
            f'Mean_domain{domain_id}_{idx}': torch.mean(self.const_mean).item()
        }, step=epoch)
        
        args.tracker.log({
            f'Std_domain{domain_id}_{idx}': torch.mean(self.const_std).item()
        }, step=epoch)
    
    def plot_data_features(self, args, idx, epoch):
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
    
        scatter = plt.scatter(transformed_mean[:, 0], transformed_std[:, 0], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'results/{args.dataset}/{args.method}_{args.train_domains}_{args.test_domains}_{args.style_idx}', f'style{idx}_features_epoch{epoch}.png')
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
        
        if sampling:
            #create generator to get value from normal distribution 
            gaussian_generator = torch.distributions.Normal(loc=self.const_mean, scale=self.const_std)
            sample = gaussian_generator.sample()
            sample = torch.reshape(sample, (1, sample.shape[0], 1, 1))
            mu = out.mean(dim=[2, 3], keepdim=True)
            # import pdb
            # pdb.set_trace()
            diff = sample - mu
            out = self.reparameterize(out, diff)  
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
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
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

        return x_normed * sig_mix + mu_mix