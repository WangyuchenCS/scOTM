import torch
import torch.nn as nn
from torch import Tensor, optim
from tqdm import tqdm
from tqdm.notebook import tqdm
from dataset import AnnDataSet
from torch.utils.data import DataLoader
import scanpy as sc
from torch.distributions import Normal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import ot
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import gc


# from memory_profiler import profile


class scOTM(nn.Module):
    def __init__(self, input_dim=6998, latent_dim=200, hidden_dim=1000, 
                 noise_rate=0.1, kl_weight=5e-3, cycle_weight=0.01, add_weight=50, num_heads=4, loss_type='mmd', mmd_kernel='gaussian', device=None):
        super(scOTM, self).__init__()
        if num_heads >= 1:
            self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.device = device
        self.num_heads = num_heads
        self.cycle_weight = cycle_weight  
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.add_weight = add_weight
        self.loss_type = loss_type  # 'mmd', or 'wasserstein'
        self.mmd_kernel = mmd_kernel  # 'gaussian', 'laplacian', 'linear'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.eps = 1e-8  # Small value to avoid division by zero
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):  
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)  # Generate random noise
        z = mu + eps * std 
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        if self.num_heads >= 1:
            z = z.unsqueeze(1) 
            attn_output, _ = self.attn(z, z, z)
            z = attn_output.squeeze(1) 
        return z, mu, logvar



    def decode(self, z):
        return self.decoder(z)
    

    # def compute_mmd(self,z, prior_z):
    #     """MMD loss between encoded z and prior z ~ N(0, I)"""
    #     def kernel(x, y):
    #         xx = x @ x.T
    #         yy = y @ y.T
    #         xy = x @ y.T
    #         rx = (xx.diag().unsqueeze(0).expand_as(xx))
    #         ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #         K = torch.exp(-0.5 * (rx.T + rx - 2 * xx))
    #         L = torch.exp(-0.5 * (ry.T + ry - 2 * yy))
    #         P = torch.exp(-0.5 * (rx.T + ry - 2 * xy))
    #         return K, L, P
    #     # K, L, P = kernel(z, z), kernel(prior_z, prior_z), kernel(z, prior_z)
    #     # return K[0].mean() + L[0].mean() - 2 * P[0].mean()
    #     K, _, _ = kernel(z, z)
    #     _, L, _ = kernel(prior_z, prior_z)
    #     _, _, P = kernel(z, prior_z)
    #     return K.mean() + L.mean() - 2 * P.mean()

    def compute_mmd(self, z, prior_z):
        def pairwise_distances(x, y):
            x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
            y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
            return x_norm + y_norm - 2.0 * torch.matmul(x, y.T)

        pd_zz = pairwise_distances(z, z)
        pd_pp = pairwise_distances(prior_z, prior_z)
        pd_zp = pairwise_distances(z, prior_z)

        if self.mmd_kernel == 'gaussian':
            K = torch.exp(-0.5 * pd_zz)
            L = torch.exp(-0.5 * pd_pp)
            P = torch.exp(-0.5 * pd_zp)
        elif self.mmd_kernel == 'laplacian':
            K = torch.exp(-pd_zz.sqrt())
            L = torch.exp(-pd_pp.sqrt())
            P = torch.exp(-pd_zp.sqrt())
        elif self.mmd_kernel == 'linear':
            K = z @ z.T
            L = prior_z @ prior_z.T
            P = z @ prior_z.T
        else:
            raise ValueError("Unsupported mmd_kernel. Choose from 'gaussian', 'laplacian', or 'linear'.")

        return K.mean() + L.mean() - 2 * P.mean()
    

    # def softsort(self, x, tau=1.0):
    #     n = x.size(0)
    #     one = torch.ones((n, 1), device=x.device)
    #     A = x.unsqueeze(1) - x.unsqueeze(0)
    #     B = torch.abs(A)
    #     C = torch.matmul(B, one @ one.T)
    #     D = torch.exp(-C / tau)
    #     P = D / D.sum(dim=1, keepdim=True)
    #     return P @ x

    # def compute_wasserstein(self, z, prior_z):
    #     # Use soft sort approximation
    #     z_soft = self.softsort(z)
    #     pz_soft = self.softsort(prior_z)
    #     return F.mse_loss(z_soft, pz_soft)


    # def compute_sinkhorn(self, z, prior_z, epsilon=0.5, n_iter=50):
    #     z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
    #     prior_z = (prior_z - prior_z.mean(dim=0)) / (prior_z.std(dim=0) + 1e-6)

    #     cost_matrix = torch.cdist(z, prior_z, p=2) ** 2  # (B, B)
    #     B = cost_matrix.size(0)
    #     mu = torch.full((B,), 1.0 / B, device=z.device)
    #     nu = torch.full((B,), 1.0 / B, device=z.device)
    #     u = torch.ones_like(mu)
    #     v = torch.ones_like(nu)

    #     K = torch.exp(-cost_matrix / epsilon)
    #     K_plus_eps = K + self.eps

    #     for _ in range(n_iter):
    #         u = mu / (K_plus_eps @ v + self.eps)
    #         v = nu / (K_plus_eps.T @ u + self.eps)

    #     transport_matrix = torch.diag(u) @ K @ torch.diag(v)
    #     sinkhorn_distance = torch.sum(transport_matrix * cost_matrix)
    #     return sinkhorn_distance



    def compute_sinkhorn(self, z, prior_z, epsilon=10.0, n_iter=50):
        # Normalize to control cost magnitude
        z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
        prior_z = (prior_z - prior_z.mean(dim=0)) / (prior_z.std(dim=0) + 1e-6)

        cost_matrix = torch.cdist(z, prior_z, p=2)  # (B, B)
        B = cost_matrix.size(0)
        mu = torch.full((B,), 1.0 / B, device=z.device)
        nu = torch.full((B,), 1.0 / B, device=z.device)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        K = torch.exp(-cost_matrix / epsilon) + self.eps

        for _ in range(n_iter):
            u = mu / (K @ v + self.eps)
            v = nu / (K.T @ u + self.eps)

        transport_matrix = torch.diag(u) @ K @ torch.diag(v)
        sinkhorn_distance = torch.sum(transport_matrix * cost_matrix)
        return sinkhorn_distance



    def compute_wasserstein(self, z, prior_z):
        return self.compute_sinkhorn(z, prior_z)


    # def compute_wasserstein(self, z, prior_z):
    #     z_sorted, _ = torch.sort(z, dim=0)
    #     pz_sorted, _ = torch.sort(prior_z, dim=0)
    #     return F.mse_loss(z_sorted, pz_sorted)


    def forward(self, x, perturbation, fm=None):
        noise = torch.randn_like(x)
        if perturbation is None:
            perturbation = torch.zeros_like(x)
        if fm is not None:
            perturbation = perturbation + fm
        x_noisy = x + perturbation + noise * self.noise_rate
        z, mu, logvar = self.encode(x_noisy)
        x_hat = self.decode(z)

        if self.loss_type == 'mmd':
            prior_z = torch.randn_like(z)
            loss_match = 1000.0 * self.compute_mmd(z, prior_z)
        elif self.loss_type == 'wasserstein':
            prior_z = torch.randn_like(z)
            loss_match = 10.0 * self.compute_wasserstein(z, prior_z)
        else:
            raise ValueError("Unsupported loss_type.")

        loss_rec = ((x - x_hat) ** 2).sum(dim=1)

        if self.cycle_weight != 0: 
            x_hat = x_hat + noise * self.noise_rate  
            z_hat, _, _ = self.encode(x_hat)
            x_cycle = self.decode(z_hat)
            loss_cycle = ((x - x_cycle) ** 2).sum(dim=1)
        else:
            loss_cycle = torch.zeros_like(loss_rec)

        return x_hat, loss_rec, loss_match, loss_cycle, z
    

    def get_latent_adata(self, adata):
        device = self.device
        x = Tensor(adata.to_df().values).to(device) 
        perturbation = Tensor(adata.obsm['embedding']).to(device) if 'embedding' in adata.obsm else None
        x = x + perturbation if perturbation is not None else x
        # except:
        #     x = Tensor(adata).to(device)
        latent_z = self.encode(x)[0].cpu().detach().numpy()
        latent_adata = sc.AnnData(X=latent_z, obs=adata.obs.copy())
        return latent_adata


    def get_loss(self, x, perturbation, fm=None):
        x_hat, loss_rec, addloss, loss_cycle, z = self.forward(x, perturbation, fm)
        return x_hat, loss_rec, addloss, loss_cycle


    def train_scOTM(self, train_adata, epochs=100, batch_size=128, lr=5e-4, weight_decay=1e-5, wandb_run=None):
        device = self.device
        anndataset = AnnDataSet(train_adata)
        train_loader = DataLoader(anndataset, batch_size=batch_size, shuffle=True, drop_last=False)  # batch_size = 128
        scOTC_loss, loss_rec, addloss = 0, 0, 0
        optim_scOTC = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optim_scOTC, step_size=1, gamma=0.99)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description("Training Epoch {}".format(epoch))
            x_ = []
            x_hat_ = []
            loss_ = []
            loss_rec_ = []
            addloss_ = []
            loss_cycle_ = []
            for idx, (x,_, perturbation, fm) in enumerate(train_loader):   # x: torch.Size([128, 6998])
                x = x.to(device)
                perturbation = perturbation.to(device) if perturbation is not None else None
                fm = fm.to(device) if fm is not None else None
                x_hat, loss_rec, addloss, loss_cycle = self.get_loss(x, perturbation, fm)
                scOTC_loss = (0.4 * loss_rec + 0.3 * (self.add_weight * addloss) + self.cycle_weight * loss_cycle).mean()
                optim_scOTC.zero_grad()
                scOTC_loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), 10)
                optim_scOTC.step()
                loss_ += [scOTC_loss.item()]
                loss_rec_ += [loss_rec.mean().item()]
                addloss_ += [addloss.mean().item()]
                loss_cycle_ += [loss_cycle.mean().item()]
                # print(f'loss: {scOTC_loss.item()}')
            if wandb_run:
                wandb_run.log({"scOTC_loss": np.mean(loss_),
                               "recon_loss": np.mean(loss_rec_),
                               "add_loss": np.mean(addloss_),
                               "cycle_loss": np.mean(loss_cycle_)})
            pbar.set_postfix(SCOPA_loss=np.mean(loss_), 
                             recon_loss=np.mean(loss_rec_),
                             add_loss=np.mean(addloss_),
                             cycle_loss=np.mean(loss_cycle_))
            x_.append(x)
            x_hat_.append(x_hat)
            # scheduler.step()
        x_ = torch.cat(x_, dim=0)
        x_hat_ = torch.cat(x_hat_, dim=0)
        torch.cuda.empty_cache()
        gc.collect()
        return x_, x_hat_




    def predict_new(self, train_adata, cell_to_pred, key_dic, ratio=0.05, e=0, r=1):
        ctrl_to_pred = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                    (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]  
        
        
        ctrl_adata = train_adata[(train_adata.obs[key_dic['cell_type_key']] != cell_to_pred) &
                                 (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key'])] 
        stim_adata = train_adata[(train_adata.obs[key_dic['condition_key']] == key_dic['stim_key'])]  

        ctrla = self.get_latent_adata(ctrl_adata) 
        stima = self.get_latent_adata(stim_adata)

        ## Latent Visualization
        adata_combined = ctrla.concatenate(
            stima,
            batch_key="batch",      
            batch_categories=["control", "stimulated"],  
            uns_merge="unique"
        )
        # PCA / UMAP
        sc.pp.neighbors(adata_combined, use_rep='X')
        sc.tl.umap(adata_combined)
        sc.pl.umap(
                adata_combined,
                color=["cell_type", "condition"],
                save="_combined_latent_umap.pdf", 
                frameon=False,
                title=["Latent Space (cell type)", "Latent Space (condition)"], 
                # legend_loc="on data", 
                legend_fontsize=10,
                legend_fontoutline=1,
        )


        ctrl = ctrla.to_df().values
        stim = stima.to_df().values

        M = ot.dist(stim, ctrl, metric='euclidean') 
        G = ot.emd(torch.ones(stim.shape[0]) / stim.shape[0],
                   torch.ones(ctrl.shape[0]) / ctrl.shape[0], 
                   torch.tensor(M),  
                   numItermax=100000) 
        match_idx = torch.max(G, 0)[1].numpy() 
        stim_new = stim[match_idx] #
        delta_list = stim_new - ctrl # 

        mean_ctrl = torch.from_numpy(ctrl).to(self.device).mean(dim=0) 
        mean_stim = torch.from_numpy(stim).to(self.device).mean(dim=0) 
        mse_loss = nn.MSELoss()
        loss = mse_loss(mean_ctrl, mean_stim).item()

        test_za = self.get_latent_adata(ctrl_to_pred)  
        test_z = test_za.to_df().values
        
        cos_sim = cosine_similarity(np.array(test_z).reshape(-1, self.latent_dim),
                                    np.array(ctrl).reshape(-1, self.latent_dim))

        n_top = int(np.ceil(ctrl.shape[0] * ratio))  
        top_indices = np.argsort(cos_sim)[0][-n_top:] 
        normalized_weights = cos_sim[0][top_indices] / np.sum(cos_sim[0][top_indices]) 
        delta_pred = np.sum(normalized_weights[:, np.newaxis] * np.array(delta_list).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + r*delta_pred 
        if e:
            pred_z = pred_z + e*ratio*loss
        pred_x = self.decode(Tensor(pred_z).to(self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(X=pred_x, obs=ctrl_to_pred.obs.copy(), var=ctrl_to_pred.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred_adata,ctrla,stima,test_za


