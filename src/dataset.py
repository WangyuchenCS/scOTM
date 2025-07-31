from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
import torch
from torch import nn
import random



class AnnDataSet(Dataset):
    def __init__(self, adata):
        '''
        Build dataset of adata
        :param adata: adata of training or testing set
        '''
        self.data = adata.to_df().values
        try:
            self.cell_type = adata.obs['cell_type']  # PBMC study
        except KeyError:
            try:
                self.cell_type = adata.obs['cell_label']  # Hpoly 
            except KeyError:
                self.cell_type = adata.obs['louvain']  # species 
        # self.D_label = adata.obs['condition'] 
        # celltype to index
        unique_labels = self.cell_type.unique()  
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        # map celltype to index
        self.cell_type = self.cell_type.map(self.label_mapping).values
        
        # adata.uns['use_perturbation_embedding'] = True
        # adata.uns['use_fm_embedding'] = True
        use_fm_embedding = adata.uns.get('use_fm_embedding', False)
        use_perturbation_embedding = adata.uns.get('use_perturbation_embedding', False)
        self.fm_embedding = adata.obsm.get('fm_embedding', None) if use_fm_embedding else None  # embedding of fm embedding
        self.perturbation_embedding = adata.obsm.get('perturbation_embedding', None) if use_perturbation_embedding else None  # embedding of perturbation
        print(f"perturbation_embedding: {self.perturbation_embedding is not None}, LLM_embedding: {self.fm_embedding is not None}")
        # # map condition to index
        # D_label = self.D_label
        # unique_D_labels = D_label.unique()
        # self.D_label_mapping = {label: idx for idx, label in enumerate(unique_D_labels)}
        # self.D_label = self.D_label.map(self.D_label_mapping).values

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.cell_type[index]
        dim = self.data.shape[1]
        # D_label = self.D_label[index]
        # device = x.device
        # perturbation = None if self.perturbation_embedding is None else self.perturbation_embedding[index]
        # fm = None if self.fm_embedding is None else self.fm_embedding[index]
        perturbation = (
            self.perturbation_embedding[index]
            if self.perturbation_embedding is not None
            else torch.zeros(dim, dtype=torch.float32)
        )

        fm = (
            self.fm_embedding[index]
            if self.fm_embedding is not None
            else torch.zeros(dim, dtype=torch.float32)
        )
        return x, y, perturbation, fm



    def __len__(self):
        return self.data.shape[0]


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def preprocessing(adata, fe=None, LLMe=None):
    linear = nn.Linear(fe.shape[0], adata.shape[1])
    P01574_64 = linear(torch.tensor(fe, dtype=torch.float32)).detach()
    embedding = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32) 
    stim_mask = adata.obs["condition"] == "stimulated"
    embedding[stim_mask.values] = P01574_64.numpy()  

    adata.obsm["perturbation_embedding"] = embedding

    linear = nn.Linear(LLMe.shape[1], adata.shape[1])
    fm_embedding = linear(torch.tensor(LLMe, dtype=torch.float32)).detach()
    adata.obsm["fm_embedding"] = fm_embedding.numpy()

    if fe is not None:
        adata.uns['use_fm_embedding'] = True
    else:
        adata.uns['use_fm_embedding'] = False

    if LLMe is not None:
        adata.uns['use_perturbation_embedding'] = True
    else:
        adata.uns['use_perturbation_embedding'] = False

    return adata