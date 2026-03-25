import scipy
import scprep
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


def get_normalize_method(gene_method=None, protein_method=None, **kwargs):
    methods = {}
    if gene_method is not None:
        methods['gene'] = _get_single_method(gene_method, **kwargs)
    if protein_method is not None:
        methods['protein'] = _get_single_method(protein_method, **kwargs)
    return methods

def _get_single_method(name, **kwargs):
    methods = {
        "log1p": log1p,
        "stdiff": stdiff_normalize,
        "scVGAE": scVGAE_normalize,
        "scale": scale,
        "identity": identity,
    }
    if name not in methods:
        raise ValueError(f"Unknown normalize method: {name}")
    return methods[name]

def identity(adata):
    return adata.copy()


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data


# https://github.com/fdu-wangfeilab/stDiff/blob/master/test-stDiff.py#L47
def stdiff_normalize(adata):
    process_adata = adata.copy()
    sc.pp.normalize_total(process_adata, target_sum=1e4)
    sc.pp.log1p(process_adata)
    process_adata = scale(process_adata)
    if isinstance(process_adata.X, scipy.sparse.csr_matrix):
        process_adata.X.data = process_adata.X.data * 2 - 1
    else:
        process_adata.X = process_adata.X * 2 - 1
    return process_adata


def data_augment(adata, fixed, noise_std):
    augmented_adata = adata.copy()    
    if fixed: 
        augmented_adata.X = augmented_adata.X + np.full(adata.X.shape, noise_std)
    else:
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_std, adata.X.shape))   
    return adata.concatenate(augmented_adata, join='outer')


def scVGAE_normalize(adata):
    process_adata = adata.copy()
    process_adata.X = scprep.normalize.library_size_normalize(process_adata.X)
    process_adata.X = scprep.transform.sqrt(process_adata.X)
    return process_adata

def protein_log1p(adata, layer='protein'):
    """对蛋白质层进行 log1p 变换"""
    adata_copy = adata.copy()
    if layer in adata_copy.layers:
        data = adata_copy.layers[layer]
        data = np.log1p(data)
        adata_copy.layers[layer] = data
    else:
        # 假设蛋白质在 .X 中？根据实际结构调整
        adata_copy.X = np.log1p(adata_copy.X)
    return adata_copy

def protein_scale(adata, layer='protein', max_abs=True):
    """对蛋白质进行最大绝对值缩放"""
    from sklearn.preprocessing import MaxAbsScaler
    adata_copy = adata.copy()
    if layer in adata_copy.layers:
        scaler = MaxAbsScaler()
        scaled = scaler.fit_transform(adata_copy.layers[layer].T).T
        adata_copy.layers[layer] = scaled
    else:
        adata_copy.X = MaxAbsScaler().fit_transform(adata_copy.X.T).T
    return adata_copy