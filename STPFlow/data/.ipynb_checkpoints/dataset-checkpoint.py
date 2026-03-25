import os
import json
import numpy as np
from typing import List
from pathlib import Path
import scanpy as sc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from STPFlow.data.sampling_utils import PatchSampler
from sklearn.neighbors import NearestNeighbors
from STPFlow.data.normalize_utils import get_normalize_method 
from STPFlow.hest_utils.file_utils import read_assets_from_h5


def load_adata(expr_path, genes=None, barcodes=None, normalize_method=None):
    """
    Load AnnData from h5ad file, filter by barcodes and genes, apply normalization.
    Returns AnnData object (not DataFrame).
    
    Args:
        expr_path: path to .h5ad file
        genes: list of gene names to keep
        barcodes: list of barcodes to keep
        normalize_method: callable that takes an AnnData and returns normalized AnnData
                         (e.g., from get_normalize_method in normalize_utils)
    Returns:
        AnnData object (filtered and normalized)
    """
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize_method is not None:
        adata = normalize_method(adata)
    return adata
    
class HESTDatasetPath:
    name: str | None = None
    h5_path: str | None = None
    h5ad_path: str | None = None
    gene_list_path: str | None = None

    def __init__(self, name, h5_path, h5ad_path, gene_list_path, **kwargs):
        self.name = name
        self.h5_path = h5_path
        self.h5ad_path = h5ad_path
        self.gene_list_path = gene_list_path

        for k, v in kwargs.items():
            setattr(self, k, v)


class SPData:
    def __init__(self, gene_exp, protein_exp, coords, cell_type=None, features=None):
        self.gene_exp = gene_exp          # [N, n_genes]
        self.protein_exp = protein_exp    # [N, n_proteins]
        self.coords = coords               # [N, 2]
        self.cell_type = cell_type         # [N] 或 [N, type_dim]
        self.features = features           # [N, feat_dim] 可选

        # 可选去中心化
        self.coords[:, 0] -= self.coords[:, 0].mean()
        self.coords[:, 1] -= self.coords[:, 1].mean()

    def __len__(self):
        return len(self.gene_exp)

    def chunk(self, index):
        return SPData(
            gene_exp=self.gene_exp[index],
            protein_exp=self.protein_exp[index],
            coords=self.coords[index],
            cell_type=self.cell_type[index] if self.cell_type is not None else None,
            features=self.features[index] if self.features is not None else None
        )

class HESTDataset(Dataset):
    def __init__(self, 
                 dataset_path,            # HESTDatasetPath 对象，包含 .h5, .h5ad, gene_list.json 路径
                 gene_normalize_method,   # 基因表达归一化方法，如 'log1p', 'stdiff', None
                 protein_normalize_method, # 蛋白质表达归一化方法，如 'log1p', 'scale', None
                 lr_pairs,                # 配体-受体对列表，每个元素为 (ligand_gene, receptor_gene)
                 knn_k=10,                 # KNN 的邻居数
                 use_image_feat=False,     # 是否使用图像特征（embeddings）
                 cell_type_col=None,       # adata.obs 中细胞类型列的名称，若为 None 则不使用细胞类型
                 lr_use_log1p=False):      # LR 得分是否基于 log1p 变换后的表达（默认 False，使用原始计数）
        super().__init__()

        # 1. 读取基础数据（坐标、图像特征等）
        data_dict, _ = read_assets_from_h5(dataset_path.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = data_dict["coords"]                          # [N, 2]
        embeddings = data_dict.get("embeddings", None)        # 可能不存在

        # 2. 读取基因列表
        with open(dataset_path.gene_list_path, 'r') as f:
            genes = json.load(f)['genes']                     # 基因名称列表

        # 3. 加载 AnnData 对象（包含基因表达和蛋白质层）
        adata = load_adata(dataset_path.h5ad_path, genes=genes, barcodes=barcodes)

        # 4. 归一化处理
        # 基因表达归一化（注意：LR 得分可能需要未归一化的原始值，因此我们保留原始值用于 LR 计算）
        gene_exp_raw = adata.X.copy()  # 保存原始计数用于 LR 计算
        if gene_normalize_method is not None:
            norm_func = get_normalize_method(gene_normalize_method)
            adata = norm_func(adata)   # 假设函数直接修改 adata.X
        gene_exp_norm = adata.X        # 归一化后的基因表达

        # 蛋白质表达归一化
        if 'protein' in adata.layers:
            protein_exp_raw = adata.layers['protein'].copy()
            if protein_normalize_method is not None:
                # 这里需要确保归一化函数可以处理蛋白质层，或者我们自己实现简单的归一化
                # 为简化，我们假设 protein_normalize_method 是字符串，使用对应函数
                # 可以定义一个辅助函数 normalize_protein(adata, method)
                protein_exp = self._normalize_protein(adata, protein_normalize_method)
            else:
                protein_exp = protein_exp_raw
        else:
            raise ValueError("AnnData object must have a 'protein' layer.")

        # 5. 细胞类型处理
        if cell_type_col is not None and cell_type_col in adata.obs:
            cell_type_raw = adata.obs[cell_type_col].values
            # 将细胞类型字符串编码为整数（后续模型会嵌入）
            self.cell_type_labels, self.cell_type_mapping = self._encode_cell_types(cell_type_raw)
        else:
            self.cell_type_labels = None

        # 6. 构建 KNN 图（全局边索引）
        nbrs = NearestNeighbors(n_neighbors=knn_k, metric='euclidean').fit(coords)
        distances, indices = nbrs.kneighbors(coords)  # [N, k]
        # 构建边索引 [2, E]
        edge_list = []
        for i, neighs in enumerate(indices):
            for j in neighs:
                if i != j:          # 排除自环
                    edge_list.append([i, j])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]

        # 7. 计算每条边的 LR 得分
        # 将配体-受体对转换为基因索引
        # 根据物种加载对应的配受体文件
        if species == 'human':
            lr_pairs_file = os.path.join(lr_pairs_dir, 'cellchat_lr_pairs_human.json')
        elif species == 'mouse':
            lr_pairs_file = os.path.join(lr_pairs_dir, 'cellchat_lr_pairs_mouse.json')
        else:
            raise ValueError(f"Unknown species: {species}")
        
        with open(lr_pairs_file, 'r') as f:
            lr_pairs = json.load(f)  # 列表 of [ligand, receptor] 或 dict

        self.lr_pairs = lr_pairs
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        ligand_indices = []
        receptor_indices = []
        for lig, rec in lr_pairs:
            if lig in gene_to_idx and rec in gene_to_idx:
                ligand_indices.append(gene_to_idx[lig])
                receptor_indices.append(gene_to_idx[rec])
            else:
                print(f"Warning: {lig} or {rec} not in gene list, skipping.")
        if len(ligand_indices) == 0:
            raise ValueError("No valid LR pairs found in gene list.")

        ligand_indices = torch.tensor(ligand_indices, dtype=torch.long)
        receptor_indices = torch.tensor(receptor_indices, dtype=torch.long)

        # 计算 LR 得分（基于原始表达或归一化表达？建议基于原始表达）
        gene_exp_for_lr = torch.from_numpy(gene_exp_raw).float()  # [N, G]
        self.lr_scores = self._compute_lr_scores_for_edges(
            gene_exp_for_lr, self.edge_index, ligand_indices, receptor_indices
        )  # [E]

        # 8. 存储为 SPData 对象（简化结构，用 tensor 存储）
        self.sp_data = {
            'gene_exp': torch.from_numpy(gene_exp_norm).float(),
            'protein_exp': torch.from_numpy(protein_exp).float(),
            'coords': torch.from_numpy(coords).float(),
            'cell_type': torch.from_numpy(self.cell_type_labels).long() if self.cell_type_labels is not None else None,
            'features': torch.from_numpy(embeddings).float() if use_image_feat and embeddings is not None else None
        }

        # 可选：去中心化坐标
        self.sp_data['coords'][:, 0] -= self.sp_data['coords'][:, 0].mean()
        self.sp_data['coords'][:, 1] -= self.sp_data['coords'][:, 1].mean()

    def _normalize_protein(self, adata, method):
        """对蛋白质层进行归一化，返回归一化后的 numpy 数组"""
        # 这里简单实现几种方法，可根据需要扩展
        protein = adata.layers['protein'].copy()
        if method == 'log1p':
            protein = np.log1p(protein)
        elif method == 'scale':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            protein = scaler.fit_transform(protein.T).T
        elif method == 'none' or method is None:
            pass
        else:
            raise ValueError(f"Unknown protein normalization method: {method}")
        return protein

    def _encode_cell_types(self, cell_type_raw):
        """将细胞类型字符串编码为整数标签"""
        unique_types = np.unique(cell_type_raw)
        mapping = {typ: i for i, typ in enumerate(unique_types)}
        labels = np.array([mapping[typ] for typ in cell_type_raw], dtype=np.int64)
        return labels, mapping

    @staticmethod
    def _compute_lr_scores_for_edges(gene_exp, edge_index, ligand_indices, receptor_indices):
        """计算给定边的 LR 得分"""
        src, dst = edge_index[0], edge_index[1]
        # gene_exp: [N, G], src/dst: [E]
        L_src = gene_exp[src][:, ligand_indices]  # [E, P]
        R_dst = gene_exp[dst][:, receptor_indices]  # [E, P]
        lr_scores = (L_src * R_dst).sum(dim=-1)   # [E]
        return lr_scores

    def __len__(self):
        # 如果采用全切片训练，一个样本就是一个切片，所以长度为1
        return 1

    def __getitem__(self, idx):
        # 返回整个切片的数据（字典形式）
        item = {k: v for k, v in self.sp_data.items() if v is not None}
        # 加入图信息
        item['edge_index'] = self.edge_index
        item['lr_scores'] = self.lr_scores
        return item

class MultiHESTDataset(Dataset):
    def __init__(self, dataset_list: List[HESTDatasetPath], normalize_method, distribution="beta_3_1", sample_times=5):
        super().__init__()

        self.dataset_list = dataset_list
        self.sp_datasets = []
        self.n_chunks, self.sample_times = [], sample_times
        self.patch_sampler = PatchSampler(distribution)

        for i, dataset in enumerate(self.dataset_list):
            data_dict, _ = read_assets_from_h5(dataset.h5_path)
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = data_dict["coords"]
            embeddings = data_dict["embeddings"]

            with open(os.path.join(dataset.gene_list_path), 'r') as f:
                genes = json.load(f)['genes']

            labels = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
            labels = labels.values

            self.n_chunks.append(sample_times)

            self.sp_datasets.append(
                SPData(
                    features=torch.from_numpy(embeddings).float(),
                    labels=torch.from_numpy(labels).float(),
                    coords=torch.from_numpy(coords).float()
                )
            )
        
    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
    chunk = self.sp_data.chunk(self.patch_sampler(self.sp_data.coords))
    return {
        'gene_exp': chunk.gene_exp,
        'protein_exp': chunk.protein_exp,
        'coords': chunk.coords,
        'cell_type': chunk.cell_type,
        'features': chunk.features,  # 可选
    }


def collate_fn(batch):
    """
    batch: list of dict, each dict contains:
        'gene_exp': [N_i, G]
        'protein_exp': [N_i, P]
        'coords': [N_i, 2]
        'cell_type': [N_i] (optional)
        'features': [N_i, F] (optional)
        'edge_index': [2, E_i]
        'lr_scores': [E_i]
    Returns:
        A dict with batched tensors.
    """
    keys = batch[0].keys()
    batched = {}
    # 拼接节点级数据
    node_keys = ['gene_exp', 'protein_exp', 'coords', 'cell_type', 'features']
    for k in node_keys:
        if k in batch[0]:
            batched[k] = torch.cat([d[k] for d in batch], dim=0)

    # 处理边索引：需要添加节点偏移
    edge_indices = []
    lr_scores_list = []
    node_offset = 0
    for d in batch:
        edge_index = d['edge_index']  # [2, E]
        edge_indices.append(edge_index + node_offset)
        lr_scores_list.append(d['lr_scores'])
        node_offset += d['gene_exp'].size(0)
    batched['edge_index'] = torch.cat(edge_indices, dim=1)  # [2, total_E]
    batched['lr_scores'] = torch.cat(lr_scores_list, dim=0)  # [total_E]

    # 可选：记录每个样本的节点数，用于后续反标准化或分割
    batched['num_nodes_per_sample'] = torch.tensor([d['gene_exp'].size(0) for d in batch])

    return batched

