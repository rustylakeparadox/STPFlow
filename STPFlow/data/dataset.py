import json
import numpy as np
import scanpy as sc
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.neighbors import NearestNeighbors
from STPFlow.data.normalize_utils import get_normalize_method 


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

def collate_fn(batch):
    """
    batch: list of dict
    """
    batched = {}

    # ===== 节点级数据 =====
    node_keys = ['gene_ids', 'gene_expr', 'protein_expr', 'coords', 'cell_type']

    for k in node_keys:
        if k in batch[0] and batch[0][k] is not None:
            batched[k] = torch.cat([d[k] for d in batch], dim=0)
        else:
            batched[k] = None

    # ===== 处理 edge_index =====
    edge_indices = []
    lr_scores_list = []
    node_offset = 0

    for d in batch:
        edge_index = d['edge_index']
        edge_indices.append(edge_index + node_offset)

        if d.get('lr_scores') is not None:
            lr_scores_list.append(d['lr_scores'])

        node_offset += d['gene_expr'].size(0)

    batched['edge_index'] = torch.cat(edge_indices, dim=1)

    if len(lr_scores_list) > 0:
        batched['lr_scores'] = torch.cat(lr_scores_list, dim=0)
    else:
        batched['lr_scores'] = None

    # ===== 每个 sample 的节点数 =====
    batched['num_nodes_per_sample'] = torch.tensor(
        [d['gene_expr'].size(0) for d in batch]
    )

    return batched

class HESTDatasetPath:
    name: Optional[str] = None
    h5ad_path: Optional[str] = None

    def __init__(self, name, h5ad_path,  **kwargs):
        self.name = name
        self.h5ad_path = h5ad_path

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
                dataset_path,                # HESTDatasetPath 对象
                gene_normalize_method,        # 基因表达归一化方法，如 'log1p', 'stdiff', None
                protein_normalize_method,     # 蛋白质表达归一化方法
                lr_pairs,                     # 配体-受体对列表
                gene2idx,                      # 新参数：基因到ID的映射
                max_genes_per_spot=2000,       # 新参数：每个spot最多保留基因数
                knn_k=10,
                use_image_feat=False,
                cell_type_col=None,
                lr_use_log1p=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.gene_normalize_method = gene_normalize_method
        self.protein_normalize_method = protein_normalize_method
        self.lr_pairs = lr_pairs
        self.gene2idx = gene2idx
        self.max_genes = max_genes_per_spot
        self.knn_k = knn_k
        self.use_image_feat = use_image_feat
        self.cell_type_col = cell_type_col
        self.lr_use_log1p = lr_use_log1p
        
        adata = sc.read_h5ad(dataset_path.h5ad_path, backed='r' if use_image_feat else None)
        
        barcodes = adata.obs_names.astype(str).tolist()   # 形状 (n_spots,)
        
        def parse_coords_from_name(name):
            """从 'humanGBM_33x30' 格式中提取 x, y 坐标"""
            try:
                # 提取最后一个下划线后的部分
                coord_part = name.split('_')[-1]
                x_str, y_str = coord_part.split('x')
                return float(x_str), float(y_str)
            except:
                return None, None
        
        coords_list = []
        for name in adata.obs_names:
            x, y = parse_coords_from_name(name)
            if x is None or y is None:
                raise ValueError(f"Could not parse coordinates from obs_name: {name}")
            coords_list.append([x, y])
        
        coords = np.array(coords_list, dtype=np.float32)  # [N, 2]
        print(f"Coordinates extracted from obs_names, shape: {coords.shape}")
        
        gene_exp_raw = adata.X
        if hasattr(gene_exp_raw, "toarray"):
            gene_exp_raw = gene_exp_raw.toarray()
        else:
            gene_exp_raw = np.asarray(gene_exp_raw)   # (n_spots, n_genes)

        # 基因归一化（注意：原函数 norm_func 可能需要 AnnData，这里我们直接处理 numpy 数组）
        gene_exp_norm = self._normalize_gene(gene_exp_raw, gene_normalize_method)
        
        genes = adata.var_names.astype(str).tolist()
        
        # ---- 5. 提取蛋白质表达（假设在 obsm['protein'] 中）----
        if 'protein' not in adata.obsm:
            raise ValueError("AnnData must have 'protein' in obsm.")
        protein_exp_raw = adata.obsm['protein']
        if hasattr(protein_exp_raw, "toarray"):
            protein_exp_raw = protein_exp_raw.toarray()
        else:
            protein_exp_raw = np.asarray(protein_exp_raw)
        protein_exp = self._normalize_protein(protein_exp_raw, protein_normalize_method)
        protein_names = None
        if 'protein_names' in adata.uns:
            protein_names = adata.uns['protein_names']
        elif 'protein_name' in adata.var:
            protein_names = adata.var['protein_name'].tolist()
        else:
            # 如果没有蛋白质名称，使用默认名称
            n_proteins = protein_exp_raw.shape[1]
            protein_names = [f'protein_{i}' for i in range(n_proteins)]
            print(f"Using default protein names: {protein_names[:5]}...")

        # 5. 细胞类型处理
        if cell_type_col is not None and cell_type_col in adata.obs:
            cell_type_raw = adata.obs[cell_type_col].values
            cell_type_labels, self.cell_type_mapping = self._encode_cell_types(cell_type_raw)
        else:
            cell_type_labels = None

        # 6. 构建 KNN 图（全局边索引）
        nbrs = NearestNeighbors(n_neighbors=knn_k, metric='euclidean').fit(coords)
        indices = nbrs.kneighbors(coords, return_distance=False)  # [N, k]
        edge_list = []
        for i, neighs in enumerate(indices):
            for j in neighs:
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]

        # 7. 计算每条边的 LR 得分
        # 将配体-受体对转换为基因索引       
        protein_names = None
        if hasattr(adata, 'obsm') and 'protein' in adata.obsm:
            # 假设蛋白质名称存储在 adata.uns 或 adata.var 中
            if 'protein_names' in adata.uns:
                protein_names = adata.uns['protein_names']
            else:
                # 如果没有存储蛋白质名称，使用默认列名
                protein_names = [f'protein_{i}' for i in range(protein_exp_raw.shape[1])]
        protein_to_idx = {name: idx for idx, name in enumerate(protein_names)}
        ligand_indices = []
        receptor_indices = []
        print(f"Processing LR pairs...")
        for pair_name, pair_info in self.lr_pairs.items():
            ligands = pair_info.get('ligand', [])
            receptors = pair_info.get('receptor', [])
            
            if isinstance(ligands, str):
                ligands = [ligands]
            if isinstance(receptors, str):
                receptors = [receptors]
            
            # 扩展受体（处理逗号分隔）
        expanded_receptors = []
        for rec in receptors:
            if isinstance(rec, str) and ',' in rec:
                expanded_receptors.extend([r.strip() for r in rec.split(',')])
            else:
                expanded_receptors.append(rec.strip() if isinstance(rec, str) else rec)
        
        # 在蛋白质表达中查找
        for lig in ligands:
            lig = lig.strip()
            if lig not in protein_to_idx:
                print(f"Warning: Ligand {lig} not in protein list, skipping.")
                continue
            
            for rec in expanded_receptors:
                if rec in protein_to_idx:
                    ligand_indices.append(protein_to_idx[lig])
                    receptor_indices.append(protein_to_idx[rec])
                else:
                    print(f"Warning: Receptor {rec} not in protein list, skipping.")
                            
        ligand_indices = torch.tensor(ligand_indices, dtype=torch.long)
        receptor_indices = torch.tensor(receptor_indices, dtype=torch.long)
        # 使用蛋白质表达矩阵
        protein_exp_raw_tensor = torch.from_numpy(protein_exp_raw).float()
        lr_scores = self._compute_lr_scores_for_edges(
            protein_exp_raw_tensor, edge_index, ligand_indices, receptor_indices
        )

    # 构建每个 spot 的 gene_ids 和 gene_expr
    # ---------- 9. 为所有 spot 预计算 gene_ids 和 gene_expr ----------
        n_spots = len(coords)
        gene_ids_list = []
        gene_expr_list = []
        for i in range(n_spots):
            expr = gene_exp_norm[i]   # 归一化后的表达
            non_zero = expr > 0
            genes_in_spot = [genes[j] for j in np.where(non_zero)[0]]
            expr_values = expr[non_zero].astype(np.float32)

            # 转换为 ID
            gene_ids = [self.gene2idx.get(g, self.gene2idx['<UNK>']) for g in genes_in_spot]

            # 按表达量排序，截断/填充
            if len(gene_ids) > self.max_genes:
                idx_sorted = np.argsort(expr_values)[-self.max_genes:]
                gene_ids = [gene_ids[i] for i in idx_sorted]
                expr_values = expr_values[idx_sorted]
            else:
                pad_len = self.max_genes - len(gene_ids)
                gene_ids.extend([0] * pad_len)
                expr_values = np.pad(expr_values, (0, pad_len), constant_values=0)

            gene_ids_list.append(gene_ids)
            gene_expr_list.append(expr_values)

        gene_ids_all = torch.tensor(gene_ids_list, dtype=torch.long)       # (N_spots, max_genes)
        gene_expr_all = torch.tensor(gene_expr_list, dtype=torch.float32) # (N_spots, max_genes)

        self.sp_data = {
        'gene_ids': gene_ids_all,
        'gene_expr': gene_expr_all,
        'protein_expr': torch.from_numpy(protein_exp).float(),  # 改成 protein_expr
        'coords': torch.from_numpy(coords).float(),
        'cell_type': torch.from_numpy(cell_type_labels).long() if cell_type_labels is not None else None,
        'edge_index': edge_index,
        'lr_scores': lr_scores,
        }

        # 可选：去中心化坐标（方便可视化）
        self.sp_data['coords'][:, 0] -= self.sp_data['coords'][:, 0].mean()
        self.sp_data['coords'][:, 1] -= self.sp_data['coords'][:, 1].mean()
        
    def _normalize_gene(self, gene_array, method):
        """基因表达归一化"""
        if method == 'log1p':
            return np.log1p(gene_array)
        elif method == 'scale':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(gene_array)
        elif method == 'none' or method is None:
            return gene_array
        else:
            raise ValueError(f"Unknown gene normalization method: {method}")

    def _normalize_protein(self, protein_array, method):
        """对蛋白质层进行归一化，返回归一化后的 numpy 数组"""
        # 这里简单实现几种方法，可根据需要扩展
        protein = protein_array.copy()
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
    def _compute_lr_scores_for_edges(expr_matrix, edge_index, ligand_indices, receptor_indices):
        """计算给定边的 LR 得分"""
        src, dst = edge_index[0], edge_index[1]
        # expr_matrix: [N, M], src/dst: [E]
        L_src = expr_matrix[src][:, ligand_indices]  # [E, P]
        R_dst = expr_matrix[dst][:, receptor_indices]  # [E, P]
        lr_scores = (L_src * R_dst).sum(dim=-1)   # [E]
        return lr_scores

    def __len__(self):
    # 一个切片就是一个样本
        return 1

    def __getitem__(self, idx):
        # 返回整个切片的数据（因为 len=1，idx 恒为 0）
        return self.sp_data