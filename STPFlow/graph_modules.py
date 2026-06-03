from scipy.spatial import Delaunay
import numpy as np
import torch
import torch.nn as nn

class FullBatchAugmentMixin:
    def set_augmentation(
        self,
        gene_dropout=0.0,
        gene_noise_std=0.0,
        uce_dropout=0.0,
        uce_branch_drop=0.0,
        edge_dropout=0.0,
    ):
        self.gene_dropout = gene_dropout
        self.gene_noise_std = gene_noise_std
        self.uce_dropout = uce_dropout
        self.uce_branch_drop = uce_branch_drop
        self.edge_dropout = edge_dropout

    def _augment_gene(self, x_gene):
        x = x_gene

        gene_dropout = getattr(self, "gene_dropout", 0.0)
        gene_noise_std = getattr(self, "gene_noise_std", 0.0)

        if gene_dropout > 0:
            mask = (torch.rand_like(x) > gene_dropout).float()
            x = x * mask / (1.0 - gene_dropout)

        if gene_noise_std > 0:
            x = x + gene_noise_std * torch.randn_like(x)

        return x

    def _augment_uce(self, x_uce):
        if x_uce is None:
            return None

        x = x_uce

        uce_dropout = getattr(self, "uce_dropout", 0.0)
        uce_branch_drop = getattr(self, "uce_branch_drop", 0.0)

        if uce_dropout > 0:
            mask = (torch.rand_like(x) > uce_dropout).float()
            x = x * mask / (1.0 - uce_dropout)

        if uce_branch_drop > 0:
            if torch.rand((), device=x.device).item() < uce_branch_drop:
                x = torch.zeros_like(x)

        return x

    def _dropout_edge(self, edge_index):
        edge_dropout = getattr(self, "edge_dropout", 0.0)

        if edge_index is None or edge_dropout <= 0:
            return edge_index

        src, dst = edge_index
        E = edge_index.size(1)

        keep = torch.rand(E, device=edge_index.device) > edge_dropout

        # self-loop 不丢
        self_loop = src == dst
        keep = keep | self_loop

        return edge_index[:, keep]

    def get_full_batch(self, device=None, augment=False):
        """
        Full-batch access for full-graph training.

        augment=True:
            only use for training.
        augment=False:
            use for validation / test.
        """
        x_gene = self.x_gene.float()
        x_protein = self.x_protein.float()

        x_uce = None
        if self.x_uce is not None:
            x_uce = self.x_uce.float()

        edge_index = self.edge_index.long() if self.edge_index is not None else None

        if device is not None:
            x_gene = x_gene.to(device)
            x_protein = x_protein.to(device)

            if x_uce is not None:
                x_uce = x_uce.to(device)

            if edge_index is not None:
                edge_index = edge_index.to(device)

        if augment:
            x_gene = self._augment_gene(x_gene)
            x_uce = self._augment_uce(x_uce)
            edge_index = self._dropout_edge(edge_index)

        return {
            "x_gene": x_gene,
            "x_protein": x_protein,
            "x_uce": x_uce,
            "edge_index": edge_index,
        }

def build_delaunay_edge_index(coords, add_self_loops=True):
    """
    coords: [N, 2]
    return edge_index: [2, E]
    """
    tri = Delaunay(coords)
    simplices = tri.simplices  # [T, 3]

    edge_set = set()

    for simplex in simplices:
        a, b, c = simplex
        pairs = [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]
        for u, v in pairs:
            edge_set.add((int(u), int(v)))

    if add_self_loops:
        for i in range(coords.shape[0]):
            edge_set.add((i, i))

    edge_list = np.array(list(edge_set), dtype=np.int64)
    edge_index = edge_list.T  # [2, E]
    return edge_index

class GraphSubsetDataset(FullBatchAugmentMixin):
    def __init__(self, parent_dataset, indices):
        indices = np.array(sorted(indices), dtype=np.int64)
        self.indices = indices

        self.x_gene = parent_dataset.x_gene[indices]
        self.x_protein = parent_dataset.x_protein[indices]
        self.x_uce = parent_dataset.x_uce[indices] if parent_dataset.x_uce is not None else None
        self.coords = parent_dataset.coords[indices]
        self.gene_names = parent_dataset.gene_names
        self.protein_names = parent_dataset.protein_names

        # 诱导子图：只保留子集内部的边，并重映射节点编号
        old_to_new = {old: new for new, old in enumerate(indices.tolist())}
        ei = parent_dataset.edge_index.numpy()
        src, dst = ei[0], ei[1]

        keep_mask = np.isin(src, indices) & np.isin(dst, indices)
        sub_src = src[keep_mask]
        sub_dst = dst[keep_mask]

        new_src = np.array([old_to_new[x] for x in sub_src], dtype=np.int64)
        new_dst = np.array([old_to_new[x] for x in sub_dst], dtype=np.int64)

        self.edge_index = torch.from_numpy(np.vstack([new_src, new_dst])).long()

        print(f"[INFO] Built subset dataset: n={len(indices)}, edges={self.edge_index.shape[1]}")
        self.set_augmentation(
            gene_dropout=0.0,
            gene_noise_std=0.0,
            uce_dropout=0.0,
            uce_branch_drop=0.0,
            edge_dropout=0.0,
        )
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = {
            "x_gene": self.x_gene[idx],
            "x_protein": self.x_protein[idx],
            "node_idx": torch.tensor(idx, dtype=torch.long),
        }

        if self.x_uce is not None:
            item["x_uce"] = self.x_uce[idx]

        return item

class SimpleGATv2Layer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        """
        h: [N, H]
        edge_index: [2, E]
        """
        src, dst = edge_index

        h_src = h[src]   # [E, H]
        h_dst = h[dst]   # [E, H]

        pair_feat = torch.cat([h_src, h_dst], dim=-1)   # [E, 2H]
        pair_feat = self.leaky_relu(self.W(pair_feat))  # [E, H]
        e = self.attn(pair_feat).squeeze(-1)            # [E]

        # edge softmax over incoming edges of each dst
        e_max = torch.full((h.size(0),), -float("inf"), device=h.device, dtype=h.dtype)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[dst])

        denom = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        denom.scatter_add_(0, dst, e_exp)
        alpha = e_exp / (denom[dst] + 1e-12)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(h)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_src)

        out = self.norm(h + out)
        return out