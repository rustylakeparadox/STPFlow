import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_attention import SpatialAttentionLayer 
from .config import ModelConfig 


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def get_batched_edge_index(edge_index, batch_size, num_nodes_per_graph):
    """
    将单图的边索引扩展为包含 batch 偏移的边索引。

    Args:
        edge_index (torch.Tensor): 原始边索引，形状 [2, E] (0-indexed, 单图)
        batch_size (int): 当前 batch 中的图数量 B
        num_nodes_per_graph (int): 每个图的节点数 N（假设所有图相同）

    Returns:
        batched_edge_index (torch.Tensor): 形状 [2, B * E]，适用于拼接后的节点特征
    """
    device = edge_index.device
    E = edge_index.size(1)

    # 1. 将边索引复制 batch_size 次，得到 [2, B * E]
    batched_edge_index = edge_index.repeat(1, batch_size)

    # 2. 计算每个 batch 的偏移量 offsets = [0, N, 2N, ..., (B-1)*N]
    offsets = torch.arange(batch_size, device=device) * num_nodes_per_graph

    # 3. 为每条边分配对应的偏移量：offsets 中每个元素重复 E 次
    #    shape: [B * E]
    edge_offsets = offsets.repeat_interleave(E)

    # 4. 将偏移量加到两个行上
    batched_edge_index[0] += edge_offsets
    batched_edge_index[1] += edge_offsets

    return batched_edge_index

class GeneEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, 
                 pool_type='mean', unknown_strategy='zero'):
        """
        Args:
            vocab_size: 基因词汇表大小（包含PAD和UNK）
            embed_dim: 基因嵌入维度
            hidden_dim: 输出的spot特征维度
            max_genes: 每个spot的最大基因数（用于padding）
            pool_type: 聚合方式，'mean' 或 'attention'
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pool_type = pool_type
        self.unknown_strategy = unknown_strategy

        if pool_type == 'mean':
            # 平均池化后接一个MLP
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif pool_type == 'attention':
            # 注意力池化
            self.attn = nn.Linear(embed_dim, 1)
            # 注意力池化后也需要一个MLP（输出维度为hidden_dim）
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

    def forward(self, gene_ids, gene_expr):
        """
        Args:
            gene_ids: (batch, n_spots, n_genes_per_spot) 整数张量，0表示padding
            gene_expr: (batch, n_spots, n_genes_per_spot) 浮点张量，表达量
        Returns:
            spot_features: (batch, n_spots, hidden_dim)
        """
        # 支持两种输入形状：
        # 1) (B, N, L) （带 batch 维度） -> 嵌入后 (B, N, L, D_emb)
        # 2) (N, L) （无 batch，通常为将所有节点拼接在一起） -> 嵌入后 (N, L, D_emb)
        gene_emb = self.embedding(gene_ids)

        if gene_emb.dim() == 4:
            # (B, N, L, D_emb)
            weighted_emb = gene_emb * gene_expr.unsqueeze(-1)
            mask = (gene_ids != 0).float().unsqueeze(-1)

            if self.pool_type == 'mean':
                sum_emb = (weighted_emb * mask).sum(dim=2)          # (B, N, D_emb)
                valid_count = mask.sum(dim=2).clamp(min=1)          # (B, N, 1)
                pooled = sum_emb / valid_count                       # (B, N, D_emb)
            else:
                attn_scores = self.attn(weighted_emb).squeeze(-1)    # (B, N, L)
                attn_scores = attn_scores.masked_fill(mask.squeeze(-1) == 0, -1e9)
                attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, N, L, 1)
                pooled = (weighted_emb * attn_weights).sum(dim=2)    # (B, N, D_emb)
        elif gene_emb.dim() == 3:
            # (N, L, D_emb)
            weighted_emb = gene_emb * gene_expr.unsqueeze(-1)
            mask = (gene_ids != 0).float().unsqueeze(-1)

            if self.pool_type == 'mean':
                # sum over genes dim=1
                sum_emb = (weighted_emb * mask).sum(dim=1)          # (N, D_emb)
                valid_count = mask.sum(dim=1).clamp(min=1)          # (N, 1)
                pooled = sum_emb / valid_count                       # (N, D_emb)
            else:
                attn_scores = self.attn(weighted_emb).squeeze(-1)    # (N, L)
                attn_scores = attn_scores.masked_fill(mask.squeeze(-1) == 0, -1e9)
                attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (N, L, 1)
                pooled = (weighted_emb * attn_weights).sum(dim=1)    # (N, D_emb)
        else:
            raise ValueError(f"Unsupported gene_emb dim: {gene_emb.dim()}")

        # 通过聚合网络映射到hidden_dim
        spot_features = self.aggregator(pooled)
        return spot_features
    
class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # ===== Gene encoder =====
        if config.use_gene_embedding:
            self.gene_encoder = GeneEncoder(
                vocab_size=config.n_genes,
                embed_dim=config.gene_embed_dim,
                hidden_dim=config.hidden_dim,
                pool_type=config.pool_type,
                unknown_strategy=config.unknown_gene_strategy
            )
        else:
            self.gene_encoder = nn.Linear(config.n_genes, config.hidden_dim)

        # ===== Protein encoder =====
        self.protein_encoder = nn.Linear(config.n_proteins, config.hidden_dim)

        # ===== Cell type encoder =====
        if config.n_cell_types > 0:
            self.type_encoder = nn.Embedding(config.n_cell_types, config.hidden_dim)
        else:
            self.type_encoder = nn.Linear(config.type_dim, config.hidden_dim)

        # ===== Time embedding =====
        self.time_embedder = TimestepEmbedder(config.hidden_dim)

        # ===== Spatial attention =====
        self.spatial_attn = SpatialAttentionLayer(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_lr_prior=config.use_lr_prior,
            lr_lambda=config.lr_lambda
        )

        # ===== Output head =====
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_proteins)
        )

        self.loss_func = nn.MSELoss()

    def forward(
        self,
        p_t,
        t,
        gene_ids,
        gene_expr,
        cell_type,
        coords,
        edge_index,
        lr_mat=None,
        labels=None,
        valid_mask=None
    ):
        # ===== Gene feature =====
        if self.config.use_gene_embedding:
            gene_feat = self.gene_encoder(gene_ids, gene_expr)  # (B, N, H) or (N, H)
        else:
            gene_feat = self.gene_encoder(gene_expr)            # (B, N, H) or (N, H)

        # 如果 gene_encoder 返回的是 (N, H)（没有 batch 维度），将其包装为 (1, N, H)
        if gene_feat.dim() == 2:
            gene_feat = gene_feat.unsqueeze(0)
            # 对齐 p_t / cell_type / labels 的形状（将节点维作为 batch 中唯一图的节点）
            if p_t is not None and p_t.dim() == 2:
                p_t = p_t.unsqueeze(0)
            if cell_type is not None and cell_type.dim() == 1:
                cell_type = cell_type.unsqueeze(0)
            if labels is not None and labels.dim() == 2:
                labels = labels.unsqueeze(0)
            if valid_mask is not None and valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)

        B, N, H = gene_feat.shape

        # ===== Protein feature =====
        if p_t.dim() == 2:
            p_t = p_t.unsqueeze(1).expand(-1, N, -1)
        prot_feat = self.protein_encoder(p_t)

        # ===== Cell type feature =====
        # If no cell type information provided, create zeros with correct shape
        if cell_type is None:
            if self.config.n_cell_types > 0:
                # use integer indices (all zeros -> index 0)
                cell_type = torch.zeros((B, N), dtype=torch.long, device=gene_feat.device)
            else:
                # use continuous type features (zeros)
                cell_type = torch.zeros((B, N, self.config.type_dim), dtype=torch.float, device=gene_feat.device)

        if cell_type.dim() == 1:
            cell_type = cell_type.unsqueeze(1).expand(-1, N)

        if cell_type.dtype in [torch.long, torch.int]:
            type_feat = self.type_encoder(cell_type)
        else:
            type_feat = self.type_encoder(cell_type.float())

        # ===== Time embedding =====
        t_emb = self.time_embedder(t)            # (B, H)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)

        # ===== Fusion =====
        h = gene_feat + prot_feat + type_feat + t_emb

        # ===== Graph attention =====
        h_flat = h.reshape(B * N, H)
        edge_index_batch = get_batched_edge_index(edge_index, B, N)

        h_attn = self.spatial_attn(
            h_flat,
            edge_index_batch,
            lr_mat,
            N
        )

        h_attn = h_attn.view(B, N, H)

        # ===== Output =====
        v_pred = self.output_head(h_attn)

        # ===== Loss =====
        loss = None
        if labels is not None:
            if valid_mask is not None:
                v_pred_masked = v_pred[valid_mask]
                labels_masked = labels[valid_mask]
                loss = self.loss_func(v_pred_masked, labels_masked)
            else:
                loss = self.loss_func(v_pred, labels)

        return v_pred, loss

    @torch.no_grad()
    def inference(self, p_t, t, gene_ids, gene_expr, cell_type, coords, edge_index, lr_mat=None):
        """
        Inference wrapper for sampling/evaluation. Returns v_pred shaped to match inputs.
        Accepts p_t as [N, P] or [B, N, P]; returns v_pred as [N, P] when B==1 or [B, N, P].
        """
        # Move inputs to model device
        device = next(self.parameters()).device
        if torch.is_tensor(p_t):
            p_t = p_t.to(device)
        if torch.is_tensor(t):
            t = t.to(device)
        if torch.is_tensor(gene_ids):
            gene_ids = gene_ids.to(device)
        if torch.is_tensor(gene_expr):
            gene_expr = gene_expr.to(device)
        if torch.is_tensor(cell_type):
            cell_type = cell_type.to(device)
        if torch.is_tensor(coords):
            coords = coords.to(device)
        if torch.is_tensor(edge_index):
            edge_index = edge_index.to(device)
        if torch.is_tensor(lr_mat):
            lr_mat = lr_mat.to(device)

        v_pred, _ = self.forward(
            p_t=p_t,
            t=t,
            gene_ids=gene_ids,
            gene_expr=gene_expr,
            cell_type=cell_type,
            coords=coords,
            edge_index=edge_index,
            lr_mat=lr_mat,
            labels=None,
            valid_mask=None
        )

        # If batch dimension is 1, squeeze to [N, P] for compatibility with test loop
        if v_pred.dim() == 3 and v_pred.size(0) == 1:
            return v_pred.squeeze(0)
        return v_pred