import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_attention import SpatialAttentionLayer  # 需要新建图注意力层
from .config import ModelConfig  #ModelConfig文件需要修改


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



class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super(Denoiser, self).__init__()

        self.config = config
        self.n_proteins = config.n_proteins
        self.hidden_dim = config.hidden_dim
        self.use_lr_prior = getattr(config, 'use_lr_prior', True)
        self.lr_lambda = getattr(config, 'lr_lambda', 1.0)

        self.gene_encoder = nn.Linear(config.n_genes, config.hidden_dim)
        self.protein_encoder = nn.Linear(config.n_proteins, config.hidden_dim)

        if hasattr(config, 'n_cell_types') and config.n_cell_types > 0:
            self.type_encoder = nn.Embedding(config.n_cell_types, config.type_dim)
        else:
            self.type_encoder = nn.Linear(config.type_dim, config.type_dim)

        self.time_embedder = TimestepEmbedder(config.hidden_dim)

       self.spatial_attn = SpatialAttentionLayer(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_lr_prior=self.use_lr_prior,
            lr_lambda=self.lr_lambda
        )
        self.loss_func = nn.MSELoss()

    def inference(self, p_t, t, gene, cell_type, coords, edge_index, lr_mat=None):
         """
        During inference, only the predicted vector field is returned
        """

        v_pred, _ = self.forward(p_t, t, gene, cell_type, coords, edge_index, lr_mat)
        return v_pred

    def forward(self, p_t, t, gene, cell_type, coords, edge_index, lr_mat=None, labels=None):
         """
        Args:
            p_t:     [B, N_pro] or [B, num_spots, N_pro]  Noise protein at the current time step
            t:       [B] or [B, 1]  time step（标量）
            gene:    [B, N_gene] or [B, num_spots, N_gene]  gene expression
            cell_type: [B] or [B, num_spots, type_dim]  celltype(Discrete index or feature vector)
            coords:  [B, num_spots, 2]  spatial coordination
            edge_index: [2, E]  
            lr_mat:  [num_spots, num_spots] or sparse tensor  Ligand-receptor scoring matrix（可选）
            labels:  [B, num_spots, N_pro]  protein ground truth（for loss calculation）

        Returns:
            prediction: Predicted vector field
            loss:       If labels are provided, the MSE loss will be returned; otherwise, None will be returned
        """
        
        gene_feat = self.gene_encoder(gene)                     # [B, N_spots, H]
        prot_feat = self.protein_encoder(p_t)                    # [B, N_spots, H]

        if cell_type.dim() == 2:  # [B, type_dim] 可能缺少 spot 维度
            cell_type = cell_type.unsqueeze(1).expand(-1, gene.size(1), -1)
        if cell_type.dtype in [torch.long, torch.int]:
            type_feat = self.type_encoder(cell_type)             # [B, N_spots, type_dim]
        else:
            type_feat = self.type_encoder(cell_type)             # 线性变换

        #time embeddings
        t_emb = self.time_embedder(t)                             # [B, H]
        t_emb = t_emb.unsqueeze(1).expand(-1, gene.size(1), -1)   # [B, N_spots, H]

        #融合特征
        h = gene_feat + prot_feat + type_feat + t_emb             # [B, N_spots, H]

        #图注意力增强
        B, N, H = h.shape
        h_flat = h.view(B * N, H)                                 # [B*N, H]
        #此处代码需要后续完善，edge_index是否包含batch信息？需要根据 batch 大小构建 batch 化的边索引，是否需要加上偏移？
        #在数据预处理阶段需要输出edge_index

        #调用spatial_attention
        h_attn = self.spatial_attn(h_flat, edge_index, lr_mat, N)  # [B*N, H]
        # 恢复 batch 维度
        h_attn = h_attn.view(B, N, H)

        # 输出向量场
        v_pred = self.output_head(h_attn)                          # [B, N_spots, N_pro]

        # 计算loss function（若提供 labels）
        loss = None
        if labels is not None:
            # 可选的 pad_mask（如果某些 spot 无效）
            pad_mask = (gene.sum(dim=-1) == 0)  # 示例：根据基因表达是否为0判断有效spot
            loss = self.loss_func(v_pred[~pad_mask], labels[~pad_mask])

        return v_pred, loss