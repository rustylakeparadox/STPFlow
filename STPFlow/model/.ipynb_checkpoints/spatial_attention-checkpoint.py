import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_lr_prior=True, lr_lambda=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_lr_prior = use_lr_prior
        self.lr_lambda = lr_lambda

        # 多头注意力的线性变换
        self.W_q = nn.Linear(hidden_dim, hidden_dim * n_heads, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim * n_heads, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim * n_heads, bias=False)

        # 输出投影
        self.W_out = nn.Linear(hidden_dim * n_heads, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, edge_index, lr_mat=None, num_nodes_per_sample=None):
        """
        Args:
            h: [total_nodes, hidden_dim]  所有样本的节点特征拼合
            edge_index: [2, total_edges]  边索引（已包含 batch 偏移）
            lr_mat: 可选的稀疏矩阵或向量，包含每条边的 LR 得分
            num_nodes_per_sample: 每个样本的节点数（用于恢复 batch）
        Returns:
            h_out: [total_nodes, hidden_dim]  更新后的节点特征
        """
        N, H = h.shape
        # 多头线性变换
        q = self.W_q(h).view(N, self.n_heads, H)  # [N, n_heads, H]
        k = self.W_k(h).view(N, self.n_heads, H)
        v = self.W_v(h).view(N, self.n_heads, H)

        # 根据 edge_index 提取源节点和目标节点
        src, dst = edge_index[0], edge_index[1]  # [E]

        q_src = q[src]  # [E, n_heads, H]
        k_dst = k[dst]  # [E, n_heads, H]

        # 计算注意力分数（点积）
        attn_score = (q_src * k_dst).sum(dim=-1)  # [E, n_heads]

        # 如果使用 LR 先验，将 LR 得分加到注意力分数上（假设 lr_mat 为 [E] 或可广播到 n_heads）
        if self.use_lr_prior and lr_mat is not None:
            # lr_mat 应为每条边的 LR 得分，形状 [E]
            lr_score = lr_mat[src, dst] if lr_mat.dim() == 2 else lr_mat
            # 将 LR 得分加到每个头上（或可学习每个头的权重）
            attn_score = attn_score + self.lr_lambda * lr_score.unsqueeze(-1)

        # 按目标节点 softmax
        attn_weight = self._edge_softmax(attn_score, dst, N)  # [E, n_heads]

        # 加权聚合
        v_dst = v[dst]  # [E, n_heads, H]
        h_out = torch.zeros(N, self.n_heads, H, device=h.device)
        h_out.index_add_(0, src, attn_weight.unsqueeze(-1) * v_dst)  # [N, n_heads, H]

        # 合并多头并输出
        h_out = h_out.view(N, self.n_heads * H)  # [N, n_heads*H]
        h_out = self.W_out(h_out)                # [N, H]

        return h_out

    def _edge_softmax(self, score, index, num_nodes):
        """按 index（目标节点）进行 softmax"""
        score_exp = torch.exp(score - score.max())
        norm = torch.zeros(num_nodes, self.n_heads, device=score.device)
        norm.scatter_add_(0, index.unsqueeze(-1).expand_as(score_exp), score_exp)
        norm = norm[index]  # [E, n_heads]
        return score_exp / (norm + 1e-10)