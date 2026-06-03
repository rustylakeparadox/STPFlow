import torch
import torch.nn as nn
from graph_modules import SimpleGATv2Layer
from latent_flow_model import TimestepEmbedder  

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: [B]
        return self.net(t.unsqueeze(-1))  # [B, H]
    
class GeneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_genes, config.embed_dim)
        self.fc = nn.Linear(config.embed_dim, config.hidden_dim)

    def forward(self, x_gene):
        # x_gene: [B, N] -> Gene IDs
        x_emb = self.embedding(x_gene)  # [B, N, D_embed]
        return self.fc(x_emb)  # [B, N, H]

class FlowMatchingModel(nn.Module):
    def __init__(self, config):
        super(FlowMatchingModel, self).__init__()
        self.gene_encoder = GeneEncoder(config)
        self.protein_encoder = nn.Linear(config.n_proteins, config.hidden_dim)
        self.spatial_attn = SimpleGATv2Layer(config.hidden_dim, dropout=0.1)
        self.time_embedder = TimestepEmbedder(config.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_proteins)
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x_gene, z_t, t, edge_index, labels, uce_embedding=None):
        # Generate features
        gene_feat = self.gene_encoder(x_gene)
        
        # Time embedding
        t_emb = self.time_embedder(t)
        
        # Concatenate features
        h_raw = gene_feat + t_emb

        # Graph attention
        h_flat = h_raw.reshape(-1, h_raw.size(-1))
        h_attn = self.spatial_attn(h_flat, edge_index)

        # Prediction
        v_pred = self.output_head(h_attn)
        
        loss = self.loss_func(v_pred, labels)  
        return v_pred, loss