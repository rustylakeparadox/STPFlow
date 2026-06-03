import torch
import torch.nn as nn
from graph_modules import SimpleGATv2Layer

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

class LatentFlowModel(nn.Module):
    def __init__(
        self,
        gene_dim,
        z_dim,
        hidden_dim=256,
        uce_dim=1280,
        use_uce=False,
        uce_weight_init=0.1, 
        dropout = 0.1,  
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.uce_dim = uce_dim
        self.use_uce = use_uce

        self.rna_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.t_encoder = TimestepEmbedder(hidden_dim)  # Time step embedder
        self.uce_weight = nn.Parameter(torch.tensor(uce_weight_init), requires_grad=True)       

        # UCE branch
        if uce_dim is None:
            raise ValueError("uce_dim must be provided")

        # pre-graph FiLM modulation
        self.uce_to_film = nn.Sequential(
            nn.Linear(uce_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # post-graph residual fusion
        self.post_uce_proj = nn.Sequential(
            nn.Linear(uce_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.post_graph_norm = nn.LayerNorm(hidden_dim)

        self.fuse_norm = nn.LayerNorm(hidden_dim)

        # fixed 2-layer spatial encoder with GATv2
        self.graph_layers = nn.ModuleList([
            SimpleGATv2Layer(hidden_dim, dropout=0.1),
            SimpleGATv2Layer(hidden_dim, dropout=0.1),
        ])
        
        # These are used only to predict z_base = f(RNA, UCE, graph).
        self.prior_graph_layers = nn.ModuleList([
            SimpleGATv2Layer(hidden_dim, dropout=dropout),
            SimpleGATv2Layer(hidden_dim, dropout=dropout),
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        
        self.prior_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

        self.loss_fn = nn.MSELoss()
        
    def _encode_gene_with_uce_film(self, x_gene, x_uce=None):
        """
        Encode RNA and optionally apply UCE FiLM modulation.

        If use_uce=False:
            RNA only -> gene_feat

        If use_uce=True:
            RNA + UCE FiLM -> gene_feat
        """
        # Always encode RNA first
        gene_feat = self.rna_encoder(x_gene)  # [N, H]

        # w/o UCE ablation: directly return RNA feature
        if not self.use_uce:
            return gene_feat

        # Full model: UCE is required
        if x_uce is None:
            raise ValueError("x_uce is required when use_uce=True")

        film = self.uce_to_film(x_uce)        # [N, 2H]
        gamma, beta = film.chunk(2, dim=-1)

        # stabilize FiLM scale
        gamma = torch.tanh(gamma)

        gene_feat = self.cond_norm((1.0 + gamma) * gene_feat + beta)

        return gene_feat

    def encode_condition(self, x_gene, x_uce=None, edge_index=None):
        """
        Encode condition information:
            use_uce=True:  RNA + UCE + spatial graph -> h_cond
            use_uce=False: RNA + spatial graph -> h_cond

        This is used by predict_prior().
        """
        if edge_index is None:
            raise ValueError("edge_index is required for spatial graph modeling")

        h = self._encode_gene_with_uce_film(x_gene, x_uce)

        # Graph propagation for condition prior
        for layer in self.prior_graph_layers:
            h = layer(h, edge_index)

        # Post-graph UCE fusion only when UCE is enabled
        if self.use_uce:
            if x_uce is None:
                raise ValueError("x_uce is required when use_uce=True")

            post_uce = self.post_uce_proj(x_uce)
            h = self.post_graph_norm(h + 0.5 * post_uce)
        else:
            h = self.post_graph_norm(h)

        return h
    
    def predict_prior(self, x_gene, x_uce=None, edge_index=None):
        """
        Predict RNA-conditioned protein latent prior:
            z_base = f(RNA, UCE, graph)

        This function does NOT use flow matching.
        It directly maps RNA / UCE / spatial graph features to protein latent space.

        Returns:
            z_base: [N, z_dim]
        """
        h_cond = self.encode_condition(
            x_gene=x_gene,
            x_uce=x_uce,
            edge_index=edge_index,
        )

        z_base = self.prior_head(h_cond)
        return z_base

    def forward_wo_flow_matching(self, x_gene, edge_index=None, x_uce=None, labels=None):
        """
        w/o Flow Matching ablation.

        Architecture:
            RNA / UCE / spatial graph
                    ↓
            condition encoder
                    ↓
            prior_head
                    ↓
            predicted protein latent z_hat

        This branch removes:
            - z_t input
            - timestep t
            - velocity prediction v_hat
            - flow matching loss

        Args:
            x_gene:     RNA expression, [N, gene_dim]
            edge_index: spatial graph edges
            x_uce:      UCE embedding, [N, uce_dim], optional if use_uce=False
            labels:     true protein latent z_true, [N, z_dim]

        Returns:
            z_hat: predicted protein latent, [N, z_dim]
            loss:  MSE(z_hat, labels) if labels is provided
        """
        z_hat = self.predict_prior(
            x_gene=x_gene,
            x_uce=x_uce,
            edge_index=edge_index,
        )

        loss = None
        if labels is not None:
            loss = self.loss_fn(z_hat, labels)

        return z_hat, loss

    def forward_direct(self, x_gene, edge_index=None, x_uce=None, labels=None):
        """
        Alias for backward compatibility.
        Same as forward_wo_flow_matching().
        """
        return self.forward_wo_flow_matching(
            x_gene=x_gene,
            edge_index=edge_index,
            x_uce=x_uce,
            labels=labels,
        )

    def forward(self, x_gene, z_t, t, edge_index=None, x_uce=None, labels=None):
        """
        Full STPFlow flow matching branch.

        Architecture:
            RNA / UCE condition + z_t + t
                    ↓
            spatial graph encoder
                    ↓
            velocity head
                    ↓
            v_hat

        Args:
            x_gene:     RNA expression, [N, gene_dim]
            z_t:        interpolated protein latent, [N, z_dim]
            t:          timestep, [N] or [B]
            edge_index: spatial graph edges
            x_uce:      UCE embedding, [N, uce_dim], optional if use_uce=False
            labels:     velocity target, [N, z_dim]

        Returns:
            v_hat: predicted velocity, [N, z_dim]
            loss:  MSE(v_hat, labels) if labels is provided
        """
        # ===== RNA encoding + optional UCE FiLM =====
        gene_feat = self._encode_gene_with_uce_film(x_gene, x_uce)

        z_feat = self.z_encoder(z_t)         # [N, H]
        t_feat = self.t_encoder(t)           # [N, H]

        # Fuse RNA condition, latent state, and time
        h = self.fuse_norm(gene_feat + z_feat + t_feat)

        if edge_index is None:
            raise ValueError("edge_index is required for spatial graph modeling")

        # ===== spatial propagation =====
        for layer in self.graph_layers:
            h = layer(h, edge_index)

        # ===== optional post-graph UCE fusion =====
        if self.use_uce:
            if x_uce is None:
                raise ValueError("x_uce is required when use_uce=True")

            post_uce = self.post_uce_proj(x_uce)
            h = self.post_graph_norm(h + 0.5 * post_uce)
        else:
            h = self.post_graph_norm(h)

        v_hat = self.head(h)

        loss = None
        if labels is not None:
            loss = self.loss_fn(v_hat, labels)

        return v_hat, loss