class ModelConfig:
    def __init__(
        self,
        # ===== Spatial / Graph =====
        dim=2,                      # 坐标维度
        n_neighbors=16,             # KNN图邻居数
        valid_radius=1e6,           # 半径图
        rbf_count=64,
        rbf_sigma=0.1,

        # ===== Model size =====
        d_model=128,                # 主隐藏维度
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        attn_dropout=0.1,
        act="gelu",

        # ===== Gene / Protein =====
        n_genes=3000,               # 基因词汇表大小
        n_proteins=200,             # 蛋白数量（输出维度）
        
        gene_embed_dim=128,
        max_genes_per_spot=2000,
        use_gene_embedding=True,
        pool_type="mean",           # mean / attention
        unknown_gene_strategy="zero",

        # ===== Cell type =====
        n_cell_types=10,
        type_dim=32,

        # ===== Flow Matching / Diffusion =====
        time_embed_dim=128,
        sigma_min=0.01,
        sigma_max=1.0,

        # ===== LR prior =====
        use_lr_prior=True,
        lr_lambda=1.0,

        # ===== Training =====
        embedding_grad_frac=1.0,
        loss_type="mse"
    ):
        # ===== Spatial =====
        self.dim = dim
        self.n_neighbors = n_neighbors
        self.valid_radius = valid_radius
        self.rbf_count = rbf_count
        self.rbf_sigma = rbf_sigma

        # ===== Model =====
        self.d_model = d_model
        self.hidden_dim = d_model   # 统一 hidden_dim 和 d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.act = act

        # ===== Gene / Protein =====
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.gene_embed_dim = gene_embed_dim
        self.max_genes_per_spot = max_genes_per_spot
        self.use_gene_embedding = use_gene_embedding
        self.pool_type = pool_type
        self.unknown_gene_strategy = unknown_gene_strategy

        # ===== Cell type =====
        self.n_cell_types = n_cell_types
        self.type_dim = type_dim

        # ===== Flow Matching =====
        self.time_embed_dim = time_embed_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # ===== LR prior =====
        self.use_lr_prior = use_lr_prior
        self.lr_lambda = lr_lambda

        # ===== Training =====
        self.embedding_grad_frac = embedding_grad_frac
        self.loss_type = loss_type

        self._hidden_dim_check()

    def _hidden_dim_check(self):
        assert self.d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads"
            
def build_config_from_args(args):
    return ModelConfig(
        d_model=args.hidden_dim,
        n_layers=args.n_layers,
        n_genes=args.n_genes,
        n_proteins=args.n_proteins,
        n_cell_types=0 if args.cell_type_col is None else 10,
        type_dim=args.type_dim,
        use_lr_prior=args.use_lr_prior,
        lr_lambda=args.lr_lambda,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        n_neighbors=args.n_neighbors,
        n_heads=args.n_heads,
        act=args.activation,
        gene_embed_dim=args.gene_embed_dim,
        max_genes_per_spot=args.max_genes_per_spot,
        use_gene_embedding=args.use_gene_embedding,
        unknown_gene_strategy="zero",
        pool_type=args.gene_pool_type
    )