class ModelConfig():
    def __init__(
        self,
        dim=3,                      # 坐标维度（通常为2或3）
        d_input=64,                  # 输入特征维度（如基因/蛋白质特征投影前的维度，现可能不再使用，但保留兼容性）
        d_model=64,                   # 隐层维度
        n_layers=4,                   # 网络层数
        n_genes=50,                    # 基因数量
        n_proteins=50,                  # 蛋白质数量（输出维度）
        n_cell_types=10,                 # 细胞类型数量（若为离散标签，设为实际类别数；若为连续比例向量，可设为0）
        type_dim=16,                     # 细胞类型嵌入维度（若为连续向量，则是输入特征维度）
        use_lr_prior=True,                # 是否使用配体-受体先验——新增
        lr_lambda=1.0,                    # LR先验权重系数——新增
        dropout=0.1,
        attn_dropout=0.1,
        n_neighbors=16,                   # 邻域大小（用于KNN图）
        valid_radius=1e6,                  # 有效半径（若使用半径图）
        embedding_grad_frac=1.0,
        n_heads=4,                         # 注意力头数
        rbf_count=64,                       # RBF核数量（若用于距离编码）
        rbf_sigma=0.1,                       # RBF核宽度
        act="gelu",                           # 激活函数
        **kwargs,
    ):
        self.dim = dim
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_genes = n_genes
        self.n_proteins = n_proteins           
        self.n_cell_types = n_cell_types       
        self.type_dim = type_dim               
        self.use_lr_prior = use_lr_prior       
        self.lr_lambda = lr_lambda             
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.n_neighbors = n_neighbors
        self.valid_radius = valid_radius
        self.embedding_grad_frac = embedding_grad_frac
        self.n_heads = n_heads
        self.act = act
        self.rbf_count = rbf_count
        self.rbf_sigma = rbf_sigma

        # 接收额外参数并设为属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self._hidden_dim_check()

    def _hidden_dim_check(self):
        # 确保 d_model 可被 n_heads 整除
        assert self.d_model % self.n_heads == 0, f"d_model should be divisible by n_heads"