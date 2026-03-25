class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.loss_fn = nn.MSELoss()  
        # 空间平滑正则
        self.lambda_smooth = getattr(config, 'lambda_smooth', 0.0)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # 采样时间步和噪声（flow matching核心）
            B = batch['protein_exp'].size(0)  # total nodes in batch
            t = torch.rand(B, 1, device=self.device)  # [total_nodes, 1]
            noise = torch.randn_like(batch['protein_exp'])
            p_t = (1 - t) * batch['protein_exp'] + t * noise  # 线性插值
            
            # 前向传播
            v_pred, loss_main = self.model(
                p_t=p_t,
                t=t.squeeze(-1),
                gene=batch['gene_exp'],
                cell_type=batch.get('cell_type'),
                coords=batch['coords'],
                edge_index=batch['edge_index'],
                lr_mat=batch.get('lr_scores'), 
                labels=batch['protein_exp']  # 用于计算loss
            )
            
            # 空间平滑正则（鼓励相邻节点预测的向量场相似）
            if self.lambda_smooth > 0:
                src, dst = batch['edge_index'][0], batch['edge_index'][1]
                smooth_loss = torch.mean((v_pred[src] - v_pred[dst]).pow(2).sum(dim=-1))
                loss = loss_main + self.lambda_smooth * smooth_loss
            else:
                loss = loss_main
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sampled_protein = self.sample(batch)  
            all_preds.append(sampled_protein.cpu())
            all_targets.append(batch['protein_exp'].cpu())
        
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        # 需要确认一下是否有protein_names
        protein_names = getattr(self.config, 'protein_names', None)
        metrics = evaluate_protein_predictions(preds, targets, protein_names)
        return metrics
    
    def sample(self, batch, num_steps=100):
        """从噪声开始通过ODE生成蛋白质"""
        # 简化：Euler 法
        p_t = torch.randn_like(batch['protein_exp'])  # 初始噪声
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(p_t.size(0), 1, device=self.device) * (i * dt)
            v = self.model.inference(
                p_t=p_t,
                t=t.squeeze(-1),
                gene=batch['gene_exp'],
                cell_type=batch.get('cell_type'),
                coords=batch['coords'],
                edge_index=batch['edge_index'],
                lr_mat=batch.get('lr_scores')
            )
            p_t = p_t + v * dt
        return p_t
    
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            if epoch % self.config.val_freq == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_pearson={val_metrics['pearson_mean']:.4f}")
            self.scheduler.step()


def evaluate_protein_predictions(preds, targets, protein_names=None):
    """
    preds: [N, P] torch.Tensor or numpy array
    targets: [N, P] torch.Tensor or numpy array
    protein_names: list of str, length P
    Returns dict with per-protein metrics and aggregated stats.
    """
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    
    P = preds.shape[1]
    l2_errors = []
    pearson_corrs = []
    pearson_dicts = []
    
    for i in range(P):
        pred = preds[:, i]
        target = targets[:, i]
        l2 = np.mean((pred - target) ** 2)
        l2_errors.append(l2)
        # 避免全零或常数导致的 Pearson 计算错误
        if np.std(pred) == 0 or np.std(target) == 0:
            pearson = np.nan
        else:
            pearson, _ = pearsonr(target, pred)
        pearson_corrs.append(pearson)
        if protein_names is not None:
            pearson_dicts.append({'name': protein_names[i], 'pearson_corr': pearson})
    
    results = {
        'l2_errors': l2_errors,
        'pearson_corrs': pearson_dicts if protein_names else pearson_corrs,
        'pearson_mean': float(np.nanmean(pearson_corrs)),
        'pearson_std': float(np.nanstd(pearson_corrs)),
        'l2_error_mean': float(np.mean(l2_errors)),
        'l2_error_std': float(np.std(l2_errors)),
    }
    return results