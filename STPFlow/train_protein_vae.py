import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import torch.nn.functional as F
import scanpy as sc
from protein_vae import ProteinVAE, vae_loss, corr_loss
from datasets import normalize_protein
from evaluate import evaluate_protein_vae

def normalize_protein(x, method="log1p"):
    x = x.copy()
    if method == "log1p":
        return np.log1p(x)
    elif method == "none" or method is None:
        return x
    else:
        raise ValueError(f"Unsupported protein normalize method: {method}")


# 自定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, h5ad_path, protein_normalize="log1p"):
        adata = sc.read_h5ad(h5ad_path)
        X_protein = adata.X
        if hasattr(X_protein, "toarray"):
            X_protein = X_protein.toarray()
        else:
            X_protein = np.asarray(X_protein)

        X_protein = normalize_protein(X_protein, protein_normalize).astype(np.float32)
        self.x_protein = torch.from_numpy(X_protein)
        self.protein_names = list(adata.var_names)

        print(f"[INFO] protein shape: {self.x_protein.shape}")

    def __len__(self):
        return self.x_protein.shape[0]

    def __getitem__(self, idx):
        return self.x_protein[idx]


# 评估指标：计算 Pearson 相关系数
def metric_func_protein(preds_all: np.ndarray, y_test: np.ndarray, proteins=None):
    pearson_corrs = []
    pearson_proteins = []

    if proteins is not None:
        proteins = list(proteins)

    n_nan_proteins = 0

    for i in range(y_test.shape[1]):
        preds = preds_all[:, i]
        target_vals = y_test[:, i]

        try:
            pearson_corr, _ = pearsonr(target_vals, preds)
        except Exception:
            pearson_corr = np.nan

        pearson_corrs.append(pearson_corr)
        if np.isnan(pearson_corr):
            n_nan_proteins += 1

        pearson_proteins.append({
            "name": proteins[i] if proteins is not None and i < len(proteins) else f"protein_{i}",
            "pearson_corr": float(pearson_corr) if not np.isnan(pearson_corr) else np.nan,
        })

    if n_nan_proteins > 0:
        print(f"[WARN] {n_nan_proteins} proteins have NaN Pearson correlation")

    return {
        "pearson_corrs": pearson_proteins,
        "pearson_mean": float(np.nanmean(pearson_corrs)),
        "pearson_std": float(np.nanstd(pearson_corrs)),
        "n_test": int(y_test.shape[0]),
    }


# 计算潜在空间
@torch.no_grad()
def export_latent(model, loader, device):
    model.eval()
    zs = []
    xs = []
    for x in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        z = mu
        zs.append(z.cpu().numpy())
        xs.append(x.cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(xs, axis=0)

def get_beta(epoch, max_beta, warmup_epochs):
    if warmup_epochs <= 0:
        return max_beta
    return min(max_beta, max_beta * epoch / warmup_epochs)

def train_protein_vae(args,device = None):
    if device.type == "cuda":
        print("[DEBUG ProteinVAE] GPU:", torch.cuda.get_device_name(device))
    # 创建保存目录
    os.makedirs(args.result_root, exist_ok=True)

    # 准备数据
    dataset = ProteinDataset(args.train_adt_path, protein_normalize=args.protein_normalize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 初始化模型
    model = ProteinVAE(
        input_dim=dataset.x_protein.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1e18
    best_state = None

    # 开始训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, recons, kls = [], [], []

        for x in loader:
            x = x.to(device)
            x_hat, mu, logvar, z = model(x)
            beta_t = get_beta(epoch, args.beta, args.kl_warmup_epochs)
            recon = F.mse_loss(x_hat, x, reduction="mean")

            kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            pcc_loss = corr_loss(x_hat, x)

            loss = recon + beta_t * kl + args.lambda_corr * pcc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            recons.append(recon.item())
            kls.append(kl.item())

        avg_loss = float(np.mean(losses))
        avg_recon = float(np.mean(recons))
        avg_kl = float(np.mean(kls))

        print(
            f"epoch={epoch:03d} "
            f"beta={beta_t:.6g} "
            f"loss={avg_loss:.6f} "
            f"recon={avg_recon:.6f} "
            f"kl={avg_kl:.6f}"
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": dataset.x_protein.shape[1],
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "protein_normalize": args.protein_normalize,
            }
    print("[INFO] Evaluating model performance after training...")
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    print("[INFO] Evaluating best ProteinVAE checkpoint after training...")
    metrics = evaluate_protein_vae(model, dataset, device, compute_metrics=True)

    torch.save(best_state, args.protein_vae_ckpt)