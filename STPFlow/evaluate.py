import torch
import numpy as np
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

@torch.no_grad()
def encode_mu(vae, x_protein):
    mu, logvar = vae.encode(x_protein)
    return mu

@torch.no_grad()
def evaluate_protein_vae(model, dataset, device, compute_metrics=True):
    model.eval()

    x_protein = dataset.x_protein.float().to(device)

    mu, logvar = model.encode(x_protein)
    x_hat = model.decode(mu)

    recon_loss = torch.nn.functional.mse_loss(
        x_hat,
        x_protein,
        reduction="mean",
    )

    preds_all = x_hat.detach().cpu().numpy()
    y_all = x_protein.detach().cpu().numpy()

    pearson_corrs = []
    pearson_proteins = []

    for i in range(y_all.shape[1]):
        preds = preds_all[:, i]
        target = y_all[:, i]

        if np.std(preds) < 1e-8 or np.std(target) < 1e-8:
            corr = np.nan
        else:
            corr, _ = pearsonr(target, preds)

        name = (
            dataset.protein_names[i]
            if hasattr(dataset, "protein_names")
            else f"protein_{i}"
        )

        pearson_corrs.append(corr)
        pearson_proteins.append({
            "name": name,
            "pearson_corr": float(corr) if not np.isnan(corr) else np.nan,
            "true_std": float(np.std(target)),
            "pred_std": float(np.std(preds)),
        })

    metrics = {
        "recon_loss": float(recon_loss.item()),
        "pearson_mean": float(np.nanmean(pearson_corrs)),
        "pearson_std": float(np.nanstd(pearson_corrs)),
        "pearson_corrs": pearson_proteins,
    }

    print(f"Reconstruction Loss: {metrics['recon_loss']:.4f}")
    print(f"Pearson Mean: {metrics['pearson_mean']:.4f}")

    return metrics

def metric_func_protein(preds_all: np.ndarray, y_test: np.ndarray, proteins=None):
    pearson_corrs = []
    pearson_proteins = []
    n_nan_proteins = 0

    if proteins is not None:
        proteins = list(proteins)

    for i in range(y_test.shape[1]):
        preds = preds_all[:, i]
        target_vals = y_test[:, i]

        if np.std(target_vals) < 1e-8 or np.std(preds) < 1e-8:
            pearson_corr = np.nan
        else:
            try:
                pearson_corr, _ = pearsonr(target_vals, preds)
            except Exception:
                pearson_corr = np.nan

        if np.isnan(pearson_corr):
            n_nan_proteins += 1

        pearson_corrs.append(pearson_corr)
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
    
@torch.no_grad()
def evaluate_spatial(
    flow_model,
    protein_vae,
    dataset,
    device,
    protein_names=None,
    n_steps=20,
    noise_seed=0,
    compute_metrics=True,
    latent_mean=None,
    latent_std=None,
    use_conditional_prior=False,
    prior_noise_scale_eval=0.0,
    wo_flow_matching=False,
):
    """
    Evaluate protein prediction.

    Full STPFlow:
        Flow-matching inference in standardized protein latent space.

    w/o Flow Matching:
        Directly predict standardized protein latent:
            z_pred = flow_model.predict_prior(RNA, UCE, graph)

    In both cases:
        standardized latent -> raw VAE latent -> protein_vae.decode(...)
    """
    flow_model.eval()
    protein_vae.eval()

    # ============================================================
    # 0. Move dataset tensors to device
    # ============================================================
    x_gene = dataset.x_gene.float().to(device)

    x_uce = None
    if getattr(dataset, "x_uce", None) is not None:
        x_uce = dataset.x_uce.float().to(device)

    edge_index = None
    if getattr(dataset, "edge_index", None) is not None:
        edge_index = dataset.edge_index.long().to(device)

    x_protein = None
    if hasattr(dataset, "x_protein") and dataset.x_protein is not None:
        x_protein = dataset.x_protein.float().to(device)

    N = x_gene.size(0)

    if latent_mean is None or latent_std is None:
        raise ValueError("latent_mean and latent_std must be provided for evaluation.")

    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    z_dim = latent_mean.shape[1]

    # ============================================================
    # 1. Predict standardized protein latent
    # ============================================================
    if wo_flow_matching:
        # ========================================================
        # w/o Flow Matching:
        # Direct latent regression.
        #
        # This must match training:
        #     z_hat = flow_model.forward_direct(...)
        # During evaluation:
        #     z_t = flow_model.predict_prior(...)
        # ========================================================
        z_t = flow_model.predict_prior(
            x_gene=x_gene,
            x_uce=x_uce,
            edge_index=edge_index,
        )

    else:
        # ========================================================
        # Full STPFlow:
        # Flow-matching inference with Euler integration.
        # ========================================================
        generator = torch.Generator(device=device)
        generator.manual_seed(noise_seed)

        # --------------------------------------------------------
        # 1.1 Initialize latent state z_t
        # --------------------------------------------------------
        if use_conditional_prior:
            z_base = flow_model.predict_prior(
                x_gene=x_gene,
                x_uce=x_uce,
                edge_index=edge_index,
            )

            if prior_noise_scale_eval > 0:
                noise = torch.randn(
                    z_base.shape,
                    device=device,
                    generator=generator,
                )
                z_t = z_base + prior_noise_scale_eval * noise
            else:
                z_t = z_base.clone()

        else:
            z_t = torch.randn(
                (N, z_dim),
                device=device,
                generator=generator,
            )

        # --------------------------------------------------------
        # 1.2 Euler integration in standardized latent space
        # --------------------------------------------------------
        if n_steps > 0:
            dt = 1.0 / n_steps

            for step in range(n_steps):
                t_scalar = (step + 0.5) / n_steps
                t = torch.full((N,), t_scalar, device=device)

                v_hat, _ = flow_model(
                    x_gene=x_gene,
                    z_t=z_t,
                    t=t,
                    edge_index=edge_index,
                    labels=None,
                    x_uce=x_uce,
                )

                z_t = z_t + dt * v_hat

    # ============================================================
    # 2. Convert standardized latent back to raw VAE latent
    # ============================================================
    z_final_raw = z_t * latent_std + latent_mean

    protein_hat = protein_vae.decode(z_final_raw)

    # In case decode returns (recon, ...)
    if isinstance(protein_hat, tuple):
        protein_hat = protein_hat[0]

    preds_all = protein_hat.detach().cpu().numpy()

    # ============================================================
    # 3. Metrics
    # ============================================================
    if compute_metrics:
        if x_protein is None:
            raise ValueError("x_protein is required when compute_metrics=True.")

        y_all = x_protein.detach().cpu().numpy()

        protein_mse = torch.mean((protein_hat - x_protein) ** 2)

        # latent-space endpoint MSE
        enc_out = protein_vae.encode(x_protein)
        if isinstance(enc_out, tuple):
            z_true_raw = enc_out[0]
        else:
            z_true_raw = enc_out

        z_true = (z_true_raw - latent_mean) / latent_std
        latent_mse = torch.mean((z_t - z_true) ** 2)

        metrics = metric_func_protein(preds_all, y_all, protein_names)
        metrics["recon_loss"] = protein_mse.item()
        metrics["protein_mse"] = protein_mse.item()
        metrics["latent_mse"] = latent_mse.item()

    else:
        y_all = None
        metrics = {}

    return metrics, preds_all, y_all