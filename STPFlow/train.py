import os
import json
import argparse
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train_protein_vae import ProteinVAE, train_protein_vae
from datasets import DenseRNAProteinDataset, select_top_variable_genes_by_train
from graph_modules import GraphSubsetDataset
from latent_flow_model import LatentFlowModel
from uce_wrapper import prepare_uce_features
from evaluate import evaluate_spatial

def load_or_train_protein_vae(args, device):
    if not os.path.exists(args.protein_vae_ckpt):
        print(f"{args.protein_vae_ckpt} not found, training ProteinVAE from scratch...")
        train_protein_vae(args,device=device)  # 确认这个函数内部会保存 ckpt

    vae_ckpt = torch.load(args.protein_vae_ckpt, map_location=device)

    protein_vae = ProteinVAE(
        input_dim=vae_ckpt["input_dim"],
        latent_dim=vae_ckpt["latent_dim"],
        hidden_dim=vae_ckpt["hidden_dim"],
    ).to(device)

    protein_vae.load_state_dict(vae_ckpt["model_state_dict"])
    protein_vae.eval()

    for p in protein_vae.parameters():
        p.requires_grad = False

    return protein_vae, vae_ckpt

def split_train_val_by_spatial_patch(
    coords,
    val_ratio=0.25,
    seed=42,
    n_bins_x=4,
    n_bins_y=4,
):
    """
    Split one spatial section into train/val by spatial patches.

    coords:
        shape [N, 2], spatial coordinates.

    val_ratio:
        proportion of spatial patches assigned to validation.

    n_bins_x, n_bins_y:
        number of spatial bins along x/y.
    """
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords should have shape [N, 2], got {coords.shape}")

    x = coords[:, 0]
    y = coords[:, 1]

    x_edges = np.linspace(x.min(), x.max(), n_bins_x + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins_y + 1)

    x_bin = np.digitize(x, x_edges[1:-1], right=False)
    y_bin = np.digitize(y, y_edges[1:-1], right=False)

    patch_labels = x_bin + n_bins_x * y_bin

    unique_patches = np.unique(patch_labels)

    train_patches, val_patches = train_test_split(
        unique_patches,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_mask = np.isin(patch_labels, train_patches)
    val_mask = np.isin(patch_labels, val_patches)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    return train_indices, val_indices, patch_labels

@torch.no_grad()
def encode_mu(vae, x_protein):
    mu, logvar = vae.encode(x_protein)
    return mu

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] device:", device)
    if device.type == "cuda":
        print("[DEBUG] GPU:", torch.cuda.get_device_name(device))
    uce_cache_root = os.path.join(args.result_root, "uce_cache")
    uce_cache_root = uce_cache_root.rstrip(os.sep)
    print(f"uce_cache_root: {uce_cache_root}")
    
    protein_vae, vae_ckpt = load_or_train_protein_vae(args, device)
    
    if args.use_uce:
        train_uce_path, train_uce_obs_names_path, _ = prepare_uce_features(
            rna_path=args.train_rna_path,
            cache_dir=os.path.join(uce_cache_root),
            sample_name="train",
            species=args.species,
            model_loc=args.uce_model_loc,
            batch_size=args.uce_batch_size,
            force_recompute=args.force_recompute_uce,
        )

        test_uce_path, test_uce_obs_names_path, _ = prepare_uce_features(
            rna_path=args.test_rna_path,
            cache_dir=os.path.join(uce_cache_root),
            sample_name="test",
            species=args.species,
            model_loc=args.uce_model_loc,
            batch_size=args.uce_batch_size,
            force_recompute=args.force_recompute_uce,
        )
    else:
        train_uce_path = None
        train_uce_obs_names_path = None
        test_uce_path = None
        test_uce_obs_names_path = None

    full_train_dataset = DenseRNAProteinDataset(
        args.train_rna_path,
        args.train_adt_path,
        gene_normalize=args.gene_normalize,
        protein_normalize=args.protein_normalize,
        uce_path=train_uce_path,
        uce_obs_names_path=train_uce_obs_names_path,
    )

    test_dataset = DenseRNAProteinDataset(
        args.test_rna_path,
        args.test_adt_path,
        gene_normalize=args.gene_normalize,
        protein_normalize=args.protein_normalize,
        uce_path=test_uce_path,
        uce_obs_names_path=test_uce_obs_names_path,
    )
    
    # ---------- align train/test genes by intersection ----------
    train_genes = list(full_train_dataset.gene_names)
    test_genes = list(test_dataset.gene_names)
    
    test_gene_to_idx = {g: i for i, g in enumerate(test_genes)}
    common_genes = [g for g in train_genes if g in test_gene_to_idx]
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between train and test RNA.")

    train_idx = [train_genes.index(g) for g in common_genes]
    test_idx = [test_gene_to_idx[g] for g in common_genes]

    full_train_dataset.x_gene = full_train_dataset.x_gene[:, train_idx]
    test_dataset.x_gene = test_dataset.x_gene[:, test_idx]

    full_train_dataset.gene_names = common_genes
    test_dataset.gene_names = common_genes
    
    full_train_dataset, test_dataset = select_top_variable_genes_by_train(
        full_train_dataset,
        test_dataset,
        n_top_genes=args.n_top_genes,
    )

    # 检查并输出只在train中存在的基因
    dropped_train_only = [g for g in train_genes if g not in common_genes]
    if len(dropped_train_only) > 0:
        print(
            f"[WARN] Dropping {len(dropped_train_only)} train-only genes not found in test. "
            f"Examples: {dropped_train_only[:10]}"
        )
    
    # ---------- split full_train_dataset into train/val by spatial patches ----------
    if not hasattr(full_train_dataset, "coords"):
        raise AttributeError(
            "full_train_dataset has no attribute `coords`. "
            "Please store spatial coordinates in DenseRNAProteinDataset as self.coords."
        )

    coords = full_train_dataset.coords

    train_indices, val_indices, patch_labels = split_train_val_by_spatial_patch(
        coords=coords,
        val_ratio=args.val_ratio,
        seed=args.seed,
        n_bins_x=args.patch_bins_x,
        n_bins_y=args.patch_bins_y,
    )

    train_dataset = GraphSubsetDataset(full_train_dataset, train_indices)
    train_dataset.set_augmentation(
        gene_dropout=args.gene_dropout,
        gene_noise_std=args.gene_noise_std,
        uce_dropout=args.uce_dropout_aug,
        uce_branch_drop=args.uce_branch_drop,
        edge_dropout=args.edge_dropout,
    )
    val_dataset = GraphSubsetDataset(full_train_dataset, val_indices)

    # ---------- build model ----------
    flow_model = LatentFlowModel(
        gene_dim=train_dataset.x_gene.shape[1],
        z_dim=vae_ckpt["latent_dim"],
        hidden_dim=args.hidden_dim,
        use_uce=args.use_uce,
        uce_dim=args.uce_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1e9
    best_result = None

    print("[INFO] Start latent_flow_x0 full-batch training...")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=10,
        factor=0.5,
    )

    # ---------- move full train graph to device ----------
    x_gene = train_dataset.x_gene.float().to(device)
    x_protein = train_dataset.x_protein.float().to(device)

    x_uce = None
    if train_dataset.x_uce is not None:
        x_uce = train_dataset.x_uce.float().to(device)

    edge_index = train_dataset.edge_index.long().to(device)

    print("[INFO] full-batch train tensors:")
    print("  x_gene:", x_gene.shape)
    print("  x_protein:", x_protein.shape)
    if x_uce is not None:
        print("  x_uce:", x_uce.shape)
    print("  edge_index:", edge_index.shape)

    # ---------- compute latent mean/std once before training ----------
    base_batch = train_dataset.get_full_batch(device=device, augment=False)

    x_protein_base = base_batch["x_protein"]

    protein_vae.eval()
    with torch.no_grad():
        z_train_raw = encode_mu(protein_vae, x_protein_base)
    latent_mean = z_train_raw.mean(dim=0, keepdim=True)
    latent_std = z_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

    print("[INFO] latent_mean shape:", latent_mean.shape)
    print("[INFO] latent_std shape:", latent_std.shape)

    min_delta = 1e-4

    for epoch in range(1, args.epochs + 1):
        flow_model.train()

        # ============================================================
        # 1. Get full-batch training data with dynamic augmentation
        # ============================================================
        batch = train_dataset.get_full_batch(
            device=device,
            augment=True,
        )

        x_gene = batch["x_gene"]
        x_protein = batch["x_protein"]
        x_uce = batch["x_uce"]
        edge_index = batch["edge_index"]

        # ============================================================
        # 2. Encode target protein latent
        # ============================================================
        with torch.no_grad():
            z1_raw = encode_mu(protein_vae, x_protein)

        # standardized target protein latent
        z1 = (z1_raw - latent_mean) / latent_std

        # Optional latent target noise, only for training
        if getattr(args, "latent_target_noise", 0.0) > 0:
            z1_train = z1 + args.latent_target_noise * torch.randn_like(z1)
        else:
            z1_train = z1

        # ============================================================
        # 3. Build source latent z0
        # ============================================================
        if args.use_conditional_prior:
            z_base = flow_model.predict_prior(
                x_gene=x_gene,
                x_uce=x_uce,
                edge_index=edge_index,
            )

            # source latent starts near RNA-conditioned prior
            z0 = z_base.detach() + args.prior_noise_scale * torch.randn_like(z_base)

        else:
            z_base = None
            z0 = torch.randn_like(z1_train)

        # ============================================================
        # 4. Sample t
        # ============================================================
        if args.t_bias_to_one:
            u = torch.rand(z1_train.size(0), device=device)
            t = 1.0 - u ** 2
        else:
            t = torch.rand(z1_train.size(0), device=device)

        z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1_train
        v_target = z1_train - z0

        # ============================================================
        # 5. Forward + losses
        # ============================================================
        optimizer.zero_grad(set_to_none=True)

        if args.wo_flow_matching:
            # ========================================================
            # Ablation: w/o Flow Matching
            # Directly predict protein latent z1 from RNA / UCE / graph
            # ========================================================
            z_hat, latent_loss = flow_model.forward_direct(
                x_gene=x_gene,
                x_uce=x_uce,
                edge_index=edge_index,
                labels=z1,          # target is true protein latent, NOT v_target
            )

            loss = latent_loss

            # For logging compatibility
            fm_loss = torch.tensor(0.0, device=device)
            endpoint_loss = torch.tensor(0.0, device=device)
            prior_loss = latent_loss

        else:
            # ========================================================
            # Full STPFlow: Flow Matching
            # ========================================================
            v_hat, fm_loss = flow_model(
                x_gene=x_gene,
                z_t=z_t,
                t=t,
                labels=v_target,
                x_uce=x_uce,
                edge_index=edge_index,
            )

            # Endpoint prediction from current velocity
            z1_pred = z_t + (1.0 - t.unsqueeze(-1)) * v_hat
            endpoint_loss = F.mse_loss(z1_pred, z1)

            # Conditional prior loss
            if args.use_conditional_prior:
                z_base = flow_model.predict_prior(
                    x_gene=x_gene,
                    x_uce=x_uce,
                    edge_index=edge_index,
                )
                prior_loss = F.mse_loss(z_base, z1)
            else:
                prior_loss = torch.tensor(0.0, device=device)

            loss = (
                fm_loss
                + args.endpoint_weight * endpoint_loss
                + args.prior_weight * prior_loss
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)

        optimizer.step()

        avg_loss = float(loss.item())
        # ============================================================
        # 6. Validation, no augmentation
        # ============================================================
        if epoch == 1 or epoch % args.eval_every == 0:
            train_metrics, _, _ = evaluate_spatial(
                flow_model,
                protein_vae,
                train_dataset,
                device,
                protein_names=train_dataset.protein_names,
                n_steps=args.n_steps,
                noise_seed=0,
                compute_metrics=True,
                latent_mean=latent_mean,
                latent_std=latent_std,
                use_conditional_prior=args.use_conditional_prior,
                prior_noise_scale_eval=args.prior_noise_scale_eval,
                wo_flow_matching=args.wo_flow_matching,
            )

            train_pcc = float(train_metrics["pearson_mean"])
            train_recon_loss = float(np.mean(train_metrics["recon_loss"]))
        else:
            train_pcc = np.nan
            train_recon_loss = np.nan
            
        metrics, preds_all, y_all = evaluate_spatial(
            flow_model,
            protein_vae,
            val_dataset,
            device,
            protein_names=val_dataset.protein_names,
            n_steps=args.n_steps,
            noise_seed=0,
            compute_metrics=True,
            latent_mean=latent_mean,
            latent_std=latent_std,
            use_conditional_prior=args.use_conditional_prior,
            prior_noise_scale_eval=args.prior_noise_scale_eval,
            wo_flow_matching=args.wo_flow_matching,
        )

        avg_val_loss = float(np.mean(metrics["recon_loss"]))
        val_pcc = float(metrics["pearson_mean"])

        #scheduler.step(val_pcc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"FM: {float(fm_loss.item()):.4f} - "
            f"Prior: {float(prior_loss.item()):.4f} - "
            f"Endpoint: {float(endpoint_loss.item()):.4f} - "
            f"Train Loss: {train_recon_loss:.4f} - "
            f"Train Pearson Mean: {train_pcc:.6f} - "
            f"Validation Loss: {avg_val_loss:.4f} - "
            f"Validation Pearson Mean: {val_pcc:.6f} - "
            f"LR: {current_lr:.2e}"
        )

        # ============================================================
        # 7. Save best checkpoint
        # ============================================================
        if val_pcc > best_metric + min_delta:
            best_metric = val_pcc
            best_result = metrics
            best_epoch = epoch

            torch.save(
                {
                    "flow_model_state_dict": flow_model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "best_pearson_mean": best_metric,
                    "latent_mean": latent_mean.detach().cpu(),
                    "latent_std": latent_std.detach().cpu(),
                    "gene_names": common_genes,
                    "protein_names": train_dataset.protein_names,
                    "best_epoch": best_epoch,

                    # new conditional-prior settings
                    "use_conditional_prior": args.use_conditional_prior,
                    "prior_noise_scale": args.prior_noise_scale,
                    "prior_noise_scale_eval": args.prior_noise_scale_eval,
                    "prior_weight": args.prior_weight,
                    "endpoint_weight": args.endpoint_weight,
                    "latent_corr_weight": getattr(args, "latent_corr_weight", 0.0),

                    # augmentation settings
                    "gene_dropout": args.gene_dropout,
                    "gene_noise_std": args.gene_noise_std,
                    "uce_dropout_aug": args.uce_dropout_aug,
                    "uce_branch_drop": args.uce_branch_drop,
                    "edge_dropout": args.edge_dropout,
                    "latent_target_noise": getattr(args, "latent_target_noise", 0.0),
                },
                os.path.join(args.save_dir, "best_latent_flow_x0_spatial.pt"),
            )

            with open(os.path.join(args.save_dir, "best_results.json"), "w") as f:
                json.dump(best_result, f, indent=4)

            print(f"[INFO] New best val PCC: {best_metric:.6f}")
                
    # ============================================================
    # Optional: refit on full training slice after hyperparameter
    # selection by validation.
    # ============================================================
    if args.refit_full_train:
        print("[INFO] Refit model on full training slice...")
        print(f"[INFO] Best epoch selected by validation: {best_epoch}")

        full_indices = np.arange(len(full_train_dataset))
        refit_dataset = GraphSubsetDataset(full_train_dataset, full_indices)

        refit_dataset.set_augmentation(
            gene_dropout=args.gene_dropout,
            gene_noise_std=args.gene_noise_std,
            uce_dropout=args.uce_dropout_aug,
            uce_branch_drop=args.uce_branch_drop,
            edge_dropout=args.edge_dropout,
        )

        if args.use_uce:
            if refit_dataset.x_uce is None:
                raise ValueError("args.use_uce=True, but refit_dataset.x_uce is None.")
            args.uce_dim = refit_dataset.x_uce.shape[1]
            print("[INFO] Refit inferred uce_dim:", args.uce_dim)

        flow_model = LatentFlowModel(
            gene_dim=refit_dataset.x_gene.shape[1],
            z_dim=vae_ckpt["latent_dim"],
            hidden_dim=args.hidden_dim,
            use_uce=args.use_uce,
            uce_dim=args.uce_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            flow_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Recompute latent mean/std using the full training slice.
        base_batch = refit_dataset.get_full_batch(device=device, augment=False)
        x_protein_base = base_batch["x_protein"]

        protein_vae.eval()
        with torch.no_grad():
            z_train_raw = encode_mu(protein_vae, x_protein_base)

        latent_mean = z_train_raw.mean(dim=0, keepdim=True)
        latent_std = z_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

        print("[INFO] Refit latent_mean shape:", latent_mean.shape)
        print("[INFO] Refit latent_std shape:", latent_std.shape)

        for refit_epoch in range(1, best_epoch + 1):
            flow_model.train()

            batch = refit_dataset.get_full_batch(
                device=device,
                augment=True,
            )

            x_gene = batch["x_gene"]
            x_protein = batch["x_protein"]
            x_uce = batch["x_uce"]
            edge_index = batch["edge_index"]

            with torch.no_grad():
                z1_raw = encode_mu(protein_vae, x_protein)

            z1 = (z1_raw - latent_mean) / latent_std

            if getattr(args, "latent_target_noise", 0.0) > 0:
                z1_train = z1 + args.latent_target_noise * torch.randn_like(z1)
            else:
                z1_train = z1

            if args.use_conditional_prior:
                z_base = flow_model.predict_prior(
                    x_gene=x_gene,
                    x_uce=x_uce,
                    edge_index=edge_index,
                )
                z0 = z_base.detach() + args.prior_noise_scale * torch.randn_like(z_base)
            else:
                z_base = None
                z0 = torch.randn_like(z1_train)

            if args.t_bias_to_one:
                u = torch.rand(z1_train.size(0), device=device)
                t = 1.0 - u ** 2
            else:
                t = torch.rand(z1_train.size(0), device=device)

            z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1_train
            v_target = z1_train - z0

            optimizer.zero_grad(set_to_none=True)

            if args.wo_flow_matching:
                z_hat, latent_loss = flow_model.forward_direct(
                    x_gene=x_gene,
                    x_uce=x_uce,
                    edge_index=edge_index,
                    labels=z1,
                )

                loss = latent_loss
                fm_loss = torch.tensor(0.0, device=device)
                endpoint_loss = torch.tensor(0.0, device=device)
                prior_loss = latent_loss

            else:
                v_hat, fm_loss = flow_model(
                    x_gene=x_gene,
                    z_t=z_t,
                    t=t,
                    labels=v_target,
                    x_uce=x_uce,
                    edge_index=edge_index,
                )

                z1_pred = z_t + (1.0 - t.unsqueeze(-1)) * v_hat
                endpoint_loss = F.mse_loss(z1_pred, z1)

                if args.use_conditional_prior:
                    z_base = flow_model.predict_prior(
                        x_gene=x_gene,
                        x_uce=x_uce,
                        edge_index=edge_index,
                    )
                    prior_loss = F.mse_loss(z_base, z1)
                else:
                    prior_loss = torch.tensor(0.0, device=device)

                loss = (
                    fm_loss
                    + args.endpoint_weight * endpoint_loss
                    + args.prior_weight * prior_loss
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()

            if refit_epoch == 1 or refit_epoch % args.eval_every == 0 or refit_epoch == best_epoch:
                print(
                    f"[REFIT] Epoch {refit_epoch}/{best_epoch} - "
                    f"Loss: {float(loss.item()):.4f} - "
                    f"FM: {float(fm_loss.item()):.4f} - "
                    f"Prior: {float(prior_loss.item()):.4f} - "
                    f"Endpoint: {float(endpoint_loss.item()):.4f}"
                )

        torch.save(
            {
                "flow_model_state_dict": flow_model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "best_epoch_from_validation": best_epoch,
                "latent_mean": latent_mean.detach().cpu(),
                "latent_std": latent_std.detach().cpu(),
                "gene_names": common_genes,
                "protein_names": refit_dataset.protein_names,
                "use_conditional_prior": args.use_conditional_prior,
                "prior_noise_scale": args.prior_noise_scale,
                "prior_noise_scale_eval": args.prior_noise_scale_eval,
                "prior_weight": args.prior_weight,
                "endpoint_weight": args.endpoint_weight,
                "gene_dropout": args.gene_dropout,
                "gene_noise_std": args.gene_noise_std,
                "uce_dropout_aug": args.uce_dropout_aug,
                "uce_branch_drop": args.uce_branch_drop,
                "edge_dropout": args.edge_dropout,
                "latent_target_noise": getattr(args, "latent_target_noise", 0.0),
            },
            os.path.join(args.save_dir, "refit_full_train_latent_flow.pt"),
        )

        print("[INFO] Saved refit checkpoint.")

    else:
        # 加载 validation best checkpoint
        best_ckpt = torch.load(
            os.path.join(args.save_dir, "best_latent_flow_x0_spatial.pt"),
            map_location=device,
        )

        flow_model.load_state_dict(best_ckpt["flow_model_state_dict"])
        flow_model.eval()

        if "latent_mean" in best_ckpt and "latent_std" in best_ckpt:
            latent_mean = best_ckpt["latent_mean"].to(device)
            latent_std = best_ckpt["latent_std"].to(device)

    flow_model.eval()
    # 用 best model 在 test_dataset 上预测
    test_metrics, test_pred, test_true = evaluate_spatial(
        flow_model,
        protein_vae,
        test_dataset,
        device,
        protein_names=test_dataset.protein_names,
        n_steps=args.n_steps,
        noise_seed=0,
        compute_metrics=True,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_conditional_prior=args.use_conditional_prior,
        prior_noise_scale_eval=args.prior_noise_scale_eval,
        wo_flow_matching=args.wo_flow_matching, 
    )
    test_results_df = pd.DataFrame(test_pred, columns=test_dataset.protein_names)
    test_results_df["true_labels"] = test_true.tolist()

    test_results_df.to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(args.save_dir, "final_test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    np.savez_compressed(
        os.path.join(args.save_dir, "best_predictions.npz"),
        y_pred=test_pred,
        y_true=test_true,
    )

    print(f"[INFO] Final test pearson_mean = {test_metrics['pearson_mean']:.6f}")
    print(f"[INFO] Saved test predictions to: {os.path.join(args.save_dir, 'best_predictions.npz')}")

    print(f"[INFO] Best pearson_mean = {best_metric:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rna_path", type=str, required=True)
    parser.add_argument("--train_adt_path", type=str, required=True)
    parser.add_argument("--test_rna_path", type=str, required=True)
    parser.add_argument("--test_adt_path", type=str, required=True)
    parser.add_argument("--protein_vae_ckpt", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32, help="The latent dimension size for ProteinVAE.")
    parser.add_argument("--beta", type=float, default=1e-5, help="KL Divergence weight for ProteinVAE.")
    parser.add_argument("--kl_warmup_epochs", type=int, default=200, help="Number of epochs for KL warm-up.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gene_normalize", type=str, default="log1p")
    parser.add_argument("--protein_normalize", type=str, default="log1p")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--uce_dim", type=int, default=1280)
    parser.add_argument("--uce_obs_names_path", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_uce", action="store_true")
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--uce_model_loc", type=str,
                        default="../UCE-main/model_files/4layer_model.torch")
    parser.add_argument("--uce_batch_size", type=int, default=32)
    parser.add_argument("--result_root",type=str,help="Dataset-level result root. UCE cache will be stored under result_root/uce_cache/.")
    parser.add_argument("--force_recompute_uce", action="store_true")
    parser.add_argument("--patch_bins_x", type=int, default=4)
    parser.add_argument("--patch_bins_y", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--endpoint_weight", type=float, default=1.0)
    parser.add_argument("--lambda_corr", type=float, default=0.1)
    parser.add_argument("--use_conditional_prior", action="store_true")

    parser.add_argument("--prior_noise_scale", type=float, default=0.05)
    parser.add_argument("--prior_noise_scale_eval", type=float, default=0.0)

    parser.add_argument("--prior_weight", type=float, default=5.0)

    parser.add_argument("--t_bias_to_one", action="store_true")
    parser.add_argument("--gene_dropout", type=float, default=0.03)
    parser.add_argument("--gene_noise_std", type=float, default=0.003)

    parser.add_argument("--uce_dropout_aug", type=float, default=0.03)
    parser.add_argument("--uce_branch_drop", type=float, default=0.05)

    parser.add_argument("--edge_dropout", type=float, default=0.05)
    parser.add_argument("--latent_target_noise", type=float, default=0.01)
    parser.add_argument("--refit_full_train", action="store_true")
    
    parser.add_argument(
        "--wo_flow_matching",
        action="store_true",
        help="Ablation: replace flow matching with deterministic latent regression."
    )
    args = parser.parse_args()

    main(args)