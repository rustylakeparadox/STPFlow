import os
import json
import wandb
import argparse
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
import pickle
from pathlib import Path
import glob

import torch
from torch.utils.data import ConcatDataset, DataLoader

from STPFlow.utils import get_current_time, merge_fold_results
from STPFlow.data.dataset import HESTDataset, HESTDatasetPath
from STPFlow.data.normalize_utils import get_normalize_method
from STPFlow.model.denoiser import Denoiser
from STPFlow.app.test import test
from STPFlow.data.dataset import collate_fn
from STPFlow.hest_utils.file_utils import build_gene_vocab
from STPFlow.model.config import build_config_from_args


# 简单的高斯先验，用于生成初始噪声
class SimpleGaussianPrior:
    def sample_from_prior(self, shape, device='cpu'):
        return torch.randn(shape, device=device)


def main(args, split_id, train_samples, test_samples, val_save_dir, checkpoint_save_dir):
    # 获取归一化函数
    gene_norm = args.gene_normalize
    protein_norm = args.protein_normalize

    train_h5ad_paths = [os.path.join(args.source_dataroot, f"{sid}.h5ad") for sid in train_samples]
    
    # 如果指定了词汇表文件且存在，则加载；否则从 h5ad 构建
    if args.gene_vocab and Path(args.gene_vocab).exists():
        with open(args.gene_vocab, 'rb') as f:
            gene2idx = pickle.load(f)
        print(f"从 {args.gene_vocab} 加载基因词汇表，大小：{len(gene2idx)}")
    else:
        gene2idx = build_gene_vocab(
            train_h5ad_paths,
            gene_key=args.gene_key
        )
        if args.gene_vocab:
            with open(args.gene_vocab, 'wb') as f:
                pickle.dump(gene2idx, f)
            print(f"基因词汇表已保存至 {args.gene_vocab}")
        else:
            print("基因词汇表动态构建完成，未保存文件。")
            
    train_datasets = []
    for sample_id in train_samples:
        h5ad_path = os.path.join(args.source_dataroot,  f"{sample_id}.h5ad")

        ds = HESTDataset(
            dataset_path=HESTDatasetPath(  
                name=sample_id,
                h5ad_path=h5ad_path
            ),
            gene_normalize_method=gene_norm,
            protein_normalize_method=protein_norm,
            lr_pairs=args.lr_pairs,
            gene2idx=gene2idx,
            max_genes_per_spot=args.max_genes_per_spot,
            knn_k=args.n_neighbors,
            use_image_feat=args.use_image_feat,
            cell_type_col=args.cell_type_col,
            lr_use_log1p=False
        )
        train_datasets.append(ds)

    # 合并训练数据集
    train_dataset = ConcatDataset(train_datasets)
    # Use batch_size=1 to keep each HESTDataset (one slice) as a separate sample
    # and avoid variable-node stacking issues in the model.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
        
        # 验证集：每个切片单独一个 loader
    val_loaders = []
    for sample_id in test_samples:
        h5ad_path = os.path.join(args.source_dataroot, f"{sample_id}.h5ad")
        ds = HESTDataset(
            dataset_path=HESTDatasetPath(
                name=sample_id,
                h5ad_path=h5ad_path,
                gene_list_path=None
            ),
            gene_normalize_method=gene_norm,
            protein_normalize_method=protein_norm,
            lr_pairs=args.lr_pairs,
            gene2idx=gene2idx,
            max_genes_per_spot=args.max_genes_per_spot,
            knn_k=args.n_neighbors,
            use_image_feat=args.use_image_feat,
            cell_type_col=args.cell_type_col,
            lr_use_log1p=False
        )
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        val_loaders.append(loader)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # Ensure model config matches constructed gene vocabulary size
    try:
        args.n_genes = len(gene2idx)
    except Exception:
        # fallback: keep provided args.n_genes
        pass
    # Determine number of proteins from datasets (ensure model linear matches all data)
    try:
        protein_dims = []
        for i, ds in enumerate(train_datasets):
            try:
                dim = ds.sp_data['protein_expr'].shape[1]
                protein_dims.append((f"train[{i}]", dim))
            except Exception:
                protein_dims.append((f"train[{i}]", None))
        try:
            for i, loader in enumerate(val_loaders):
                try:
                    ds = loader.dataset
                    dim = ds.sp_data['protein_expr'].shape[1]
                    protein_dims.append((f"val[{i}]", dim))
                except Exception:
                    protein_dims.append((f"val[{i}]", None))
        except Exception:
            pass

        # Print discovered dims for debugging
        print("[DEBUG] discovered protein dimensions:")
        for name, dim in protein_dims:
            print(f"[DEBUG]   {name}: {dim}")

        max_proteins = max([d for _, d in protein_dims if d is not None], default=args.n_proteins)
        args.n_proteins = int(max_proteins)
        print(f"[DEBUG] final args.n_proteins={args.n_proteins}")
    except Exception as e:
        print(f"[DEBUG] error while determining protein dims: {e}")
    config = build_config_from_args(args)
    model = Denoiser(config).to(device)
    # Debug: print model protein encoder shape
    try:
        print(f"[DEBUG] model.protein_encoder.weight.shape={tuple(model.protein_encoder.weight.shape)}")
        print(f"[DEBUG] config.n_proteins={config.n_proteins}")
    except Exception:
        pass

        # 使用简单的高斯先验（用于 test.py 中的采样）
    diffusier = SimpleGaussianPrior()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    best_pearson = -1
    best_val_dict = None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)

    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # 采样时间步和噪声（线性插值流匹配）
            # 使用 collate_fn 提供的 num_nodes_per_sample 来确定 batch 大小（B）
            B = int(batch['num_nodes_per_sample'].size(0))
            # 将按节点排列的 protein_expr 变为 (B, N, n_proteins)
            if B == 1:
                protein = batch['protein_expr'].unsqueeze(0)  # (1, N, F)
            else:
                # 如果将来支持 B>1，需要把 concat 的 tensor 按节点数量拆分并 pad/stack
                node_counts = batch['num_nodes_per_sample'].tolist()
                parts = batch['protein_expr'].split(node_counts, dim=0)
                protein = torch.stack([p for p in parts], dim=0)

            t = torch.rand(B, 1, device=device)
            noise = torch.randn_like(protein)
            # t: (B,1,1) -> broadcast 到 (B,N,F)
            p_t = (1 - t.view(B, 1, 1)) * protein + t.view(B, 1, 1) * noise
            # Debug prints for shapes
            try:
                print(f"[DEBUG] batch['protein_expr'].shape={tuple(batch['protein_expr'].shape)}")
                print(f"[DEBUG] num_nodes_per_sample={tuple(batch['num_nodes_per_sample'].tolist())}")
                print(f"[DEBUG] protein.shape={tuple(protein.shape)} p_t.shape={tuple(p_t.shape)} t.shape={tuple(t.shape)}")
            except Exception:
                pass

            v_pred, loss = model(
                p_t=p_t,
                t=t.squeeze(-1),
                gene_ids=batch['gene_ids'],
                gene_expr=batch['gene_expr'],
                cell_type=batch.get('cell_type'),
                coords=batch['coords'],
                edge_index=batch['edge_index'],
                lr_mat=batch.get('lr_scores'),
                labels=protein.squeeze(0) if B == 1 else batch['protein_expr']
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if args.use_wandb:
                wandb.log({f"{args.dataset}/Train/{split_id}/loss": loss.item()})

            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.4f}")

        if args.save_step > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
            if val_perf_dict["all"]['pearson_mean'] > best_pearson:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict
                for name, res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{name}_results.json'), 'w') as f:
                        json.dump(res, f, sort_keys=True, indent=4)
                early_stop_step = 0
            else:
                early_stop_step += 1
                if early_stop_step >= args.early_stop_patience:
                    print("Early stopping")
                    break

            if args.use_wandb:
                for name, res in val_perf_dict.items():
                    wandb.log({
                        f"{args.dataset}/Val/{split_id}/{name}/pearson_mean": res['pearson_mean'],
                        f"{args.dataset}/Val/{split_id}/{name}/pearson_std": res['pearson_std'],
                    })

    return best_val_dict["all"]


def run(args):
    # 1. 收集所有样本（每个样本用 (dataset, sample_id) 标识）
    all_samples = []
    for sample_id in args.datasets:
        h5ad_path = os.path.join(args.source_dataroot, f"{sample_id}.h5ad")
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"File not found: {h5ad_path}")
        all_samples.append(sample_id)   # 直接用样本ID

    if not all_samples:
        raise ValueError("No samples found.")

    # 2. 生成 fold 划分
    from sklearn.model_selection import KFold, train_test_split
    n_samples = len(all_samples)
    if args.n_splits == 1:
        # 单次划分
        train_samples, test_samples = train_test_split(
            all_samples, test_size=args.test_ratio, random_state=args.split_seed
        )
        folds = [(train_samples, test_samples)]
    else:
        # K 折交叉验证
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.split_seed)
        folds = []
        for train_idx, test_idx in kf.split(all_samples):
            train_samples = [all_samples[i] for i in train_idx]
            test_samples = [all_samples[i] for i in test_idx]
            folds.append((train_samples, test_samples))

    # 3. 对每一折进行训练
    all_results = []
    for fold_idx, (train_samples, test_samples) in enumerate(folds):
        print(f"Running fold {fold_idx+1}/{len(folds)}")
        fold_save_dir = os.path.join(args.save_dir, f'fold_{fold_idx}')
        os.makedirs(fold_save_dir, exist_ok=True)
        checkpoint_dir = os.path.join(fold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 调用 main 函数，传入样本标识列表
        result = main(
            args, fold_idx,
            train_samples, test_samples,
            fold_save_dir, checkpoint_dir
        )
        all_results.append(result)

    # 4. 合并所有折的结果
    if len(all_results) > 1:
        merged = merge_fold_results(all_results)
        with open(os.path.join(args.save_dir, 'results_kfold.json'), 'w') as f:
            json.dump(merged, f, indent=4)
    else:
        with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
            json.dump(all_results[0], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', required=True, help='样本ID列表，对应 source_dataroot 下的 .h5ad 文件')
    parser.add_argument('--n_splits', type=int, default=5, help='K 折交叉验证的折数（若为1，则按 --test_ratio 划分）')
    parser.add_argument('--split_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='当 n_splits=1 时，测试集比例')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--source_dataroot', default="/path/to/data")
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--gene_normalize', type=str, default='log1p')
    parser.add_argument('--protein_normalize', type=str, default='log1p')
    parser.add_argument('--exp_code', type=str, default='test')
    parser.add_argument('--gene_vocab', type=str, default=None, help='预构建的基因词汇表文件路径')
    parser.add_argument('--gene_key', type=str, default=None, help='h5ad 中存储基因名称的列名（默认使用 var_names）')
    parser.add_argument('--max_genes_per_spot', type=int, default=2000, 
                    help='每个spot最多保留的基因数（用于截断和填充）')
    parser.add_argument('--use_image_feat', action='store_true', 
                    help='是否使用图像特征（默认不使用）')
    parser.add_argument('--use_gene_embedding', action='store_true', default=True,
                    help='是否使用基因嵌入（默认使用）')
    parser.add_argument('--gene_embed_dim', type=int, default=128,
                        help='基因嵌入维度')
    parser.add_argument('--gene_pool_type', type=str, default='mean',
                        choices=['mean', 'attention'], help='基因聚合方式')
    

    # 训练超参数
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--save_step', type=int, default=-1)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)

    # 模型超参数
    parser.add_argument('--n_genes', type=int, default=50)
    parser.add_argument('--n_proteins', type=int, default=50)
    parser.add_argument('--cell_type_col', type=str, default=None, help='细胞类型列名')
    parser.add_argument('--type_dim', type=int, default=16)
    parser.add_argument('--lr_lambda', type=float, default=1.0, help='配体-受体先验权重')
    parser.add_argument('--use_lr_prior', action='store_true', default=True,
                        help='是否使用配体-受体先验')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--feature_dim', type=int, default=1024)

    # 流匹配参数
    parser.add_argument('--n_sample_steps', type=int, default=5)

    # 配体-受体对文件
    parser.add_argument('--species', type=str, default=None, help='选择lr_pairs物种数据：mouse/human')

    args = parser.parse_args()

    # 加载配体-受体对
    if args.species == 'human':
        lr_pairs_file = '../data/cellchat_lr_pairs_human.json'
    elif args.species == 'mouse':
        lr_pairs_file = '../data/cellchat_lr_pairs_mouse.json'
    else:
        raise ValueError(f"Unknown species: {args.species}")

    with open(lr_pairs_file, 'r') as f:
        args.lr_pairs = json.load(f)

    # 实验目录设置
    if args.exp_code is None:
        exp_code = f"protein_pred::{get_current_time()}"
    else:
        exp_code = args.exp_code + f"::{get_current_time()}"
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(project="spatial_proteomics", name=exp_code)
        wandb.config.update(args)

    print(f"Save dir: {save_dir}")
    print(args)

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        run(args)

    if args.use_wandb:
        wandb.finish()