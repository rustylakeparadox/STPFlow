import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from operator import itemgetter

import torch
from torch.utils.data import ConcatDataset, DataLoader

from STPFlow.utils import set_random_seed, get_current_time, merge_fold_results
from STPFlow.data.dataset import HESTDataset, HESTDatasetPath
from STPFlow.data.normalize_utils import get_normalize_method
from STPFlow.model.denoiser import Denoiser
from STPFlow.flow.interpolant import Interpolant
from STPFlow.app.flow.test import test 
from STPFlow.data.collate import collate_fn  


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    # 获取归一化函数（可为基因和蛋白质分别指定）
    # 实际可在 args 中增加 gene_norm, protein_norm
    gene_norm = get_normalize_method(args.gene_normalize)
    protein_norm = get_normalize_method(args.protein_normalize)

    print("Loading training datasets...")
    train_datasets = []
    for sample_id in train_sample_ids:
        ds = HESTDataset(
            dataset_path=HESTDatasetPath(  
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
            ),
            gene_normalize_method=gene_norm,
            protein_normalize_method=protein_norm,
            lr_pairs=args.lr_pairs,          # 需提供配体-受体对列表
            knn_k=args.n_neighbors,
            use_image_feat=args.use_image_feat,
            cell_type_col=args.cell_type_col, # 例如 'cell_type'
            lr_use_log1p=False                # 是否对基因表达取log再算LR
        )
        train_datasets.append(ds)
    
    # 训练集：将所有切片的数据集合并，每个切片作为一个样本
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # 验证集：每个切片单独一个 loader（用于按切片评估）
    val_loaders = []
    for sample_id in test_sample_ids:
        ds = HESTDataset(
            dataset_path=HESTDatasetPath(...),  # 同训练集构建
            gene_normalize_method=gene_norm,
            protein_normalize_method=protein_norm,
            lr_pairs=args.lr_pairs,
            knn_k=args.n_neighbors,
            use_image_feat=args.use_image_feat,
            cell_type_col=args.cell_type_col,
            lr_use_log1p=False
        )
        # 验证时每个 loader 只加载该切片（整个切片）
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        val_loaders.append(loader)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = Denoiser(args).to(device)

    # 流匹配插值器：使用高斯先验（蛋白质为连续值）
    diffusier = Interpolant(
        prior_sampler="gaussian",  # 固定为高斯

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
            # batch 是 collate_fn 返回的字典，已包含所有节点数据和边信息
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # 采样时间步和噪声
            B = batch['protein_exp'].size(0)  # 总节点数
            t = torch.rand(B, 1, device=device)  # [N, 1]
            noise = torch.randn_like(batch['protein_exp'])
            p_t = (1 - t) * batch['protein_exp'] + t * noise  # 线性插值

            # 前向传播
            v_pred, loss = model(
                p_t=p_t,
                t=t.squeeze(-1),
                gene=batch['gene_exp'],
                cell_type=batch.get('cell_type'),
                coords=batch['coords'],
                edge_index=batch['edge_index'],
                lr_mat=batch.get('lr_scores'),
                labels=batch['protein_exp']
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
                        # 可记录更多指标
                    })

    return best_val_dict["all"]


def run(args):
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    all_split_results = []

    for i in range(len(splits) // 2):
        print(f"Running split {i} on dataset {args.dataset}")
        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))
        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
        all_split_results.append(results)

    # 合并多个fold的结果
    kfold_results = merge_fold_results(all_split_results)  # 注意需确保 merge_fold_results 能处理蛋白质名
    with open(os.path.join(args.save_dir, 'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 原有参数保留，并增加新参数
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=["all"])
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--source_dataroot', default="/path/to/data")
    parser.add_argument('--embed_dataroot', default="/path/to/embed")
    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--protein_list', type=str, default=None)  # 蛋白质名称列表文件
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help='用于图像特征（若使用）')
    parser.add_argument('--use_image_feat', action='store_true', help='是否使用图像特征')
    parser.add_argument('--gene_normalize', type=str, default='log1p')
    parser.add_argument('--protein_normalize', type=str, default='log1p')
    parser.add_argument('--exp_code', type=str, default='test')

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
    parser.add_argument('--n_cell_types', type=int, default=10)    # 若为0表示不使用
    parser.add_argument('--type_dim', type=int, default=16)        
    parser.add_argument('--use_lr_prior', action='store_true', default=True)  
    parser.add_argument('--lr_lambda', type=float, default=1.0)    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--feature_dim', type=int, default=1024, help='图像特征维度（若使用）')

    # 流匹配参数
    parser.add_argument('--n_sample_steps', type=int, default=5)

    # 新增：配体-受体对文件
    parser.add_argument('--lr_pairs_file', type=str, default=None, help='包含配体-受体对的JSON文件')

    # 新增：细胞类型列名
    parser.add_argument('--cell_type_col', type=str, default=None, help='adata.obs中细胞类型列名')

    args = parser.parse_args()

    # 加载配体-受体对
    if args.lr_pairs_file is not None:
        with open(args.lr_pairs_file, 'r') as f:
            lr_pairs = json.load(f)  # 假设是列表 of [ligand, receptor]
        args.lr_pairs = lr_pairs
    else:
        args.lr_pairs = []

    # 加载蛋白质名称列表（可选）
    if args.protein_list is not None:
        with open(os.path.join(args.source_dataroot, args.dataset, args.protein_list), 'r') as f:
            args.protein_names = json.load(f)['proteins']
    else:
        args.protein_names = None

    set_random_seed(args.seed)

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

    if args.datasets[0] == "all":
        # 用户自定义数据集列表
        args.datasets = ["LUNG", "HCC", "COAD", ...]  # 根据实际情况

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        run(args)

    if args.use_wandb:
        wandb.finish()