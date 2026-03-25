import torch
import numpy as np
from scipy.stats import pearsonr

def metric_func_protein(preds_all: np.ndarray, y_test: np.ndarray, proteins: list):
    """计算蛋白质预测的指标"""
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_proteins = []
    
    n_nan_proteins = 0
    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        # R2 score 处理分母为零的情况
        ss_res = np.sum((target_vals - preds) ** 2)
        ss_tot = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if ss_tot == 0:
            r2 = 1.0 if ss_res == 0 else 0.0
        else:
            r2 = 1 - ss_res / ss_tot
        r2_scores.append(float(r2))

        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_proteins += 1

        score_dict = {
            'name': proteins[i] if proteins else f"protein_{i}",
            'pearson_corr': pearson_corr,
        }
        pearson_proteins.append(score_dict)

    if n_nan_proteins > 0:
        print(f"Warning: {n_nan_proteins} proteins have NaN Pearson correlation")

    return {
        'l2_errors': list(errors),
        'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_proteins,
        'pearson_mean': float(np.nanmean(pearson_corrs)),
        'pearson_std': float(np.nanstd(pearson_corrs)),
        'l2_error_q1': float(np.percentile(errors, 25)),
        'l2_error_q2': float(np.median(errors)),
        'l2_error_q3': float(np.percentile(errors, 75)),
        'r2_score_q1': float(np.percentile(r2_scores, 25)),
        'r2_score_q2': float(np.median(r2_scores)),
        'r2_score_q3': float(np.percentile(r2_scores, 75))
    }


@torch.no_grad()
def test(args, diffusier, model, loader_list, return_all=False):
    """
    验证/测试函数
    Args:
        loader_list: list of DataLoader, each loader returns a dict with keys:
                     'gene_exp', 'protein_exp', 'coords', 'cell_type', 'edge_index', 'lr_scores'
                     (每个 loader 对应一个切片，batch_size=1)
    """
    model.eval()
    all_pred, all_gt = [], []
    res_dict = {}

    for loader in loader_list:
        cur_pred, cur_gt = [], []

        for batch in loader:
            # batch 是一个字典，包含该切片的所有节点数据
            # 构造 torch.device（避免传入 int 导致在无 GPU 时出错）
            device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # 生成蛋白质：从噪声开始逐步去噪
            # 使用高斯先验（蛋白质为连续值）
            p_t = torch.randn_like(batch['protein_expr'])  # [N, P]
            ts = torch.linspace(0.01, 1.0, args.n_sample_steps, device=device)
            
            for step in range(len(ts)-1):
                t1 = ts[step].expand(p_t.size(0), 1)   # [N, 1]
                t2 = ts[step+1].expand(p_t.size(0), 1)
                
                # 调用模型推理（只前向，无loss）
                v_pred = model.inference(
                    p_t=p_t,
                    t=t1.squeeze(-1),
                    gene_ids=batch.get('gene_ids'),
                    gene_expr=batch.get('gene_expr'),
                    cell_type=batch.get('cell_type'),  # 可能为 None
                    coords=batch['coords'],
                    edge_index=batch['edge_index'],
                    lr_mat=batch.get('lr_scores')
                )
                d_t = t2 - t1
                if step == len(ts)-2:
                    # 最后一步得到预测值
                    sample = p_t + v_pred * d_t
                else:
                    # 继续去噪
                    p_t = p_t + v_pred * d_t

            cur_pred.append(sample.cpu().numpy())
            cur_gt.append(batch['protein_expr'].cpu().numpy())
        
        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        # 假设 loader.dataset 有 protein_names 属性, 这一块后面确认一下
        protein_names = getattr(loader.dataset, 'protein_names', None)
        cur_res_dict = metric_func_protein(cur_pred, cur_gt, protein_names)
        cur_res_dict.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.name] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    protein_names = getattr(loader_list[0].dataset, 'protein_names', None) if loader_list else None
    cur_res_dict = metric_func_protein(all_pred, all_gt, protein_names)
    cur_res_dict.update({'n_test': len(all_gt)})
    res_dict["all"] = cur_res_dict

    if return_all:
        return res_dict, {'preds_all': all_pred, 'targets_all': all_gt}
    return res_dict