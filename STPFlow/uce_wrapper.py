import os
import subprocess
import scanpy as sc
import numpy as np


DEFAULT_UCE_MODEL = "/home/zhangdaoliang/liuwy/STPFlow-main/UCE-main/model_files/4layer_model.torch"
DEFAULT_UCE_SCRIPT = "/home/zhangdaoliang/liuwy/STPFlow-main/UCE-main/eval_single_anndata.py"


def _infer_uce_h5ad_path(rna_path: str, out_dir: str) -> str:
    base = os.path.splitext(os.path.basename(rna_path))[0]
    return os.path.join(out_dir, f"{base}_uce_adata.h5ad")


def run_uce_eval(
    rna_path: str,
    out_dir: str,
    species: str = "human",
    model_loc: str = DEFAULT_UCE_MODEL,
    batch_size: int = 32,
    uce_script: str = DEFAULT_UCE_SCRIPT,
    force: bool = False,
) -> str:
    """
    调用 UCE 的 eval_single_anndata.py，返回生成的 *_uce_adata.h5ad 路径
    """
    os.makedirs(out_dir, exist_ok=True)
    uce_h5ad_path = _infer_uce_h5ad_path(rna_path, out_dir)

    if os.path.exists(uce_h5ad_path) and not force:
        print(f"[INFO] Reusing existing UCE output: {uce_h5ad_path}")
        return uce_h5ad_path

    cmd = [
        "python",
        uce_script,
        "--adata_path", rna_path,
        "--dir", out_dir,
        "--species", species,
        "--model_loc", model_loc,
        "--batch_size", str(batch_size),
    ]

    print("[INFO] Running UCE:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(uce_h5ad_path):
        raise FileNotFoundError(f"UCE output not found: {uce_h5ad_path}")

    return uce_h5ad_path


def export_uce_to_npy(
    raw_rna_path: str,
    uce_h5ad_path: str,
    out_prefix: str,
):
    """
    从 *_uce_adata.h5ad 提取：
      - out_prefix + "_X_uce.npy"
      - out_prefix + "_common_obs_names.npy"
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    raw_ad = sc.read_h5ad(raw_rna_path)
    uce_ad = sc.read_h5ad(uce_h5ad_path)

    raw_ad.var_names_make_unique()
    uce_ad.var_names_make_unique()

    common_obs = raw_ad.obs_names.intersection(uce_ad.obs_names)
    if len(common_obs) == 0:
        raise ValueError(
            f"No common obs_names between\n{raw_rna_path}\nand\n{uce_h5ad_path}"
        )

    uce_ad = uce_ad[common_obs].copy()

    if "X_uce" in uce_ad.obsm:
        X_uce = uce_ad.obsm["X_uce"]
        src = 'obsm["X_uce"]'
    elif "X_uce" in uce_ad.layers:
        X_uce = uce_ad.layers["X_uce"]
        src = 'layers["X_uce"]'
    elif uce_ad.X is not None:
        X_uce = uce_ad.X
        src = "X"
    else:
        raise ValueError(f"Cannot find UCE embedding matrix in {uce_h5ad_path}")

    if hasattr(X_uce, "toarray"):
        X_uce = X_uce.toarray()
    else:
        X_uce = np.asarray(X_uce)

    X_uce = X_uce.astype(np.float32)
    common_obs_arr = np.array(common_obs, dtype=object)

    x_uce_path = out_prefix + "_X_uce.npy"
    obs_path = out_prefix + "_common_obs_names.npy"

    np.save(x_uce_path, X_uce)
    np.save(obs_path, common_obs_arr, allow_pickle=True)

    print(f"[INFO] UCE source: {src}")
    print(f"[INFO] saved: {x_uce_path} shape={X_uce.shape}")
    print(f"[INFO] saved: {obs_path} n_obs={len(common_obs_arr)}")

    return x_uce_path, obs_path


def prepare_uce_features(
    rna_path: str,
    cache_dir: str,
    sample_name: str,
    species: str = "human",
    model_loc: str = DEFAULT_UCE_MODEL,
    batch_size: int = 32,
    uce_script: str = DEFAULT_UCE_SCRIPT,
    force_recompute: bool = False,
):
    """
      1. 跑 UCE（如已有缓存则复用）
      2. 从 uce_adata 导出 npy
    返回：
      x_uce_path, obs_names_path, uce_h5ad_path
    """
    print(f"train cache_dir: {cache_dir}")
    
    cache_dir = cache_dir.rstrip(os.sep)
    uce_out_dir = os.path.join(cache_dir, "uce_h5ad")
    os.makedirs(uce_out_dir, exist_ok=True)

    npy_out_prefix = os.path.join(uce_out_dir, sample_name)
    x_uce_path = npy_out_prefix + "_X_uce.npy"
    obs_path = npy_out_prefix + "_common_obs_names.npy"

    uce_h5ad_path = _infer_uce_h5ad_path(rna_path, uce_out_dir)

    if (
        os.path.exists(x_uce_path)
        and os.path.exists(obs_path)
        and not force_recompute
    ):
        print(f"[INFO] Reusing existing UCE npy cache for {sample_name}")
        return x_uce_path, obs_path, uce_h5ad_path

    if (
        os.path.exists(uce_h5ad_path)
        and not force_recompute
    ):
        print(f"[INFO] Reusing existing UCE output: {uce_h5ad_path}")
        x_uce_path, obs_path = export_uce_to_npy(
            raw_rna_path=rna_path,
            uce_h5ad_path=uce_h5ad_path,
            out_prefix=npy_out_prefix,
        )
        return x_uce_path, obs_path, uce_h5ad_path

    print(f"[INFO] No existing UCE cache found for {sample_name}, running UCE...")
    uce_h5ad_path = run_uce_eval(
        rna_path=rna_path,
        out_dir=uce_out_dir,
        species=species,
        model_loc=model_loc,
        batch_size=batch_size,
        uce_script=uce_script,
        force=force_recompute,
    )

    x_uce_path, obs_path = export_uce_to_npy(
        raw_rna_path=rna_path,
        uce_h5ad_path=uce_h5ad_path,
        out_prefix=npy_out_prefix,
    )

    return x_uce_path, obs_path, uce_h5ad_path