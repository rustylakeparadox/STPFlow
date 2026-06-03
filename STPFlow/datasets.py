import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
from graph_modules import build_delaunay_edge_index, FullBatchAugmentMixin

def normalize_gene(x, method="log1p"):
    x = x.copy()
    if method == "log1p":
        return np.log1p(x)
    elif method == "none" or method is None:
        return x
    else:
        raise ValueError(f"Unsupported gene normalize method: {method}")


def normalize_protein(x, method="log1p"):
    x = x.copy()
    if method == "log1p":
        return np.log1p(x)
    elif method == "none" or method is None:
        return x
    else:
        raise ValueError(f"Unsupported protein normalize method: {method}")


def to_dense_array(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def extract_coords(adata):
    """
    Extract spatial coordinates from one AnnData.
    Return shape: [n_obs, 2]
    """
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]

    candidate_pairs = [
        ("x", "y"),
        ("row", "col"),
        ("array_row", "array_col"),
        ("x_tform", "y_tform"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ]

    for x_col, y_col in candidate_pairs:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return np.column_stack([
                adata.obs[x_col].to_numpy(dtype=np.float32),
                adata.obs[y_col].to_numpy(dtype=np.float32),
            ])

    coords = []
    for name in adata.obs_names.astype(str):
        s = str(name)

        # "sample_33x40" or "33x40"
        suffix = s.split("_")[-1]
        if "x" in suffix:
            x_str, y_str = suffix.split("x")
            coords.append([float(x_str), float(y_str)])
            continue

        # "33_40" or "sample_33_40"
        parts = s.split("_")
        if len(parts) >= 2:
            try:
                coords.append([float(parts[-2]), float(parts[-1])])
                continue
            except ValueError:
                pass

        raise ValueError(
            f"Cannot infer spatial coordinates from obs_name: {s}. "
            "Please use adata.obsm['spatial'] or coordinate columns in adata.obs."
        )

    return np.asarray(coords, dtype=np.float32)


def extract_coords_from_aligned_pair(rna_adata, adt_adata):
    """
    RNA and ADT have already been aligned to the same obs order.
    Prefer RNA coords; if RNA lacks coords, use ADT coords.
    """
    try:
        coords = extract_coords(rna_adata)
        source = "RNA"
    except Exception as rna_error:
        try:
            coords = extract_coords(adt_adata)
            source = "ADT"
        except Exception as adt_error:
            raise ValueError(
                "Cannot extract coordinates from either RNA or ADT AnnData.\n"
                f"RNA error: {rna_error}\n"
                f"ADT error: {adt_error}"
            )

    if coords.shape[0] != rna_adata.n_obs:
        raise ValueError(
            f"Coordinate row mismatch from {source}: "
            f"coords={coords.shape}, rna_adata.n_obs={rna_adata.n_obs}"
        )

    print(f"[INFO] coords source: {source}, shape={coords.shape}")
    return coords.astype(np.float32)


def finalize_aligned_pair(rna_adata, adt_adata, align_mode):
    """
    After RNA/ADT have been subset and ordered identically,
    build matrices and coords once.
    """
    X_gene = to_dense_array(rna_adata.X).astype(np.float32)
    X_protein = to_dense_array(adt_adata.X).astype(np.float32)
    protein_names = list(adt_adata.var_names)
    coords = extract_coords_from_aligned_pair(rna_adata, adt_adata)

    assert X_gene.shape[0] == X_protein.shape[0] == coords.shape[0], (
        f"Shape mismatch after {align_mode}: "
        f"X_gene={X_gene.shape}, X_protein={X_protein.shape}, coords={coords.shape}"
    )

    print(f"[INFO] aligned by {align_mode}: RNA {X_gene.shape}, ADT {X_protein.shape}")
    return rna_adata, adt_adata, X_gene, X_protein, protein_names, coords


def load_and_align_rna_adt(rna_path, adt_path):
    rna_adata = sc.read_h5ad(rna_path)
    adt_adata = sc.read_h5ad(adt_path)

    rna_adata.var_names_make_unique()
    adt_adata.var_names_make_unique()

    # ---------- 1) prefer spot column ----------
    if "spot" in rna_adata.obs.columns and "spot" in adt_adata.obs.columns:
        rna_spot = rna_adata.obs["spot"].astype(str)
        adt_spot = adt_adata.obs["spot"].astype(str)

        common_spots = sorted(set(rna_spot) & set(adt_spot))

        if len(common_spots) > 0:
            print(f"[INFO] spot overlap: {len(common_spots)}")

            rna_map = {}
            for obs_name, spot in zip(rna_adata.obs_names, rna_spot):
                if spot not in rna_map:
                    rna_map[spot] = obs_name

            adt_map = {}
            for obs_name, spot in zip(adt_adata.obs_names, adt_spot):
                if spot not in adt_map:
                    adt_map[spot] = obs_name

            rna_keep = [rna_map[s] for s in common_spots]
            adt_keep = [adt_map[s] for s in common_spots]

            rna_adata = rna_adata[rna_keep].copy()
            adt_adata = adt_adata[adt_keep].copy()

            rna_adata.obs_names = common_spots
            adt_adata.obs_names = common_spots

            return finalize_aligned_pair(
                rna_adata,
                adt_adata,
                align_mode="spot",
            )

    # ---------- 2) fallback: transformed coordinates ----------
    has_rna_tform = {"x_tform", "y_tform"}.issubset(rna_adata.obs.columns)
    has_adt_tform = {"x_tform", "y_tform"}.issubset(adt_adata.obs.columns)

    if has_rna_tform and has_adt_tform:
        rna_keys = [
            f"{float(x):.6f}_{float(y):.6f}"
            for x, y in zip(rna_adata.obs["x_tform"], rna_adata.obs["y_tform"])
        ]
        adt_keys = [
            f"{float(x):.6f}_{float(y):.6f}"
            for x, y in zip(adt_adata.obs["x_tform"], adt_adata.obs["y_tform"])
        ]

        common_keys = sorted(set(rna_keys) & set(adt_keys))

        if len(common_keys) > 0:
            print(f"[INFO] transformed-coordinate overlap: {len(common_keys)}")

            rna_map = {}
            for key, obs_name in zip(rna_keys, rna_adata.obs_names):
                if key not in rna_map:
                    rna_map[key] = obs_name

            adt_map = {}
            for key, obs_name in zip(adt_keys, adt_adata.obs_names):
                if key not in adt_map:
                    adt_map[key] = obs_name

            rna_keep = [rna_map[k] for k in common_keys]
            adt_keep = [adt_map[k] for k in common_keys]

            rna_adata = rna_adata[rna_keep].copy()
            adt_adata = adt_adata[adt_keep].copy()

            rna_adata.obs_names = common_keys
            adt_adata.obs_names = common_keys

            return finalize_aligned_pair(
                rna_adata,
                adt_adata,
                align_mode="x_tform/y_tform",
            )

    # ---------- 3) last fallback: exact obs_names ----------
    common_obs = rna_adata.obs_names.intersection(adt_adata.obs_names)

    if len(common_obs) > 0:
        print(f"[INFO] exact obs_names overlap: {len(common_obs)}")

        rna_adata = rna_adata[common_obs].copy()
        adt_adata = adt_adata[common_obs].copy()

        return finalize_aligned_pair(
            rna_adata,
            adt_adata,
            align_mode="obs_names",
        )

    raise ValueError(
        f"Cannot align RNA and ADT by spot, transformed coordinates, or obs_names between:\n"
        f"{rna_path}\nand\n{adt_path}"
    )
        
class DenseRNAProteinDataset(FullBatchAugmentMixin,Dataset):
    def __init__(
        self,
        rna_path,
        adt_path,
        gene_normalize="log1p",
        protein_normalize="log1p",
        uce_path=None,
        uce_obs_names_path=None,
    ):

        rna_adata, adt_adata, X_gene, X_protein, protein_names, coords = load_and_align_rna_adt(
            rna_path,
            adt_path,
        )

        X_gene = normalize_gene(X_gene, method=gene_normalize)
        X_protein = normalize_protein(X_protein, method=protein_normalize)

        base_adata = rna_adata

        self.x_uce = None

        if uce_path is not None and uce_obs_names_path is not None:
            uce = np.load(uce_path)
            uce_obs_names = np.load(uce_obs_names_path, allow_pickle=True).astype(str)

            adata_obs_names = np.asarray(base_adata.obs_names).astype(str)

            uce_name_to_idx = {name: i for i, name in enumerate(uce_obs_names)}
            adata_name_to_idx = {name: i for i, name in enumerate(adata_obs_names)}

            final_obs_names = [name for name in adata_obs_names if name in uce_name_to_idx]

            keep_idx = np.array(
                [adata_name_to_idx[name] for name in final_obs_names],
                dtype=int,
            )

            uce_idx = np.array(
                [uce_name_to_idx[name] for name in final_obs_names],
                dtype=int,
            )

            X_gene = X_gene[keep_idx]
            X_protein = X_protein[keep_idx]
            coords = coords[keep_idx]
            base_adata = base_adata[final_obs_names].copy()

            self.x_uce = torch.tensor(uce[uce_idx], dtype=torch.float32)

            print(f"[INFO] aligned UCE shape: {self.x_uce.shape}")

        # ---------- final assignment ----------
        self.x_gene = torch.tensor(X_gene, dtype=torch.float32)
        self.x_protein = torch.tensor(X_protein, dtype=torch.float32)
        self.coords = torch.tensor(coords, dtype=torch.float32)

        self.gene_names = list(base_adata.var_names)
        self.protein_names = protein_names

        # ---------- build graph from final aligned coords ----------
        edge_index = build_delaunay_edge_index(
            coords,
            add_self_loops=True,
        )

        self.edge_index = torch.from_numpy(edge_index).long()

        print(f"[INFO] coords shape: {self.coords.shape}")
        print(f"[INFO] edge_index shape: {self.edge_index.shape}")

        # ---------- final consistency check ----------
        print("[INFO] final dataset shapes:")
        print("  x_gene:", self.x_gene.shape)
        print("  x_protein:", self.x_protein.shape)

        if self.x_uce is not None:
            print("  x_uce:", self.x_uce.shape)

        print("  coords:", self.coords.shape)
        print("  edge_index:", self.edge_index.shape)

        n = self.x_gene.shape[0]

        assert self.x_protein.shape[0] == n, (
            f"x_protein rows mismatch: x_gene={n}, "
            f"x_protein={self.x_protein.shape[0]}"
        )

        assert self.coords.shape[0] == n, (
            f"coords rows mismatch: x_gene={n}, "
            f"coords={self.coords.shape[0]}"
        )

        if self.x_uce is not None:
            assert self.x_uce.shape[0] == n, (
                f"x_uce rows mismatch: x_gene={n}, "
                f"x_uce={self.x_uce.shape[0]}"
            )

        assert self.edge_index.shape[0] == 2, (
            f"edge_index should have shape [2, E], got {self.edge_index.shape}"
        )

        if self.edge_index.numel() > 0:
            assert int(self.edge_index.max()) < n, (
                f"edge_index contains node id >= n. "
                f"max={int(self.edge_index.max())}, n={n}"
            )

    def __len__(self):
        return self.x_gene.shape[0]

    def __getitem__(self, idx):
        item = {
            "x_gene": self.x_gene[idx],
            "x_protein": self.x_protein[idx],
            "node_idx": torch.tensor(idx, dtype=torch.long),
        }

        if self.x_uce is not None:
            item["x_uce"] = self.x_uce[idx]

        return item

def select_top_variable_genes_by_train(full_train_dataset, test_dataset, n_top_genes):
    """
    Select top variable genes using only the training slice,
    then apply the same gene subset to both train and test datasets.

    This should be called after train/test gene intersection
    and before train/val split.
    """
    if n_top_genes is None or n_top_genes <= 0:
        print("[INFO] Using all common genes. No top-variable-gene selection.")
        return full_train_dataset, test_dataset

    n_genes = full_train_dataset.x_gene.shape[1]

    if n_top_genes >= n_genes:
        print(
            f"[INFO] n_top_genes={n_top_genes} >= total genes={n_genes}. "
            "Using all genes."
        )
        return full_train_dataset, test_dataset

    # Use only train slice to compute variance
    X_train = full_train_dataset.x_gene.float()

    gene_var = torch.var(X_train, dim=0, unbiased=False)

    # top variable gene indices
    top_idx = torch.topk(gene_var, k=n_top_genes, largest=True).indices

    # Optional: sort by original gene order for cleaner bookkeeping
    top_idx = torch.sort(top_idx).values

    old_gene_names = list(full_train_dataset.gene_names)
    selected_gene_names = [old_gene_names[i] for i in top_idx.cpu().numpy()]

    full_train_dataset.x_gene = full_train_dataset.x_gene[:, top_idx]
    test_dataset.x_gene = test_dataset.x_gene[:, top_idx]

    full_train_dataset.gene_names = selected_gene_names
    test_dataset.gene_names = selected_gene_names

    print(
        f"[INFO] Selected top {n_top_genes} variable genes "
        f"from train slice only. "
        f"New gene_dim={full_train_dataset.x_gene.shape[1]}"
    )
    print(f"[INFO] Example selected genes: {selected_gene_names[:10]}")

    return full_train_dataset, test_dataset