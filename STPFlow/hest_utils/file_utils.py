import scanpy as sc
import h5py
import numpy as np
from pathlib import Path
from pathlib import Path
from typing import List, Dict

def build_gene_vocab(h5ad_paths: List[str], gene_key: str = None) -> Dict[str, int]:
    """
    从多个 h5ad 文件中提取所有基因名称，构建基因到ID的映射。
    包含特殊 token: <PAD>=0, <UNK>=1

    Args:
        h5ad_paths: h5ad 文件路径列表
        gene_key:   AnnData.var 中存储基因名称的列名，None 则使用 var_names

    Returns:
        gene2idx: 字典 {基因名: ID}
    """
    # ----- 1. 特殊标记 -----
    special_tokens = ['<PAD>', '<UNK>']
    gene2idx = {token: i for i, token in enumerate(special_tokens)}

    # ----- 2. 收集所有基因 -----
    all_genes = set()
    for path in h5ad_paths:
        adata = sc.read_h5ad(path, backed='r')  # 只读取 metadata
        if gene_key is not None:
            genes = adata.var[gene_key].astype(str).tolist()
        else:
            genes = adata.var_names.astype(str).tolist()
        all_genes.update(genes)
        adata.file.close()  # 关闭文件句柄

    # ----- 3. 按字母顺序分配 ID -----
    for gene in sorted(all_genes):
        if gene not in gene2idx:  # 避免重复
            gene2idx[gene] = len(gene2idx)

    print(f"构建完成基因词汇表，大小: {len(gene2idx)} (包含特殊标记)")
    return gene2idx
    
def save_hdf5(output_fpath, 
                  asset_dict, 
                  attr_dict= None, 
                  mode='a', 
                  auto_chunk = True,
                  chunk_size = None):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f: # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_: 
                    data_type = h5py.string_dtype(encoding='utf-8')
                if auto_chunk:
                    chunks = True # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(key, 
                                            shape=data_shape, 
                                            chunks=chunks,
                                            maxshape=(None,) + data_shape[1:],
                                            dtype=data_type)
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")
                
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val
        
        # if attr_dict is not None:
        #     for key, attr in attr_dict.items():
        #         if (key in asset_dict.keys()) and (len(asset_dict[key].attrs.keys())==0):
        #             for attr_key, attr_val in attr.items():
        #                 dset[key].attrs[attr_key] = attr_val
                
    return output_fpath
