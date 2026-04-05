from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch

@dataclass
class ModelConfig:
    # Basic
    type: str = 'ProxyCLIPSegmentation'
    clip_type: str = 'openai'
    model_type: str = 'ViT-B/16'
    CLIP_type: str = 'ProxyCLIP'
    vfm_model: str = 'dino'
    checkpoint: Optional[str] = None
    device: str = "cuda"
    scale: int = 336

    # Hyperparameters
    beta: float = 1.2
    gamma: float = 3.0
    scale: int = 336
    name_path: Optional[str] = None
    prob_thd: float = 0.0
    bg_idx: int = 0
    logit_scale: float = 40
    gt_dir: Optional[str] = None
    work_dir: Optional[str] = None

    # Input settings
    lbl_size: Tuple[int, int] = (14, 14)
    h_feat: int = 56
    w_feat: int = 112
    img_h: int = 448
    img_w: int = 896
    token_size: Tuple[int, int] = (16, 16)
    n_patches: Tuple[int, int] = (14, 14)
    
    # Sliding window
    slide_stride: int = 112
    slide_crop: int = 224

    # Grid
    h_grids: int = 3
    w_grids: int = 7
    grid_y: Optional[torch.Tensor] = None
    grid_x: Optional[torch.Tensor] = None
    
    # Token & Feature
    token_norm: bool = False
    cossim_thres: bool = False
    ex_feats: Optional[torch.Tensor] = None
    ex_feats_broad: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    feature_broad: Optional[torch.Tensor] = None
    unnorm_img: Optional[torch.Tensor] = None
    unnorm_img_broad: Optional[torch.Tensor] = None
    
    # BBox
    bbox_list: Optional[List] = None
    bbox_broad_list: Optional[List] = None

    # KV token extension Options
    KV_token_extension: bool = False
    temperature: float = 1.0
    cutting_hp: float = 0.0

    # Threshold
    use_ori_mean: bool = False

    # Proxy Similarity
    proxy_sim: bool = False
    smoothing: bool = False
    mini_iters: int = 5
    initial_crit_pos: float = 0.55
    remove_anomaly: bool = False
    
    # Kmeans Token Extension
    clustering: bool = False
    cluster_k: int = 784

    # Optional Input
    indices: Optional[torch.Tensor] = None
    return_crop_images: bool = False
    img_name: Optional[str] = None
    num_cls_emb: int = 1
    