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

    # Label & Query
    lbl: Optional[torch.Tensor] = None
    num_cls_emb: int = 1
    remove_id: bool = False
    iteration: int = 0

    # Grid
    h_grids: int = 3
    w_grids: int = 7
    grid_y: Optional[torch.Tensor] = None
    grid_x: Optional[torch.Tensor] = None

    # Oracle
    oracle_test: Optional[torch.Tensor] = None

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
    KV_broad_extension: bool = False
    KV_MultiRes_extension: bool = False

    only_overlapped_kv_extension: bool = False
    identity_remove: bool = False
    non_overlapped_cherry_pick: bool = False
    replace_inter_with_intra_attn: bool = False
    inter_intra_cutting_hp: bool = False
    delta_sum: float = 0.0
    changed_element: Optional[Union[List, torch.Tensor]] = None    
    temperature: float = 1.0
    cutting_hp: float = 0.0
    dynamic_threshold: bool = False
    dynamic_threshold_alpha: float = 0.15
    remove_high_norm_patches: bool = False
    global_prototype: bool = False
    add_high_confidence: bool = False

    # Threshold
    use_ori_mean: bool = False

    # Proxy Similarity
    proxy_sim: bool = False
    smoothing: bool = False
    ppap: bool = False
    mini_iters: int = 5
    initial_crit_pos: float = 0.55
    initial_crit_amb: float = 0.15
    sigma_pos: float = 3
    sigma_amb: float = 3
    remove_anomaly: bool = False
    anomaly_threshold: float = 0.5
    adaptive_sharpening: bool = False

    # prev_feat
    prev_feat: bool = False
    prev_blk_idx: int = 9
    last_block: int = -1

    # Kmeans Token Extension
    clustering: bool = False
    cluster_k: int = 784

    # Attention Bias
    use_alibi: bool = False
    slope_hp: float = 50.0
    
    # Average Query
    q_k_avg: bool = False

    # Positional Embedding
    positional_embedding_remove: bool = False
    pe_crop: bool = False

    # whole logits
    whole_img_logit: bool = False
    global_rate: float = 0.4

    # Fusion
    forward_to_fusion: bool = False
    fusion_weight: float = 0.1
    inter_temperature: float = 0.5
    intermediate_kv_extension: bool = False
    start_block: int = 0
    nblock: int = -1

    # Visualization
    grid_visualize: bool = False
    attn_visualize: bool = False
    intermediate_kv_attn_visualize: bool = False
    pca_visualize: bool = False
    pca_attn_visualize: bool = False
    pca_val_visualize: bool = False
    neg_visualize: bool = False
    visualize_crop: bool = False
    before_value_visualize: bool = False
    
    # Plot Histogram
    plot_histograms: bool = False

    # Metric
    mask_metric: bool = False
    feat_inter_disc: bool = False
    feat_intra_disc: bool = False
    class_variance: bool = False
    class_entropy: bool = False
    bf_score: bool = False

    # Optional Input
    indices: Optional[torch.Tensor] = None
    return_crop_images: bool = False
    img_name: Optional[str] = None

    