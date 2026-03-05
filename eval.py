import os
import argparse
import gla_clip_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner
import time
from cfg import ModelConfig
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='SCLIP evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_coco_stuff164k.py')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='./show_dir/',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)   
    parser.add_argument('--work_dir', type=str, default='work_dir')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--CLIP_type', choices=['vanilla', 'MaskCLIP', 'SCLIP', 'ClearCLIP', 'CLIPTrase', 'NACLIP', 'ProxyCLIP', 'ProxyCLIP_KV_Expansion', 'GT'], default='vanilla')
    parser.add_argument('--lbl_size', type=int, default=28)

    # token norm
    parser.add_argument('--token_norm', action='store_true', default=False, help='Use token norm for feature fusion')
    
    # kv token extention setting
    parser.add_argument('--KV_token_extension', action='store_true', default=False, help='Remove identity token from KV Token Extension')
    parser.add_argument('--proxy_sim', action='store_true', default=False, help='Use ProxyCLIP with KV Token Extension. This is useful for ProxyCLIP with KV Token Extension')  # Use ProxyCLIP with KV Token Extension. This is useful for ProxyCLIP with KV Token Extension
    parser.add_argument('--smoothing', action='store_true', default=False, help='Use dynamic gamma for attention mechanism. This is useful for ProxyCLIP with KV Token Extension')  # Use dynamic zeta for multi-layer feature fusion mechanism. This is useful for ProxyCLIP with KV Token Extension
    parser.add_argument('--mini_iters', type=int, default=5, help='Number of mini iterations for ProxyCLIP with KV Token Extension')  # Number of mini iterations for ProxyCLIP with KV Token Extension
    parser.add_argument('--initial_crit_pos', type=float, default=0.55, help='Initial critical position for ProxyCLIP with KV Token Extension')  # Initial critical position for ProxyCLIP with KV Token Extension
    
    # VFM model
    parser.add_argument('--vfm_model',  type=str, default='dino', help='vfm model type')
    parser.add_argument('--checkpoint',  type=str, default=None, help='vfm model checkpoint')
    
    # dynamic_beta / gamma
    parser.add_argument('--dynamic_beta', action='store_true', default=False, help='Use dynamic beta for attention mechanism. This is useful for ProxyCLIP with KV Token Extension')  # Use dynamic beta for attention mechanism. This is useful for ProxyCLIP with KV Token Extension
    parser.add_argument('--dynamic_gamma', action='store_true', default=False, help='Use dynamic gamma for attention mechanism. This is useful for ProxyCLIP with KV Token Extension')  # Use dynamic gamma for attention mechanism. This is useful for ProxyCLIP with KV Token Extension
    parser.add_argument('--beta_alpha', type=float, default=0.2, help='Dynamic beta alpha for attention mechanism. This is useful for ProxyCLIP with KV Token Extension')  # Dynamic beta alpha for attention mechanism. This is useful for ProxyCLIP with KV Token Extension
    parser.add_argument('--gamma_alpha', type=float, default=1, help='Dynamic gamma alpha for attention mechanism. This is useful for ProxyCLIP with KV Token Extension')  # Dynamic gamma alpha for attention mechanism. This is useful for ProxyCLIP with KV Token Extension
    
    # visualization
    parser.add_argument('--attn_visualize', action='store_true', default=False)
    parser.add_argument('--pca_visualize', action='store_true', default=False)
    parser.add_argument('--pca_attn_visualize', action='store_true', default=False)
    parser.add_argument('--pca_val_visualize', action='store_true', default=False)
    parser.add_argument('--neg_visualize', action='store_true', default=False, help='Use negative token visualization')  # Use negative token visualization

    # metric
    parser.add_argument('--mask_metric', action='store_true', default=False, help='Use mask metric for evaluation')  # Mask metric (Average of Cosine Similarity)

    # etc    
    parser.add_argument('--num_cls_emb', type=int, default=1) 
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1.2, help='Beta hyperparameter for evaluation')
    parser.add_argument('--gamma', type=float, default=3, help='Gamma hyperparameter for evaluation')
    

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher
    
    model_cfg = ModelConfig(**cfg.model)
    model_cfg.CLIP_type = args.CLIP_type
    model_cfg.lbl_size = (args.lbl_size, args.lbl_size)
    model_cfg.gt_dir = cfg.test_dataloader.dataset.data_prefix.seg_map_path
    model_cfg.work_dir = args.work_dir
    
    # # token norm
    model_cfg.token_norm = args.token_norm
    
    # # kv token extention setting
    model_cfg.KV_token_extension = args.KV_token_extension
    
    # PPAP
    model_cfg.proxy_sim = args.proxy_sim
    model_cfg.smoothing = args.smoothing
    model_cfg.mini_iters = args.mini_iters
    model_cfg.initial_crit_pos = args.initial_crit_pos

    # vfm
    model_cfg.vfm_model = args.vfm_model
    model_cfg.checkpoint = args.checkpoint

    # dynamic beta / gamma
    model_cfg.dynamic_beta = args.dynamic_beta
    model_cfg.dynamic_gamma = args.dynamic_gamma
    model_cfg.beta_alpha = args.beta_alpha
    model_cfg.gamma_alpha = args.gamma_alpha
    cfg.model.model_cfg = model_cfg
    
    # trigger_visualization_hook(cfg, args)  

    runner = Runner.from_cfg(cfg)
    results = runner.test()    
    
    torch.cuda.empty_cache()
    results.update({'Model': cfg.model.model_type,
                    'CLIP': cfg.model.clip_type,
                    'VFM': cfg.model.vfm_model,
                    'Dataset': cfg.dataset_type})

    if runner.rank == 0:
        with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write('\n')
            f.write('Configuration: ' + args.config + '\n')
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    main()