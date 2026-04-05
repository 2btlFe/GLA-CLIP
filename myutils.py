import openpyxl
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image, resize
import math
import ipdb
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import os
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F
from PIL import Image


def dynamic_beta(x, alpha=0.2):
    return 1 + alpha * np.log1p(x)

def dynamic_gamma(x, alpha=1):
    return 1  +  alpha / (x)

palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def draw(img):
    ret = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        ret[img == cls] = color
    return ret


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)  # [1, C, 1, 1]
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, image):
        if image.dim() == 3:  # [C, H, W]
            mean = self.mean.view(-1, 1, 1).to(image.device)
            std = self.std.view(-1, 1, 1).to(image.device)
        elif image.dim() == 4:  # [B, C, H, W]
            mean = self.mean.to(image.device)
            std = self.std.to(image.device)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        return image * std + mean


def proxy_sim(ex_feats_grid, model_cfg, num_heads=12, scale=1, indices=None):
    sim = torch.bmm(ex_feats_grid, ex_feats_grid.transpose(1, 2))  # [1, N, N]
    
    for mi in range(model_cfg.mini_iters):
        mask_one = (sim > model_cfg.initial_crit_pos) # [1, N, N]
        mask_one = mask_one.to(ex_feats_grid.dtype) 
        gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

        if len(gone_proxy_idx) != 0:
            print(f"Warning: {len(gone_proxy_idx)} proxies are gone in mini_iter {mi}. Restoring them.")
            mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
        
        sum_proxy = torch.bmm(mask_one, ex_feats_grid)   # [1, N, C]
        count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
        proxy = sum_proxy / (count_proxy + 1e-6)
        proxy = F.normalize(proxy, dim=-1)
        sim = torch.bmm(proxy * scale, ex_feats_grid.transpose(1, 2))  # [1, N, N]
    
    return proxy, mask_one


class KV_Extension(nn.Module):
    def __init__(self):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    def forward(self, ex_feats_grid, num_heads=12, scale=1, 
                 beta=1.2, gamma=3.0, indices=None, v_ext=None, model_cfg=None):

        h_grids, w_grids, C, H, W = ex_feats_grid.shape
        B = h_grids * w_grids
        S = H * W

        ex_feats_grid = ex_feats_grid.permute(0, 1, 3, 4, 2).reshape(B, S, C)   
        ex_feats_grid = ex_feats_grid.reshape(1, B * S, C)                           
        ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)                   
        mask_one = None          
        attn_output = []
        
        v_ext = v_ext.reshape(B*num_heads, model_cfg.token_size[0], model_cfg.token_size[1], -1) \
                    .permute(0, 3, 1, 2).contiguous()                                       # [B*H, Dh, Th, Tw]
        v_ext = F.interpolate(v_ext, size=(H, W), mode='bilinear', align_corners=False)  # [B*H, Dh, H, W]
        
        Dh = v_ext.shape[1]
        v_ext = v_ext.view(B, num_heads, Dh, H, W).flatten(3)                    # [B,H,Dh,S]
        v_ext = v_ext.permute(1, 0, 3, 2).reshape(num_heads, B * S, Dh)          # [H,B*S,Dh]

        key = ex_feats_grid   
        if getattr(model_cfg, 'proxy_sim', False) or getattr(model_cfg, 'dynamic_gamma', False):
            if getattr(model_cfg, 'smoothing', False):
                indices = indices.flatten()
                indices_mask = indices.unsqueeze(0) == indices.unsqueeze(1)  # [N, N]
                device = key.device
                r = torch.arange(H, device=device)
                c = torch.arange(W, device=device)
                rr, cc = torch.meshgrid(r, c, indexing='ij')  # [H, W], [H, W]
                coords = torch.stack([rr.flatten(), cc.flatten()], dim=1)  # [H*W, 2]
                
                delta = coords.unsqueeze(1) - coords.unsqueeze(0)  # [H*W, H*W, 2]
                window_mask = (delta.abs().max(dim=-1)[0] <= 1) & (delta.abs().sum(dim=-1) > 0)
                neighbor_mask = torch.block_diag(*([window_mask] * (h_grids * w_grids))).to(device)  # [N, N]
                
                cand_mask = indices_mask | neighbor_mask   # [1, N, N]
                cand_mask.fill_diagonal_(False)
                
                count = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                avg_feats = (cand_mask.half() @ key) / count
                
                proxy = avg_feats
            else:
                proxy = key.clone()

            attn_weights = torch.bmm(F.normalize(proxy, dim=-1), key.transpose(1, 2))  # [1, N, N]
            
            if model_cfg.mini_iters == 0:
                mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                mask_one = mask_one.to(key.dtype) 
            else:
                for mi in range(model_cfg.mini_iters):
                    mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                    mask_one = mask_one.to(key.dtype) 
                    gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

                    if len(gone_proxy_idx) != 0:
                        mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
                    
                    proxy = torch.bmm(mask_one, key)   # [1, N, C]
                    count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
                    proxy = proxy / (count_proxy + 1e-6)
                    proxy = F.normalize(proxy, dim=-1)
                    attn_weights = torch.bmm(proxy, key.transpose(1, 2))  # [1, N, N]
            
            if getattr(model_cfg, 'proxy_sim', False) == False:
                attn_weights = torch.bmm(key, key.transpose(1, 2))  # [1, N, N]
            
        else:
            attn_weights = torch.bmm(key, key.transpose(1, 2))  # [1, B*S, B*S]

        attn_weights = attn_weights.reshape(B, S, B*S) * scale

        if getattr(model_cfg, 'token_norm', False):
            beta_ = model_cfg.beta
            gamma_ = model_cfg.gamma
            sim_mean = attn_weights.mean(dim=-1, keepdim=True)            # [B,S,1]

            is_mask_cnt_exist = False
            if getattr(model_cfg, 'dynamic_gamma', False):
                if isinstance(mask_one, torch.Tensor):
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1)      # [B,S,1]
                    is_mask_cnt_exist = True
                    gamma_ = dynamic_gamma(x=mask_cnt, alpha=model_cfg.gamma_alpha)
                else:
                    gamma_ = model_cfg.gamma

            if getattr(model_cfg, 'dynamic_beta', False):                
                
                if is_mask_cnt_exist == False:
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1) 
                beta_ = dynamic_beta(x=B, alpha=model_cfg.beta_alpha)
                
            sim_mean[sim_mean < 0] = 0
            attn_weights = (attn_weights - sim_mean * beta_) * gamma_
        else:
            attn_weights = (attn_weights - attn_weights.mean() * model_cfg.beta) * model_cfg.gamma
        
        max_per_row = attn_weights.max(dim=-1, keepdim=True).values       # [B,S,1]
        cutting_hp = torch.as_tensor(getattr(model_cfg, 'cutting_hp', 0.0),
                                        device=attn_weights.device, dtype=attn_weights.dtype)
        dynamic_cutting_hp = torch.minimum(max_per_row, cutting_hp)       # [B,S,1]
        attn_weights.masked_fill_(attn_weights < dynamic_cutting_hp, float('-inf'))
        
        attn_weights = F.softmax(attn_weights / model_cfg.temperature, dim=-1)  # [B,S,B*S]

        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, neginf=float('-inf'))

        attn_output = torch.einsum('bsn,hnd->hsbd', attn_weights, v_ext)  # [H, S, B, Dh]
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(S, B, -1)

        return attn_output  # [num_tokens=S, B, C]


class KV_Extension_ClearCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    def forward(self, ex_feats_grid, num_heads=12, scale=1, lbl_grid=None,
                 beta=1.2, gamma=3.0, indices=None, v_ext=None, model_cfg=None):

        h_grids, w_grids, S, C  = ex_feats_grid.shape
        B = h_grids * w_grids
        H = W = int(S ** 0.5)
        
        ex_feats_grid = ex_feats_grid.reshape(B, S, C)   
        ex_feats_grid = ex_feats_grid.reshape(1, B * S, C)            
            
        mask_one = None        
        attn_output = []

        v_ext = v_ext.reshape(B*num_heads, model_cfg.token_size[0], model_cfg.token_size[1], -1) \
                    .permute(0, 3, 1, 2).contiguous()                                       
        v_ext = F.interpolate(v_ext, size=(H, W), mode='bilinear', align_corners=False)  
        Dh = v_ext.shape[1]
        
        v_ext = v_ext.view(B, num_heads, Dh, H, W).flatten(3)                    # [B,H,Dh,S]
        v_ext = v_ext.permute(1, 0, 3, 2).reshape(num_heads, B * S, Dh)          # [H,B*S,Dh]

        if getattr(model_cfg, 'proxy_sim', False) or getattr(model_cfg, 'dynamic_gamma', False):
            # 1) indices mask
            indices = indices.flatten()
            indices_mask = indices.unsqueeze(0) == indices.unsqueeze(1)  # [N, N]
            
            # 2) Neighbor mask
            device = ex_feats_grid.device
            r = torch.arange(H, device=device)
            c = torch.arange(W, device=device)
            rr, cc = torch.meshgrid(r, c, indexing='ij')  # [H, W], [H, W]
            coords = torch.stack([rr.flatten(), cc.flatten()], dim=1)  # [H*W, 2]
            
            delta = coords.unsqueeze(1) - coords.unsqueeze(0)  # [H*W, H*W, 2]
            window_mask = (delta.abs().max(dim=-1)[0] <= 1) & (delta.abs().sum(dim=-1) > 0)  

            neighbor_mask = torch.block_diag(*([window_mask] * (h_grids * w_grids))).to(device)  # [N, N]            
            cand_mask = indices_mask | neighbor_mask   # [1, N, N]
            
            count = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            avg_feats = (cand_mask.half() @ ex_feats_grid) / count
            proxy = avg_feats

            ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)
            key = ex_feats_grid
            attn_weights = torch.bmm(F.normalize(proxy, dim=-1), key.transpose(1, 2))  # [1, N, N]
            
            if model_cfg.mini_iters == 0:
                mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                mask_one = mask_one.to(key.dtype) 
            else:
                for mi in range(model_cfg.mini_iters):
                    mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                    mask_one = mask_one.to(key.dtype) 
                    gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

                    if len(gone_proxy_idx) != 0:
                        mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
                    
                    proxy = torch.bmm(mask_one, key)   # [1, N, C]
                    count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
                    proxy = proxy / (count_proxy + 1e-6)
                    proxy = F.normalize(proxy, dim=-1)
                    attn_weights = torch.bmm(proxy, key.transpose(1, 2))  # [1, N, N]
        else:
            ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)
            key = ex_feats_grid
            attn_weights = torch.bmm(key, key.transpose(1, 2))

        attn_weights = attn_weights.reshape(B, S, B*S) * scale

        if getattr(model_cfg, 'token_norm', False):
            beta_ = model_cfg.beta
            gamma_ = model_cfg.gamma
            sim_mean = attn_weights.mean(dim=-1, keepdim=True)            # [B,S,1]

            if getattr(model_cfg, 'dynamic_gamma', False):
                if isinstance(mask_one, torch.Tensor):
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1)      # [B,S,1]
                    gamma_ = dynamic_gamma(x=mask_cnt, alpha=model_cfg.gamma_alpha)
                else:
                    gamma_ = model_cfg.gamma

            if getattr(model_cfg, 'dynamic_beta', False):                
                beta_ = dynamic_beta(total_token=B*S, high_conf_token=mask_cnt, alpha=model_cfg.beta_alpha)                

            sim_mean[sim_mean < 0] = 0   
            attn_weights = (attn_weights - sim_mean * beta_) * gamma_
        else:
            attn_weights = (attn_weights - attn_weights.mean() * model_cfg.beta) * model_cfg.gamma
        
        max_per_row = attn_weights.max(dim=-1, keepdim=True).values       # [B,S,1]
        cutting_hp = torch.as_tensor(getattr(model_cfg, 'cutting_hp', 0.0),
                                        device=attn_weights.device, dtype=attn_weights.dtype)
        dynamic_cutting_hp = torch.minimum(max_per_row, cutting_hp)       # [B,S,1]
        attn_weights.masked_fill_(attn_weights < dynamic_cutting_hp, float('-inf'))

        attn_weights = F.softmax(attn_weights / model_cfg.temperature, dim=-1)  # [B,S,B*S]

        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, neginf=float('-inf'))

        attn_output = torch.einsum('bsn,hnd->hsbd', attn_weights, v_ext)  # [H, S, B, Dh]
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(S, B, -1)

        return attn_output  


