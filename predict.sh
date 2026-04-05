#!/bin/bash
initial_crit_pos=0.6
mini_iters=2

beta_alpha=0.3
gamma_alpha=30
gpu=0

for Dataset in voc21 context60 coco_object city_scapes voc20 context59 coco_stuff164k ade20k
do
        timestamp=$(date +%Y%m%d_%H%M%S)
        config=configs/cfg_${Dataset}.py
        CLIP_type=ProxyCLIP
        work_dir=output/GLA-CLIP/${Dataset}
        
        CUDA_VISIBLE_DEVICES=${gpu} python eval.py \
                --config ${config} \
                --work_dir ${work_dir} \
                --show_dir ${work_dir}/visualize \
                --CLIP_type ${CLIP_type} \
                --token_norm \
                --KV_token_extension \
                --proxy_sim \
                --mini_iters ${mini_iters} \
                --dynamic_beta \
                --beta_alpha ${beta_alpha} \
                --dynamic_gamma \
                --gamma_alpha ${gamma_alpha}
done

# - Method 1. KV extension
# KV_token_extension

# - Method 2. Proxy Anchor based Attention Stabilization
# proxy_sim
# mini_iters

# - Method 3. Dynamic Normalization
# token_norm
# dynamic_beta
# beta_alpha
# dynamic_gamma
# gamma_alpha