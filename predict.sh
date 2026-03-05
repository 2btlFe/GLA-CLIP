#!/bin/bash
initial_crit_pos=0.6
mini_iters=2

beta_alpha=0.3
gamma_alpha=30
gpu=1

for Dataset in voc21
do
        timestamp=$(date +%Y%m%d_%H%M%S)
        config=configs/cfg_${Dataset}.py
        CLIP_type=ProxyCLIP
        vfm_model=dino
        project_name="Reproduce"
        
        experiment_name="${Dataset}_${CLIP_type}_Default"
        work_dir="${project_name}/${experiment_name}_${timestamp}/${Dataset}"
        CUDA_VISIBLE_DEVICES=${gpu} python eval.py \
                --config ${config} \
                --work_dir ${work_dir} \
                --show_dir ${work_dir}/visualize \
                --CLIP_type ${CLIP_type}
done
