#!/usr/bin/env bash         \

GPU_ID=$1
hidden_dim=512
LR=0.1 
attention_type="pre_self_step-post_mlp"  
SIZE=512
SEED=42
CL_TYPE=ssim
BATCH_SIZE=1
STEP_NUM=500
N_MLP_LAYER=3
ENCODER_LAYER_NUM=1
WEIGHT_DECAY=0.0001
mlp_module_type="block_wise"
self_step_attn_module_type="model_wise"
STEPS=50
THRE=0.2
OPT="sgd"
SCHDU="cosine" 
RATIO=0.8

export CUDA_VISIBLE_DEVICES=${GPU_ID}

model_args="--use_attn --num_inference_steps ${STEPS} --pretrained_model_name_or_path=sd1.5 --scheduler plms --attention_type=${attention_type} --encoder_layer_num=${ENCODER_LAYER_NUM} --mlp_layer_num=${N_MLP_LAYER} --self_step_attn_module_type=${self_step_attn_module_type} --mlp_module_type=${mlp_module_type} --hidden_dim=${hidden_dim}"

python generate_and_eval.py \
    ${model_args} --prune --dataset coco2017 --batch_size 4 --prune_ratio ${RATIO} \
    2>&1 | tee save/eval_plms.txt



