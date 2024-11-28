#!/usr/bin/env bash         \


GPU_ID=$1
hidden_dim=512
LR=0.1 # 0.1
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

export CUDA_VISIBLE_DEVICES=${GPU_ID}


DIR=save/sd1.5_ddim_${STEPS}_${SIZE}_block_wise_bs_${BATCH_SIZE}_lr_${LR}_wd_${WEIGHT_DECAY}_${OPT}_${SCHDU}_STEP_${STEP_NUM}_seed_${SEED}_${attention_type}_attntype_${self_step_attn_module_type}_mlptype_${mlp_module_type}_loss_${CL_TYPE}_stopthreshold_${THRE}_dim_${hidden_dim}
model_args="--use_attn --num_inference_steps ${STEPS} --pretrained_model_name_or_path=sd1.5 --scheduler ddim --attention_type=${attention_type} --encoder_layer_num=${ENCODER_LAYER_NUM} --mlp_layer_num=${N_MLP_LAYER} --self_step_attn_module_type=${self_step_attn_module_type} --mlp_module_type=${mlp_module_type} --hidden_dim=${hidden_dim}"

accelerate launch train_sd15.py \
  ${model_args} --scale_lr --sparse_scale=1 --consistent_scale=1 --sparsity_early_stop --stop_threshold=${THRE} \
  --resolution ${SIZE} --validation_steps=10000 --seed=${SEED} --output_dir=${DIR} --cl_type=${CL_TYPE} --text_to_img --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=2 --learning_rate=${LR} \
  --sgd_weight_decay=${WEIGHT_DECAY} --lr_scheduler=${SCHDU} --lr_warmup_steps=10 --max_train_steps=${STEP_NUM} --gradient_checkpointing \
  2>&1 | tee ${DIR}/log.txt

wait

python generate_and_eval.py \
  ${model_args} --prune --dataset coco2017 --batch_size 4 --fid --prune_ratio 0.8 --pruner_model_path ${DIR}/checkpoint.pth \
  2>&1 | tee ${DIR}/eval_log.txt

