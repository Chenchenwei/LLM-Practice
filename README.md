多机多卡采用accelerate config获取配置文件 default_config.yaml
运行命令
experiment=recall_test
accelerate launch --config_file default_config.yaml run_clm.py \
    --model_name_or_path ${model_path} \
    --train_file 'data/search_chat/dev.json' \
    --validation_file 'data/search_chat/dev.json' \
    --per_device_train_batch_size 8 \
    --cutoff_len 128 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --preprocessing_num_workers 8 \
    --prompt_model zeroPrompt \
    --peft \
    --output_dir model/baichuan-v2-7b-$experiment > search_$experiment.log

compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
