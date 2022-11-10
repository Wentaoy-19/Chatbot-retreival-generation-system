python finetune.py \
    --model_name "opt" \
    --dataset_path "/raid/projects/wentaoy4/save_dataset" \
    --model_path "/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee/" \
    --device "cuda:1" \
    --logger_path "/raid/projects/wentaoy4/log/opt_finetune_b128_e10_lr5e06.log" \
    --saved_model_path "/raid/projects/wentaoy4/model_weight/opt_finetune_b128_e10_lr5e06.pt" \
    --batch_size 1 \
    --outer_batch_size 128 \
    --epochs 10 \
    --lr 5e-6
    