python finetune.py \
    --model_name "opt" \
    --dataset_path "/raid/projects/wentaoy4/save_dataset" \
    --model_path "/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee/" \
    --device "cuda:0" \
    --logger_path "/raid/projects/wentaoy4/log/opt_train.log" \
    --saved_model_path "/raid/projects/wentaoy4/model_weight/opt_finetune.pt" \
    --batch_size 1 \
    --outer_batch_size 32 \
    --epochs 10 \
    --lr 1e-5
    