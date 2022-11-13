python finetune.py \
    --model_name "opt" \
    --dataset_path "./dataset/train_data" \
    --model_path "facebook/opt-1.3b" \
    --device "cuda:1" \
    --logger_path "../opt_finetune_b128_e20_lr5e06.log" \
    --saved_model_path "../opt_finetune_b128_e20_lr5e06.pt" \
    --batch_size 1 \
    --outer_batch_size 128 \
    --epochs 20 \
    --lr 5e-6
    