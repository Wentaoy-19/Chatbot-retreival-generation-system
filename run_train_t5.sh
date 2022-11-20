python finetune.py \
    --model_name "t5" \
    --dataset_path "./dataset/train_data" \
    --model_path "google/flan-t5-large" \
    --device "cuda:2" \
    --logger_path "../t5_finetune_b32_e20_lr1e05.log" \
    --saved_model_path "../t5_finetune_b32_e20_lr1e05.pt" \
    --batch_size 1 \
    --outer_batch_size 32 \
    --epochs 20 \
    --lr 1e-5