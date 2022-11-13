    # --gen_model_path "facebook/opt-1.3b" \
    # --gen_cp_path "/home/wentaoy4/lgm/data/model_weight/opt_finetune_b128_e20_lr5e06.pt" \
    # --gen_model_path "google/flan-t5-large" \
    # --gen_cp_path "/home/wentaoy4/lgm/data/model_weight/t5_finetune_b128_e20_lr1e05.pt" \

python main.py \
    --model_name "opt" \
    --dataset_path "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset/" \
    --index_path "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset.faiss" \
    --gen_model_path "facebook/opt-1.3b" \
    --gen_cp_path "/home/wentaoy4/lgm/data/model_weight/opt_finetune_b128_e20_lr5e06.pt" \
    --device "cuda:2" \
  
