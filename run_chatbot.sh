python main.py \
    --model_name "t5" \
    --dataset_path "/home/haob2/taqa/retreival-generation-system/dataset/ece_rag_dataset_new/squad-dataset/" \
    --index_path "/home/haob2/taqa/retreival-generation-system/dataset/ece_rag_dataset_new/squad-dataset.faiss" \
    --gen_model_path "google/flan-t5-large" \
    --device "cuda:0"
    # --gen_cp_path "/home/wentaoy4/lgm/data/model_weight/opt_finetune_b128_e20_lr5e06.pt" \