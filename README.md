# retreival-generation-system

## Code files
- `main.py` : main function for odqa/cqa chatbot 
- `module.py` : chatbot module classes 
- `finetune.py` : finetune function for t5/opt 
- `model_utils.py` : helper functions for module 
- `data_utils.py` : helper function/classes for training data 
- `reranker.py`: ranker for response quality, loaded from main_fn
- `entity_tracker.py`: entity tracker for dialog(in progress)
## Scripts 
- `push_all.sh` : push codes on github 
- `run_chatbot.sh` : script for running chatbot  
    - For single turn QA, use `--task 'odqa'`. For Conversational QA, use `--task 'cqa'` in script.
    - To serve OPT as generator, use `--gen_model_path "facebook/opt-1.3b"` and `--model_name "opt"` ; to use Flan-T5, change them to `--gen_model_path "google/flan-t5-large"` and `--model_name t5`
    - To load fine-tuned weight file, add `--gen_cp_path <path_to_weight>`. Without this line the pretrained weight of the model will be loaded.
- `run_train_opt.sh` : script for finetune opt 
    - Feel free to change: 
        - `--device`: The GPU# for training.
        - `--logger_path`: Path to save training log.
        - `--saved_model_path`: Path to save fine-tuned weight.
        - `--outer_batch_size`: "Batch size" for Gradient
        - `--dataset_path`: Path to converted training dataset
        accumulation.
        - `--epochs`: fine-tuning epochs 
        - `--lr`: learning rate 
    - **Since training data is very large, the Gradient accumulation is applied in the code. Stay** `--batch_size 1` **to avoid OUT_OF_MEMORY issue. Change** `--outer_batch_size` **to adjust the "batch size" for training.**
- `run_train_t5.sh` : script for finetune t5
    - Same as `run_train_opt.sh`
- `run_convert_dataset.sh`: script to convert dataset generated from GPT-3 to the format used for training
    - `--original_json_path`: Path to GPT-3 generated data. The default dataset is `GPT-3_semantic_search.json`.
    - `--converted_json_path`: Path to converted Json format of dataset. 
    - `--saved_dataset_path`: Path to save converted dataset
## Dataset 
- ece_rag_dataset_new : ECE120 course note dataset for DPR. 
- train_data : ECE120 finetune dataset, converted from GPT-3_semantic_search.json
- GPT-3_semantic_search.json: GPT-3 generated dataset for training. 