# retreival-generation-system

## Code files
- main.py : main function for odqa/cqa chatbot 
- module.py : chatbot module classes 
- finetune.py : finetune function for t5/opt 
- model_utils.py : helper functions for module 
- data_utils.py : helper function/classes for training data 
## Scripts 
- push_all.sh : push codes on github 
- run_chatbot.sh : script for running chatbot  
    - For single turn QA, use `--task 'odqa'`, for Conversational QA, use `--task 'cqa'` in script.
    - To serve OPT as generator, use `--gen_model_path "facebook/opt-1.3b"` and `--model_name "opt"` ; to use Flan-T5, change them to `--gen_model_path "google/flan-t5-large"` and `--model_name t5`
- run_train_opt.sh : script for finetune opt 
- run_train_t5.sh : script for finetune t5
## Dataset 
- ece_rag_dataset_new : ECE120 course note dataset for DPR. 
- train_data : ECE120 finetune dataset