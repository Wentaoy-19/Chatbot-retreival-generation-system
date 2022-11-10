from transformers import GPT2Tokenizer,OPTConfig,AutoConfig, AutoModelForCausalLM, OPTForCausalLM, Trainer, AdamW,get_linear_schedule_with_warmup
import torch
from datasets import load_dataset,load_from_disk,Features,Value
from transformers import AutoModelWithLMHead, AutoTokenizer
import random 
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
from accelerate import init_empty_weights,load_checkpoint_and_dispatch,Accelerator
import accelerate 
import logging 
import argparse

def get_logger(filename, verbosity=1, name = None):
    level_dict = {0: logging.DEBUG, 1:logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename,"w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger 

# def add_arg_parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--family', type=str, default='张',help='姓')
#     parser.add_argument('--name', type=str, default='三', help='名')
#     args = parser.parse_args()
#     return args 
def load_without_pretrain(config_path):
    model = OPTForCausalLM.from_pretrained(config_path)
    model.train()
    max_mem = 18000*1024*1024
    device_map = accelerate.infer_auto_device_map(
        model, 
        max_memory={0: max_mem,1:max_mem,2:max_mem,3:max_mem},
        no_split_module_classes=["OPTDecoderLayer"], 
        dtype='float16'
    )
    model = accelerate.dispatch_model(model,device_map = device_map,offload_dir = '/raid/projects/wentaoy4/model_cache',offload_buffers =False)
    return model

def load_with_pretrain(config_path, weight_path):
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config)
    cp = torch.load(weight_path)
    # print(cp)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(cp)  
    model.train() 

    max_mem = 1000*1024*1024
    device_map = accelerate.infer_auto_device_map(
        model, 
        max_memory={0: max_mem,1:max_mem,2:max_mem,3:max_mem},
        no_split_module_classes=["OPTDecoderLayer"], 
        dtype='float16'
    )
    # model = accelerate.load_checkpoint_in_model(model,weight_path,device_map = device_map)
    model = accelerate.dispatch_model(model,device_map = device_map,offload_dir = '/raid/projects/wentaoy4/model_cache',offload_buffers =False)
    # model = load_checkpoint_and_dispatch(model, weight_path, device_map=device_map, no_split_module_classes=["GPTJBlock"])
    return model 

# def model_dispatch(weights_path):
#     model = OPTForCausalLM.from_pretrained(weights_path)
#     model.train()
#     max_mem = 1000*1024*1024
#     device_map = accelerate.infer_auto_device_map(
#         model, 
#         max_memory={0: max_mem,1:max_mem,2:max_mem,3:max_mem},
#         no_split_module_classes=["OPTDecoderLayer"], 
#         dtype='float16'
#     )
#     model = accelerate.dispatch_model(model,device_map = device_map,offload_dir = '/raid/projects/wentaoy4/model_cache',offload_buffers =False)
#     return model

def _single_gpu_train(dataloader, 
          model,
          device = torch.device("cuda:1"),
          epochs = 5, 
          lr = 5e-6, 
          batch_size = 16, 
          num_warmup_steps = 100, 
          out_model_path = "/home/wentaoy4/lgm/opt_models/my_opt1.pt"):
    
    logger = get_logger('../data/log/squad_val_1.log')
    # model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(),lr = lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=-1)
    loss = -1
    batch_count_tot = 1
    logger.info("Start Training !")
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch}, Loss: {loss}")
        for idx, data in enumerate(dataloader):
            _data = data['input_ids'][0].to(device)
            input_tensor = _data
            label = _data
            outputs = model(input_tensor,labels = label)
            loss = outputs.loss
            # loss = torch.mean(loss)
            loss.backward()

            if(batch_count_tot % batch_size == 0):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                print(loss)
                torch.save(
                    model.state_dict(),
                    out_model_path,
                )

            batch_count_tot += 1 
            input_tensor = None
            label = None 
            _data = None  
            outputs = None 
            loss = loss.detach()
        torch.save(
                model.state_dict(),
                out_model_path,
                )
    
    logger.info("Finish Training !")
    return model 

def _multi_gpu_train(dataloader, 
          model,
          device = torch.device("cuda:0"),
          epochs = 5, 
          lr = 2e-5, 
          batch_loop = 16, 
          out_model_path =  "../data/weight_data/temp.pt",
          logger_path = "../data/log/opt.log"):
    
    logger = get_logger(logger_path)
    optimizer = AdamW(model.parameters(),lr = lr)
    loss = -1
    batch_count_tot = 1
    logger.info("Start Training !")
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch}, Loss: {loss}")
        for idx, data in enumerate(dataloader):
            _data = data['input_ids'][0].to(device)
            input_tensor = _data
            label = _data
            outputs = model(input_tensor,labels = label)
            loss = outputs.loss 
            loss = torch.mean(loss)
            print("loss:",loss)
            loss.backward()

            if(batch_count_tot % batch_loop == 0):
                input_tensor = None 
                label = None 
                _data = None 
                outputs = None 
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                print(loss)
                torch.save(
                    model.state_dict(),
                    out_model_path,
                )
                # return 

            batch_count_tot += 1 
            input_tensor = None
            label = None 
            _data = None 
            outputs = None 
            loss = loss.detach()  
            torch.cuda.empty_cache()
    
    torch.save(
        model.state_dict(),
        out_model_path,
    )

    
    logger.info("Finish Training !")
    return model 

def main_multi_gpu_train():
    device = torch.device("cuda:0")
    dataset = load_from_disk("../data/convert_dataset/squad_val_qa_dataset")  # to reload the dataset
    dataset.set_format(type = 'torch',columns=['input_ids'])
    train_loader = DataLoader(dataset,batch_size= 1, shuffle = True)
    
    # config_path = '/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee'
    config_path = '/raid/projects/wentaoy4/model_file/models--facebook--opt-30b/snapshots/ac6427b800b92360fbb7dcab3155c03e5dc1273b'
    weight_path = '/raid/projects/wentaoy4/lgm/data/weight_data/temp.pt'
    # opt_model = load_with_pretrain(config_path = config_path, weight_path = weight_path)
    opt_model = load_without_pretrain(config_path = config_path)
 
    # cp = torch.load("./opt_models/opt_squad_val_qa1.pt")
    # opt_model.load_state_dict(cp)  
    _multi_gpu_train(dataloader=train_loader, 
          model = opt_model,
          device = device,
          epochs = 20, 
          lr = 1e-4, 
          batch_loop = 2, 
          out_model_path = "../data/weight_data/temp.pt")
    
def main_single_gpu_train():
    device = torch.device("cuda:0")
    dataset = load_from_disk("../data/convert_dataset/squad_val_cqa_dataset")  # to reload the dataset
    dataset.set_format(type = 'torch',columns=['input_ids'])
    train_loader = DataLoader(dataset,batch_size= 1,shuffle = True)
    config_path = '/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee'
    opt_model = OPTForCausalLM.from_pretrained(config_path).to(device)

    cp = torch.load("../data/weight_data/squad_val_cqa1.pt")
    opt_model.load_state_dict(cp)   

    _single_gpu_train(dataloader= train_loader,lr = 1e-5,device = device,batch_size = 512,model = opt_model,epochs= 20,out_model_path = "../data/weight_data/squad_val_cqa2.pt")
    
if __name__ == "__main__":
    # main_multi_gpu_train()
    main_single_gpu_train()
