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
#
from module import *
from data_utils import *


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

    
    
def train_one_epoch(model,
    dataloader,
    optimizer,
    logger,
    outer_batch,
    saved_model_path
):
    for idx,data in enumerate(dataloader):
        input_ids = data['input_ids']
        labels = data['labels']
        loss = model.train_loss_ids(input_ids,labels)
        loss.backward()
        if(idx%outer_batch == 0 and idx!=0):
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
            logger.info("Loss: ",loss.item())
    model.save_checkpoint(saved_model_path)
    return 
        
            

def _finetune(
    dataloader,
    model,
    epochs,
    lr,
    outer_batch_size,
    saved_model_path,
    logger
):
    
    model.model.train()
    optimizer = AdamW(model.model.parameters(),lr = lr)
    for epocsh in range(epochs):
        train_one_epoch(model,dataloader,optimizer,logger,outer_batch_size,saved_model_path)
    return 


def opt_finetune_main(
    dataset_path,
    model_path,
    device,
    logger_path,
    batch_size = 2,
    outer_batch_size = 4,
    epochs = 50, 
    lr = 1e-5
):
    train_device = device
    train_model = opt_model(model_path,train_device)
    train_dataset = opt_finetune_dataset(dataset_path)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
    train_logger = get_logger(logger_path)
    train_logger.info("Start Training")
    _finetune(train_dataloader,train_model,epochs, lr, outer_batch_size,train_logger)
    train_logger.info("Finish Training")
    return 
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
# def _single_gpu_train(dataloader, 
#           model,
#           device = torch.device("cuda:1"),
#           epochs = 5, 
#           lr = 5e-6, 
#           batch_size = 16, 
#           num_warmup_steps = 100, 
#           out_model_path = "/home/wentaoy4/lgm/opt_models/my_opt1.pt"):
    
#     logger = get_logger('../data/log/squad_val_1.log')
#     # model = model.to(device)
#     model.train()
#     optimizer = AdamW(model.parameters(),lr = lr)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=-1)
#     loss = -1
#     batch_count_tot = 1
#     logger.info("Start Training !")
#     for epoch in range(epochs):
#         logger.info(f"Training epoch {epoch}, Loss: {loss}")
#         for idx, data in enumerate(dataloader):
#             _data = data['input_ids'][0].to(device)
#             input_tensor = _data
#             label = _data
#             outputs = model(input_tensor,labels = label)
#             loss = outputs.loss
#             # loss = torch.mean(loss)
#             loss.backward()

#             if(batch_count_tot % batch_size == 0):
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 model.zero_grad()
#                 print(loss)
#                 torch.save(
#                     model.state_dict(),
#                     out_model_path,
#                 )

#             batch_count_tot += 1 
#             input_tensor = None
#             label = None 
#             _data = None  
#             outputs = None 
#             loss = loss.detach()
#         torch.save(
#                 model.state_dict(),
#                 out_model_path,
#                 )
    
#     logger.info("Finish Training !")
#     return model 