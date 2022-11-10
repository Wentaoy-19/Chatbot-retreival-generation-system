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
import numpy as np
from module import *
from data_utils import *
from model_utils import *


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
    saved_model_path,
    epoch
):
    losses = []
    for idx,data in enumerate(dataloader):
        input_ids = data['input_ids']
        labels = data['labels']
        loss = model.train_loss_ids(input_ids,labels)
        loss.backward()
        if(idx%outer_batch == 0 and idx!=0):
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
            print(f"Loss: {loss.detach().item()}")
            losses.append(loss.detach().item())
    logger.info(f"Epoch: {epoch}  Loss: {np.array(losses).mean()}")
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
    for epoch in range(epochs):
        train_one_epoch(model,dataloader,optimizer,logger,outer_batch_size,saved_model_path,epoch)
    return 


def main_opt_finetune(
    dataset_path,
    model_path,
    device,
    logger_path,
    saved_model_path,
    batch_size = 2,
    outer_batch_size = 4,
    epochs = 50, 
    lr = 1e-5
):
    train_device = device
    train_model = opt_model(model_path,train_device)
    train_dataset = opt_finetune_dataset(dataset_path,train_model.tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
    train_logger = get_logger(logger_path)
    train_logger.info("Start Training")
    _finetune(train_dataloader,train_model,epochs, lr, outer_batch_size,saved_model_path,train_logger)
    train_logger.info("Finish Training")
    return 

def main_t5_finetune(
    dataset_path,
    model_path,
    device,
    logger_path,
    saved_model_path,
    batch_size = 2,
    outer_batch_size = 4,
    epochs = 50,
    lr = 1e-5
):
    train_device = device
    train_model = seq2seq_model(model_path,train_device)
    train_dataset = t5_finetune_dataset(dataset_path,train_model.tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
    train_logger = get_logger(logger_path)
    train_logger.info("Start Training")
    _finetune(train_dataloader,train_model,epochs, lr, outer_batch_size,saved_model_path,train_logger)
    train_logger.info("Finish Training")
    return 
    
    

if __name__ == "__main__":
    args = train_arg_parse()
    if(args.model_name == 'opt'):
        main_opt_finetune(dataset_path=args.dataset_path,
                          model_path = args.model_path,
                          device = torch.device(args.device),
                          logger_path=args.logger_path,
                          saved_model_path=args.saved_model_path,
                          batch_size=args.batch_size,
                          outer_batch_size=args.outer_batch_size,
                          epochs = args.epochs,
                          lr = args.lr)
    elif(args.model_name == 't5'):
        main_t5_finetune(
            dataset_path=args.dataset_path,
            model_path= args.model_path,
            device= torch.device(args.device),
            logger_path=args.logger_path,
            saved_model_path=args.saved_model_path,
            batch_size = args.batch_size,
            outer_batch_size =args.outer_batch_size,
            epochs = args.epochs,
            lr = args.lr
            )
    