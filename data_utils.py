# import csv,os
# from distutils import text_file
# from sre_parse import Tokenizer 
# from typing import List
# from datasets import load_dataset,load_from_disk
# from transformers import GPT2Tokenizer 
# from huggingface_hub import snapshot_download
# import json

import json 
from datasets import load_dataset,load_from_disk
import sys
from module import *
from huggingface_hub import snapshot_download
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch


# # tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")

# def download_model(checkpoint ='facebook/opt-13b', cache_dir = "./"):
#     # checkpoint = 'facebook/opt-13b'
#     weights_path = snapshot_download(checkpoint,cache_dir = cache_dir)

# def convert_csv(txt_path,csv_file):
#     text_file_list = os.listdir(txt_path)
#     temp_csv = open(csv_file,mode="w")
#     csv_writer = csv.writer(temp_csv,delimiter="$")
#     for name in text_file_list:
#         with open(txt_path + name) as read_file:
#             title = " ".join(name.split("-")[4:]).split(".")[0]
#             content = read_file.read()
#             content = content.replace("\n"," ")
#             total_data = [title,content]
#             csv_writer.writerow(total_data)
#     temp_csv.close()
    
# def _convert_csv(csv_file):
#     squad_dataset = load_dataset('squad')
#     idx_set = []
#     context_set = []
#     title_set = []
#     temp_csv = open(csv_file,mode="w")
#     csv_writer = csv.writer(temp_csv,delimiter="@")
    
#     for i in range(len(squad_dataset['validation'])):
#         _list = squad_dataset['validation'][i]['context'].split(' ')[:10]
#         _str = " ".join(_list)
#         if(_str in idx_set):
#             continue 
#         else:
#             idx_set.append(_str)
#             context_set.append(squad_dataset['validation'][i]['context'])
#             title_set.append(squad_dataset['validation'][i]['title'])
        
#     for i in range(len(context_set)):
#         total_data = [title_set[i], context_set[i]]
#         csv_writer.writerow(total_data)
#     temp_csv.close()

# def _convert_cqa_csv(csv_file):
#     squad_dataset = load_dataset('squad')
#     temp_csv = open(csv_file,mode="w")
#     csv_writer = csv.writer(temp_csv,delimiter="@")
#     total_data = []
    
#     for i in range(len(squad_dataset['validation'])):
#         title = squad_dataset['validation'][i]['title']
#         question = " Q:" + squad_dataset['validation'][i]['question']
#         context = "context: "+squad_dataset['validation'][i]['context']
#         answer = "A:" + squad_dataset['validation'][i]['answers']['text'][0]
#         total_data = [title, context + question+answer]
#         csv_writer.writerow(total_data)
#     temp_csv.close()
    


# def split_text(text: str, n=100, character=" ") -> List[str]:
#     """Split the text every ``n``-th occurrence of ``character``"""
#     text = text.split(character)
#     return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

# def split_documents_2(documents: dict) -> dict:
#     """Split documents into passages"""
#     total_data = []
#     # titles, texts = [], []
#     for title, text in zip(documents["title"], documents["text"]):
#         if text is not None:
#             for passage in split_text(text):
#                 temp_dic = {'title': title if title is not None else "", 'text': passage}
#                 total_data.append(temp_dic)
#                 # titles.append(title if title is not None else "")
#                 # texts.append(passage)
#     return {"data": total_data}

# def split_documents(documents:dict):
#     """Split documents into passages"""
#     titles, texts, input_ids = [], [], []
#     for title, text in zip(documents["title"], documents["text"]):
#         titles.append(title if title is not None else "")
#         texts.append(text)
#         input_ids.append(tokenizer(text,return_tensors = "pt")["input_ids"].detach())
        
#         # if text is not None:
#         #     for passage in split_text(text):
#         #         titles.append(title if title is not None else "")
#         #         texts.append(passage)
#         #         # input_ids.append(tokenizer(passage,return_tensors = "pt",truncation=True, padding = 'max_length', max_length = 800)["input_ids"].detach())
#         #         input_ids.append(tokenizer(passage,return_tensors = "pt")["input_ids"].detach())

#     return {"title": titles, "text": texts,"input_ids":input_ids}

# def squad_to_csv(squad_dataset,csv_path = "/home/wentaoy4/lgm/squad_val_qa.csv"):
#     total_line = len(squad_dataset['validation'])
#     i = 0
#     idx = 0
#     title = ""
#     pre_title = ""
#     with open(csv_path,mode = 'w') as test_file:
#         test_writer = csv.writer(test_file,delimiter='!')
#         for i in range(total_line):
#             c = squad_dataset['validation'][i]['context']
#             q = squad_dataset['validation'][i]['question']
#             a = squad_dataset['validation'][i]['answers']['text'][0]
#             title = squad_dataset['validation'][i]['title']
#             context = "%s Q:%sA:%s </s>" % (c,q, a)
#             total_data = [title,context]
#             test_writer.writerow(total_data)

# def csv_to_dataset(csv_path ="/home/wentaoy4/lgm/squad_val_qa.csv",saved_path = "/home/wentaoy4/lgm/convert_dataset/squad_val_qa_text_dataset"):
#     test_dataset = load_dataset(
#             "csv", data_files=csv_path, split="train", delimiter="@", column_names=["title", "text"]
#     )
#     dataset = test_dataset.map(split_documents, batched=True,batch_size = 16, num_proc=1)
#     dataset.save_to_disk(saved_path)

# def csv_to_json(csv_path="/home/wentaoy4/lgm/ece120.csv" , saved_path = "/home/wentaoy4/lgm/data/json_data/ece120_note.json"):
#     csv_dataset = load_dataset(
#             "csv", data_files=csv_path, split="train", delimiter="$", column_names=["title", "text"]
#     )  
#     json_data = split_documents_2(csv_dataset)
#     with open(saved_path,"w") as outfile:
#         json.dump(json_data,outfile)
        
        
        
##########

def json2dic(path:str):
    fp = open(path)
    json_data = json.load(fp)
    fp.close()
    return json_data

def convert_json_dataset(json_path,save_path):
    temp_dataset = load_dataset("json", data_files=json_path, split="train",field = "data")
    temp_dataset.save_to_disk(save_path)
    
def convert_dic_json(dic:dict,path):
    with open(path, "w") as outfile:
        json.dump(dic, outfile)

def load_converted_dataset(path:str):
    return load_from_disk(path)

def opt_json2dataset(original_path,saved_json_path, saved_dataset_path):
    temp = json2dic(original_path)
    temp = {"data":temp}
    convert_dic_json(temp,saved_json_path)
    convert_json_dataset(saved_json_path,saved_dataset_path)
    
def download_model(checkpoint ='facebook/opt-13b', cache_dir = "./"):
    weights_path = snapshot_download(checkpoint,cache_dir = cache_dir)   

class opt_finetune_dataset(Dataset):
    def __init__(self,converted_dataset,tokenizer):
        self.tokenizer = tokenizer
        self.converted_dataset = load_from_disk(converted_dataset)
        self.data_num = len(self.converted_dataset)
        self.data_stack, self.dataset = self.generate_final_data()
    def __len__(self):
        return self.data_num
    def __getitem__(self, idx):
        return {'input_ids': self.data_stack['input_ids'][idx],'labels':self.data_stack['labels'][idx]}
    def generate_prompt(self,context:str, question:str, answer:str):
        return "Answer question from context:\n" + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"+answer.replace("\n","")
    def trun_text(self,input_text:str, num_word:int = 512):
        text_list = input_text.split(" ")
        text_num = len(text_list)
        if(text_num<num_word):
            return input_text[:text_num] 
        else:
            return " ".join(text_list[:num_word])
    def generate_final_data(self):
        dc = DataCollatorForLanguageModeling(self.tokenizer,mlm = False)
        final = []
        for i in range(self.data_num):
            # context = self.converted_dataset[i]['textbook-paragraph']
            context = self.trun_text(self.converted_dataset[i]['textbook-paragraph'],200)
            question = self.converted_dataset[i]['GPT-3-Semantic-Search-Generations']['question']
            # answer = self.converted_dataset[i]['GPT-3-Semantic-Search-Generations']['answer']
            answer = self.trun_text(self.converted_dataset[i]['GPT-3-Semantic-Search-Generations']['answer'],50)
            prompt = self.generate_prompt(context,question,answer)
            prompt_ids = self.tokenizer(prompt,return_tensors = 'pt').input_ids[0]
            final.append(prompt_ids)
        return dc(final),final






# if __name__ =="__main__":
#     # squad_dataset = load_dataset("squad")
#     # squad_to_csv(squad_dataset,csv_path = "/home/wentaoy4/lgm/squad_qa_context.csv")
#     # _convert_cqa_csv("../data/csv_data/squad_val_cqa.csv")
#     csv_to_dataset(csv_path ="../data/csv_data/squad_val_cqa.csv",
#                    saved_path = "../data/convert_dataset/squad_val_cqa_dataset")
#     # convert_csv(txt_path="/home/wentaoy4/lgm/convert_txt/",csv_file="/home/wentaoy4/lgm/ece120.csv")
#     # _convert_csv(csv_file = "../data/csv_data/squad_val.csv")

    
