import csv,os
from distutils import text_file
from sre_parse import Tokenizer 
from typing import List
from datasets import load_dataset,load_from_disk
from transformers import GPT2Tokenizer 
from huggingface_hub import snapshot_download
import json

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")

def download_model(checkpoint ='facebook/opt-13b', cache_dir = "./"):
    # checkpoint = 'facebook/opt-13b'
    weights_path = snapshot_download(checkpoint,cache_dir = cache_dir)

def convert_csv(txt_path,csv_file):
    text_file_list = os.listdir(txt_path)
    temp_csv = open(csv_file,mode="w")
    csv_writer = csv.writer(temp_csv,delimiter="$")
    for name in text_file_list:
        with open(txt_path + name) as read_file:
            title = " ".join(name.split("-")[4:]).split(".")[0]
            content = read_file.read()
            content = content.replace("\n"," ")
            total_data = [title,content]
            csv_writer.writerow(total_data)
    temp_csv.close()
    
def _convert_csv(csv_file):
    squad_dataset = load_dataset('squad')
    idx_set = []
    context_set = []
    title_set = []
    temp_csv = open(csv_file,mode="w")
    csv_writer = csv.writer(temp_csv,delimiter="@")
    
    for i in range(len(squad_dataset['validation'])):
        _list = squad_dataset['validation'][i]['context'].split(' ')[:10]
        _str = " ".join(_list)
        if(_str in idx_set):
            continue 
        else:
            idx_set.append(_str)
            context_set.append(squad_dataset['validation'][i]['context'])
            title_set.append(squad_dataset['validation'][i]['title'])
        
    for i in range(len(context_set)):
        total_data = [title_set[i], context_set[i]]
        csv_writer.writerow(total_data)
    temp_csv.close()

def _convert_cqa_csv(csv_file):
    squad_dataset = load_dataset('squad')
    temp_csv = open(csv_file,mode="w")
    csv_writer = csv.writer(temp_csv,delimiter="@")
    total_data = []
    
    for i in range(len(squad_dataset['validation'])):
        title = squad_dataset['validation'][i]['title']
        question = " Q:" + squad_dataset['validation'][i]['question']
        context = "context: "+squad_dataset['validation'][i]['context']
        answer = "A:" + squad_dataset['validation'][i]['answers']['text'][0]
        total_data = [title, context + question+answer]
        csv_writer.writerow(total_data)
    temp_csv.close()
    


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def split_documents_2(documents: dict) -> dict:
    """Split documents into passages"""
    total_data = []
    # titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                temp_dic = {'title': title if title is not None else "", 'text': passage}
                total_data.append(temp_dic)
                # titles.append(title if title is not None else "")
                # texts.append(passage)
    return {"data": total_data}

def split_documents(documents:dict):
    """Split documents into passages"""
    titles, texts, input_ids = [], [], []
    for title, text in zip(documents["title"], documents["text"]):
        titles.append(title if title is not None else "")
        texts.append(text)
        input_ids.append(tokenizer(text,return_tensors = "pt")["input_ids"].detach())
        
        # if text is not None:
        #     for passage in split_text(text):
        #         titles.append(title if title is not None else "")
        #         texts.append(passage)
        #         # input_ids.append(tokenizer(passage,return_tensors = "pt",truncation=True, padding = 'max_length', max_length = 800)["input_ids"].detach())
        #         input_ids.append(tokenizer(passage,return_tensors = "pt")["input_ids"].detach())

    return {"title": titles, "text": texts,"input_ids":input_ids}

def squad_to_csv(squad_dataset,csv_path = "/home/wentaoy4/lgm/squad_val_qa.csv"):
    total_line = len(squad_dataset['validation'])
    i = 0
    idx = 0
    title = ""
    pre_title = ""
    with open(csv_path,mode = 'w') as test_file:
        test_writer = csv.writer(test_file,delimiter='!')
        for i in range(total_line):
            c = squad_dataset['validation'][i]['context']
            q = squad_dataset['validation'][i]['question']
            a = squad_dataset['validation'][i]['answers']['text'][0]
            title = squad_dataset['validation'][i]['title']
            context = "%s Q:%sA:%s </s>" % (c,q, a)
            total_data = [title,context]
            test_writer.writerow(total_data)

def csv_to_dataset(csv_path ="/home/wentaoy4/lgm/squad_val_qa.csv",saved_path = "/home/wentaoy4/lgm/convert_dataset/squad_val_qa_text_dataset"):
    test_dataset = load_dataset(
            "csv", data_files=csv_path, split="train", delimiter="@", column_names=["title", "text"]
    )
    dataset = test_dataset.map(split_documents, batched=True,batch_size = 16, num_proc=1)
    dataset.save_to_disk(saved_path)

def csv_to_json(csv_path="/home/wentaoy4/lgm/ece120.csv" , saved_path = "/home/wentaoy4/lgm/data/json_data/ece120_note.json"):
    csv_dataset = load_dataset(
            "csv", data_files=csv_path, split="train", delimiter="$", column_names=["title", "text"]
    )  
    json_data = split_documents_2(csv_dataset)
    with open(saved_path,"w") as outfile:
        json.dump(json_data,outfile)


if __name__ =="__main__":
    # squad_dataset = load_dataset("squad")
    # squad_to_csv(squad_dataset,csv_path = "/home/wentaoy4/lgm/squad_qa_context.csv")
    # _convert_cqa_csv("../data/csv_data/squad_val_cqa.csv")
    csv_to_dataset(csv_path ="../data/csv_data/squad_val_cqa.csv",
                   saved_path = "../data/convert_dataset/squad_val_cqa_dataset")
    # convert_csv(txt_path="/home/wentaoy4/lgm/convert_txt/",csv_file="/home/wentaoy4/lgm/ece120.csv")
    # _convert_csv(csv_file = "../data/csv_data/squad_val.csv")

    
