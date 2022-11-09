from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import GPT2Tokenizer, GPTJForCausalLM,OPTForCausalLM
from datasets import load_dataset, load_from_disk
import torch
from rouge import Rouge 
import csv 
import logging
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import queue
from transformers import T5Tokenizer, T5ForConditionalGeneration


'''
    Model Classes
'''
class rag_retreiver():
    def __init__(self,dataset_path, index_path,device):
        super(rag_retreiver,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="custom",
            passages_path=dataset_path,
            index_path=index_path,
        )
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True).to(self.device)

    def retreive(self,input_text:str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        # with tokenizer.as_target_tokenizer():
        #     targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # 1. Encode
        question_hidden_states = self.model.question_encoder(input_ids)[0]
        # 2. Retrieve
        docs_dict = self.retriever(input_ids.cpu().numpy(), question_hidden_states.cpu().detach().numpy(), return_tensors="pt")
        # docs_dict = self.retriever(input_ids, question_hidden_states, return_tensors="pt")
        doc_scores = torch.bmm(
            question_hidden_states.cpu().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)
        return docs_dict, doc_scores
    
    def retreive_psg(self,input_text:str):
        docs_dict, doc_scores = self.retreive(input_text)
        doc_ids = docs_dict['doc_ids'][0].tolist()
        num_docs = len(doc_ids)
        total_doc = []
        for idx in doc_ids:
            total_doc.append(self.dataset[idx]['text'])
        return total_doc,docs_dict,doc_scores 

class opt_model():
    def __init__(self,model_path,device):
        super(opt_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.gptj = OPTForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    def text_gen(self,input_text:str,max_len:int = 200):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.gptj.generate(**inputs,max_length = max_len,do_sample = True,early_stopping = True,temperature=0.8, top_p = 0.9)
        out_text = self.tokenizer.batch_decode(outputs)[0]
        return out_text

class seq2seq_model():
    def __init__(self,model_path,device):
        super(seq2seq_model,self).__init__()
        self.device = device 
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
    def generate(self,input_text:str,max_len:int = 100):
        input_ids = self.tokenizer(input_text,return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(input_ids,max_new_tokens = max_len)
        out_text = self.tokenizer.batch_decode(outputs,skip_special_tokens =True)[0]
        return out_text

class gpt_j():
    def __init__(self,model_path,device):
        super(gpt_j,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.gptj = GPTJForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    def text_gen(self,input_text:str,max_len:int = 200):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.gptj.generate(**inputs, pad_token_id = 50256,max_length = max_len,do_sample = True,early_stopping = True,temperature=0.8, top_p = 0.9)
        out_text = self.tokenizer.batch_decode(outputs)[0]
        return out_text
    
class benchmark_gptj():
    def __init__(self,model_path):
        super(benchmark_gptj,self).__init__()
        self.gptj = gpt_j(model_path)
        self.device = torch.device("cuda:0")
        self.dataset = load_dataset("coqa")
        
    def get_logger(self,filename, verbosity=1, name = None):
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
    
    def f_score(self,hypothesis,reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-l']['f']
    
    def eval(self,logger_path):
        logger = self.get_logger(logger_path)
        num_data = len(self.dataset['validation'])
        logger.info("Starting Evaluation")
        for pid in range(num_data):
            ans_list = []
            f_score_list = []
            context = self.dataset['validation'][pid]['story'].replace("\n"," ")
            n_word = len(context.split(" "))
            if(n_word>500):
                continue
            for num_his in range(len(self.dataset['validation'][pid]['questions'])-1):
                dialog_his = ""
                if(num_his!=0):
                    for i in range(num_his):
                        dialog_his = dialog_his + "\nquestion: "+self.dataset["validation"][pid]['questions'][i] + "\nanswer: " + self.dataset["validation"][pid]['answers']['input_text'][i]
                prompt = "Answer question from context\ncontext: "+context +  dialog_his +"\nquestion: " + self.dataset["validation"][pid]['questions'][num_his] + "\nanswer:"
                gt = self.dataset['validation'][pid]['answers']['input_text'][num_his]
                ans_text = self.gptj.text_gen(prompt,max_len=550)[0].split("\n")[num_his*2 +2 + 1]
                f_score = self.f_score(ans_text,gt)
                ans_list.append(ans_text)
                f_score_list.append(f_score)
            info_text = ""
            for i in range(len(ans_list)):
                info_text= info_text +  "QUESTION: " + self.dataset["validation"][pid]['questions'][i] + "\PREDICT:"+ ans_list[i] +"\nGROUNDTRUTH:" + self.dataset["validation"][pid]['answers']['input_text'][i] + "\nF-SCORE:" + str(f_score_list[i])+ "\n\n"
            info_text  = "\nCONTEXT:"+ context + "\n" + info_text 
            logger.info(info_text)
        return     
    
class qr_model():
    def __init__(self,device):
        super(qr_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained("castorini/t5-base-canard").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("castorini/t5-base-canard")
        self.sep = "|||"
    def qr(self,his_list,cur_q):
        prompt = ""
        for c in his_list:
            prompt += (c + self.sep)
        prompt += cur_q
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(inputs)
        out_text = self.tokenizer.batch_decode(outputs,skip_special_tokens = True)[0]
        return out_text