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

'''
    Helper Functions/classes
'''
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

def f_score(hypothesis,reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']

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


class his_queue():
    def __init__(self,size):
        super(his_queue,self).__init__()
        self.maxsize = size 
        self.q = [0 for _ in range(self.maxsize)]
        self.num = 0
    def put(self,conv):
        if(self.num == self.maxsize):
            for i in range(self.num-1):
                self.q[i] = self.q[i+1]
            self.q[self.num-1] = conv 
        else:
            self.q[self.num] = conv
            self.num +=1 
    def get_list(self):
        ret_list = []
        for i in range(self.num):
            ret_list.append(self.q[i][0])
            ret_list.append(self.q[i][1])
        return ret_list
    def clear(self):
        self.num = 0
        return 

'''
    Model Classes
'''
class rag_retreiver(torch.nn.Module):
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

class opt_model(torch.nn.Module):
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

class gpt_j(torch.nn.Module):
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
    
class qr_model(torch.nn.Module):
    def __init__(self,device):
        super(qr_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
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
  
        
"""
    Main Ret-Gen Model with odqa/cqa
"""   
class ret_gen_model(torch.nn.Module):
    def __init__(self,dataset_path, index_path,device):
        super(ret_gen_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        # self.gen_model = opt_model("/home/wentaoy4/lgm/data/model_file/facebook--opt-6.7b.main.8dc17cdd7b9381612e631064e569f4142d776d88",self.device)
        self.gen_model = gpt_j("/home/wentaoy4/lgm/data/model_file/EleutherAI--gpt-j-6B.main.918ad376364058dee23512629bc385380c98e57d/",self.device)
        self.retriever = rag_retreiver(dataset_path=dataset_path, index_path= index_path,device = self.device)
        self.qr_model = qr_model(self.device)
        
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
    
    def ret_psg(self,input_q):
        psg,_,_ = self.retriever.retreive_psg(input_q)
        return psg[0].replace("\n"," ")
    
    def ret_psg_list(self,input_q):
        psgs,_,_ = self.retriever.retreive_psg(input_q)
        return psgs
    
    def gen_response(self,input_q,context):
        prompt = "Answer question from context\nContext: "+context +"\nQuestion: " + input_q + "\nAnswer:"
        out_ans = self.gen_model.text_gen(prompt,250).split("\n")[3]
        return out_ans
    
    def gen_response_list(self,input_q,context_list):
        out_list = []
        for i in range(len(context_list)):
            prompt = "Answer question from context\nContext: "+context_list[i].replace("\n"," ") +"\nQuestion: " + input_q + "\nAnswer:"
            out_ans = self.gen_model.text_gen(prompt,200).split("\n")[3]
            out_list.append(out_ans)
        return out_list 
    
    def show_single_result(self,user_utter):
        psg = self.ret_psg(user_utter)
        out_ans = self.gen_response(user_utter,psg)
        print("[PASSAGE]: \n" + psg + "\n")
        print("[RESPONSE]: \n" + out_ans + "\n")
        return 
    
    def show_list_result(self,user_utter):
        psg_list = self.ret_psg_list(user_utter)
        ans_list = self.gen_response_list(user_utter,psg_list)
        print("[PASSAGE]: \n")
        for i in range(len(psg_list)):
            print(psg_list[i] + "\n")
        print("[RESPONSE]: \n")
        for j in range(len(ans_list)):
            print(ans_list[j] + "\n")
        return 
    
    def qr(self,question,queue:his_queue):
        history_list = queue.get_list()
        qr_q = self.qr_model.qr(history_list,question)
        return qr_q
    
    def odqa_chatbot(self):
        print("\n\n[INFO] Prototype of QA Chatbot system for ECE120\n\n")
        flag = 1
        while(flag):
            user_utter = input("[User Input]: ")
            if(user_utter == "quit"):
                flag = 0
                continue 
            self.show_list_result(user_utter)
            
        print("[INFO] End Session\n")
        
    def cqa_chatbot(self):
        print("\n\n[INFO] Prototype of QA Chatbot system for ECE120\n\n")
        flag = 1
        w = 5
        history_q = his_queue(size = w)
        while(flag):
            user_utter = input("[User Input]: ")
            if(user_utter == "quit"):
                flag = 0
                continue 
            if(user_utter == "clear"):
                history_q.clear()
                continue
            if(history_q.num != 0):
                user_utter = self.qr(user_utter,history_q)
            print("[QUESTION REWRITE]: " + user_utter + "\n")
            psg = self.ret_psg(user_utter)
            out_ans = self.gen_response(user_utter,psg)
            history_q.put((user_utter,out_ans))
            print("[PASSAGE]: \n" + psg + "\n")
            print("[RESPONSE]: \n" + out_ans + "\n")
            
        print("[INFO] End Session\n")




if __name__ == "__main__":
    my_device = torch.device("cuda:1")
    # my_ret = ret_gen_model(dataset_path = "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset/", index_path = "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset.faiss")
    # my_ret.get_result("/home/wentaoy4/lgm/qa.csv",delimiter = "?",logger_path = "/home/wentaoy4/lgm/data/logger/metrics_ret_gen.log")
    # my_ret.get_gpt_dataset_result(json_path = "/home/wentaoy4/UIUC_chatbot_data_generator/Fine_Tuned_Data.json",logger_path="/home/wentaoy4/lgm/data/logger/metrics_ret_gen_gptdata.log")
    # my_ret.get_gpt_dataset_result(json_path = "/home/wentaoy4/UIUC_chatbot_data_generator/GPT-3_paragraphs.json",logger_path="/home/wentaoy4/lgm/data/logger/metrics_gen_only.log")
   
    # my_benchmark = benchmark_gptj("/home/wentaoy4/lgm/data/model_file/EleutherAI--gpt-j-6B.main.918ad376364058dee23512629bc385380c98e57d/")
    # my_benchmark.eval("/home/wentaoy4/lgm/data/logger/gptj-coqa.log")
    
    my_chatbot = ret_gen_model(dataset_path = "./dataset/ece_rag_dataset_new/squad-dataset/", index_path = "./dataset/ece_rag_dataset_new/squad-dataset.faiss",device=my_device)
    # my_chatbot.odqa_chatbot()
    my_chatbot.cqa_chatbot()