from datasets import load_dataset, load_from_disk
import torch 
import logging
from module import * 
from model_utils import *
# '''
#     Helper Functions/classes
# '''
        
"""
    Main Ret-Gen Model with odqa/cqa
"""   
class ret_gen_model():
    def __init__(self,dataset_path, index_path,device):
        super(ret_gen_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        # self.gen_model = opt_model("/home/wentaoy4/lgm/data/model_file/facebook--opt-6.7b.main.8dc17cdd7b9381612e631064e569f4142d776d88",self.device)
        # self.gen_model = gpt_j("/home/wentaoy4/lgm/data/model_file/EleutherAI--gpt-j-6B.main.918ad376364058dee23512629bc385380c98e57d/",self.device)
        self.gen_model = seq2seq_model("google/flan-t5-large",self.device)
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
            # out_ans = self.gen_model.text_gen(prompt,200).split("\n")[3]
            out_ans = self.gen_model.generate(prompt,200)
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
    # my_ret = ret_gen_model(dataset_path = "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset/", index_path = "/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset.faiss")
    # my_ret.get_result("/home/wentaoy4/lgm/qa.csv",delimiter = "?",logger_path = "/home/wentaoy4/lgm/data/logger/metrics_ret_gen.log")
    # my_ret.get_gpt_dataset_result(json_path = "/home/wentaoy4/UIUC_chatbot_data_generator/Fine_Tuned_Data.json",logger_path="/home/wentaoy4/lgm/data/logger/metrics_ret_gen_gptdata.log")
    # my_ret.get_gpt_dataset_result(json_path = "/home/wentaoy4/UIUC_chatbot_data_generator/GPT-3_paragraphs.json",logger_path="/home/wentaoy4/lgm/data/logger/metrics_gen_only.log")
   
    # my_benchmark = benchmark_gptj("/home/wentaoy4/lgm/data/model_file/EleutherAI--gpt-j-6B.main.918ad376364058dee23512629bc385380c98e57d/")
    # my_benchmark.eval("/home/wentaoy4/lgm/data/logger/gptj-coqa.log")
    my_device = torch.device("cuda:1")
    my_chatbot = ret_gen_model(dataset_path = "./dataset/ece_rag_dataset_new/squad-dataset/", index_path = "./dataset/ece_rag_dataset_new/squad-dataset.faiss",device=my_device)
    my_chatbot.odqa_chatbot()
    # my_chatbot.cqa_chatbot()