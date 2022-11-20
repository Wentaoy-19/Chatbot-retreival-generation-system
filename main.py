from datasets import load_dataset, load_from_disk
import torch 
import logging
from module import * 
from model_utils import *
from reranker import *
from entity_tracker import *
        
"""
    Main Ret-Gen Model with odqa/cqa
"""   
class ret_gen_model():
    def __init__(self,model_name,dataset_path, index_path,gen_model_path,gen_cp_path,logger_path,device):
        super(ret_gen_model,self).__init__()
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        if(model_name == "opt"):
            self.gen_model = opt_model(gen_model_path,self.device)
        else:
            self.gen_model = seq2seq_model(gen_model_path,self.device)
        if(gen_cp_path != None):
            self.gen_model.load_checkpoint(gen_cp_path)
        self.retriever = rag_retreiver(dataset_path=dataset_path, index_path= index_path,device = self.device)
        self.qr_model = qr_model(self.device)
        self.re_ranker = re_ranker(self.device)
        self.entity_tracker = entity_tracker("turing machine")
        if(logger_path != None):
            self.logger = get_logger(logger_path)
        else:
            self.logger = None
        
        
    def get_logger(self,filename, verbosity=2, name = None):
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
    
    def gen_response_list(self,input_q,context_list):
        out_list = []
        for i in range(len(context_list)):
            out_ans = self.gen_model.answer_question(context_list[i].replace("\n"," "),input_q.replace("\n"," "))
            out_list.append(out_ans)
        return out_list 
    
    def show_list_result(self,user_utter):
        psg_list = self.ret_psg_list(user_utter)
        ans_list = self.gen_response_list(user_utter,psg_list)
        scr_list = self.re_ranker.rank(user_utter, ans_list)
        print("[PASSAGE]: \n")
        for i in range(len(psg_list)):
            print("-----Passage " + str(i) + "-----\n")
            print(psg_list[i] + "\n")
        print("[RESPONSE]: \n")
        for j in range(len(ans_list)):
            print("-----Answer " + str(j) + " Score: " + str(scr_list[j]) + "-----\n")
            print(ans_list[j] + "\n")
        print("[Best Response]: \n")
        # scr_list is a Tensor
        best_idx = torch.argmax(scr_list)
        print(ans_list[best_idx] + "\n")
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
                break 
            self.show_list_result(user_utter)
            
        print("[INFO] End Session\n")
        
    def cqa_chatbot(self):
        print("\n\n[INFO] Prototype of QA Chatbot system for ECE120\n\n")
        flag = 1
        while(flag):
            user_utter = input("[User Input]: ")
            if(user_utter == "quit"):
                flag = 0
                continue 
            user_utter, topic, history = self.entity_tracker.main(user_utter)
            print("[QUESTION REWRITE]: " + user_utter + "\n")
            print("[TOPIC]: "+ topic + "\n")
            psg = self.ret_psg(user_utter)
            out_ans = self.gen_model.answer_question(psg,user_utter)
            self.entity_tracker.answer_attach(out_ans)
            print("[PASSAGE]: \n" + psg + "\n")
            print("[RESPONSE]: \n" + out_ans + "\n")
            
        print("[INFO] End Session\n")

if __name__ == "__main__":
    args = main_arg_parse()
    my_chatbot = ret_gen_model(
        model_name=args.model_name,
        dataset_path = args.dataset_path,
        index_path = args.index_path,
        gen_model_path = args.gen_model_path,
        gen_cp_path = args.gen_cp_path,
        logger_path = args.logger_path,
        device = args.device
    )
    if(args.task == 'odqa'):
        my_chatbot.odqa_chatbot()
    elif(args.task == 'cqa'):
        my_chatbot.cqa_chatbot()
    
