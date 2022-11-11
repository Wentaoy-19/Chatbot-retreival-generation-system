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
    def __init__(self,model_name,dataset_path, index_path,gen_model_path,gen_cp_path,logger_path,device):
        super(ret_gen_model,self).__init__()
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        # self.gen_model = seq2seq_model(gen_model_path,self.device)
        if(model_name == "opt"):
            self.gen_model = opt_model(gen_model_path,self.device)
        else:
            self.gen_model = seq2seq_model(gen_model_path,self.device)
        if(gen_cp_path != None):
            self.gen_model.load_checkpoint(gen_cp_path)
        self.retriever = rag_retreiver(dataset_path=dataset_path, index_path= index_path,device = self.device)
        self.qr_model = qr_model(self.device)
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
    
    def gen_response(self,input_q,context):
        prompt = "Answer question from context\nContext: "+context +"\nQuestion: " + input_q + "\nAnswer:"
        out_ans = self.gen_model.text_gen(prompt,250).split("\n")[3]
        return out_ans
    
    def gen_response_list(self,input_q,context_list):
        out_list = []
        for i in range(len(context_list)):
            # prompt = "Answer question from context\nContext: "+context_list[i].replace("\n"," ") +"\nQuestion: " + input_q + "\nAnswer:"
            # # out_ans = self.gen_model.text_gen(prompt,200).split("\n")[3]
            # out_ans = self.gen_model.generate(prompt,200)
            # out_list.append(out_ans)
            out_ans = self.gen_model.answer_question(context_list[i].replace("\n"," "),input_q.replace("\n"," "))
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
    my_chatbot.odqa_chatbot()
    # my_device = torch.device("cuda:1")
    # my_chatbot = ret_gen_model(model_name = "t5",
    #                            dataset_path = "./dataset/ece_rag_dataset_new/squad-dataset/", 
    #                            index_path = "./dataset/ece_rag_dataset_new/squad-dataset.faiss",
    #                            gen_model_path = "/raid/projects/wentaoy4/model_file/models--google--flan-t5-large/snapshots/f5b192378f2e16fb61561ee418736e8c6841c4c8/",
    #                            # gen_model_path = "/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee/",
    #                            gen_cp_path = "/raid/projects/wentaoy4/model_weight/t5_finetune_b128_e10_lr5e06.pt",
    #                            # gen_cp_path = "/raid/projects/wentaoy4/model_weight/opt_finetune_b128_e10_lr5e06.pt",
    #                            # gen_cp_path = None,
    #                            # logger_path = "/raid/projects/wentaoy4/log/chatbot_opt_finetune.log",
    #                            logger_path = None,
    #                            device=my_device)
    my_chatbot.odqa_chatbot()
    # my_chatbot.cqa_chatbot()