from rouge import Rouge 
import logging
import argparse


def train_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default = 'opt')
    parser.add_argument('--dataset_path', type=str, default='/raid/projects/wentaoy4/save_dataset')
    parser.add_argument('--model_path', type=str, default='/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee/')
    parser.add_argument('--device',type=str,default = 'cuda:0')
    parser.add_argument('--logger_path',type=str,default = '/raid/projects/wentaoy4/log/temp.log')
    parser.add_argument('--saved_model_path',type=str,default = '/raid/projects/wentaoy4/model_weight/opt_temp.pt')
    parser.add_argument('--batch_size',type=int,default = 2)
    parser.add_argument('--outer_batch_size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default = 1e-5)
    args = parser.parse_args()
    return args 

def main_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type = str,default = 't5')
    parser.add_argument('--dataset_path',type = str, default = '/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset/')
    parser.add_argument('--index_path',type= str,default = '/home/wentaoy4/lgm/data/convert_dataset/ece_rag_dataset_new/squad-dataset.faiss' )
    parser.add_argument('--gen_model_path',type = str, default = 'facebook/opt-1.3b')
    parser.add_argument('--gen_cp_path',type = str,default = None)
    parser.add_argument('--logger_path',type = str, default = None)
    parser.add_argument('--device',type = str,default = 'cuda:0')
    parser.add_argument('--task',type=str,default ='odqa')
    args = parser.parse_args()
    return args
    

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


# dialog history queue 
# each item: (q,a)
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