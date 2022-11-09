from rouge import Rouge 
import logging


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