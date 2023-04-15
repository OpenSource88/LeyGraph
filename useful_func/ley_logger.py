import os
import datetime as dt


class Ley_Logger:
    def __init__(self, env):
        self.env = env
        self.log_root = os.path.join(os.path.dirname(__file__), '..', f'logs_{env.total_rank}')
        os.makedirs(self.log_root, exist_ok=True)
        self.log_fname = os.path.join(self.log_root, f'log_{env.model_name}_{env.time_num}.txt')

    def log(self, *args, oneline=False, rank=-1):
        if rank!=-1 and self.env.rank!=rank:
            return
        head = '%s [%1d] '%(dt.datetime.now().time(), self.env.rank)
        tail = '\r' if oneline else '\n'
        the_whole_line = head+' '.join(map(str, args))+tail
        print(the_whole_line, end='', flush=True)  
        with open(self.log_fname, 'a+') as f:
            print(the_whole_line, end='', file=f, flush=True)  


if __name__ == '__main__':
    pass

