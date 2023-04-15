import argparse
import datetime
import GPUtil
import numpy as np
import os
from tqdm import tqdm
import sys
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import random
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from useful_func import ley_env, ley_logger
from src import ley_agg, ley_sage_conv, ley_gin_conv, ley_module, origin_module
from src.run import *

from torch.nn import Sequential, Linear, ReLU


def main():
    parser = argparse.ArgumentParser(description="multi-gpu")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-port', '--port', default=27027, type=int,
                        help='the port use for gradient_synchronization')
    parser.add_argument('--epochs', default=2, type=int, metavar='N')
    parser.add_argument("--dataset", type=str, default='reddit',
                        choices=['papers100M', 'products', 'reddit'])
    parser.add_argument("--num-neighbors", nargs='+', type=int, default=[25, 10],
                         help="the sampled neighbors for each layer.")
    parser.add_argument("--model-num", type=int, default=2,
                        help="how many model train in leygraph. 0,1: means not leygraph, one model train")
    parser.add_argument("--user-defined-cp", type=bool, default=0,
                        help="user defines shared computataion with 2 models training")
    parser.add_argument("--model1", type=str, default='sage',
                        choices=['gat', 'gcn', 'gin', 'sage'])
    parser.add_argument("--batchsize1", type=int, default=1024,
                        help="the batch size of model 1.")
    parser.add_argument("--model2", type=str, default='gin',
                        choices=['gat', 'gcn', 'gin', 'sage'])
    parser.add_argument("--batchsize2", type=int, default=1024,
                        help="the batch size of model 2.")
    parser.add_argument("--model3", type=str, default='gcn',
                        choices=['gat', 'gcn', 'gin', 'sage'])
    parser.add_argument("--model4", type=str, default='gat',
                        choices=['gat', 'gcn', 'gin', 'sage'])
    parser.add_argument("--hidden", type=int, default=256,
                        help="hidden feature size.")
    parser.add_argument("--print-time", type=bool, default=1,
                        help='document time of each stage.')
    parser.add_argument("--sample-worker", type=int, default=4,
                        help="the number of sample worker for each gpu.")
    parser.add_argument("--data-dir", type=str, default='/home/data/gnndata',
                        help="the directory of datasets")
    args = parser.parse_args()  

    args.out = f'{args.dataset}_{args.model_num}_{args.model1}_gpu{args.gpus}_batch{args.batchsize1}_dim{args.hidden}.log'

    #########################################################
    world_size = args.gpus * args.nodes
    print(f'Let\'s use {args.nodes} nodes, each with {args.gpus} GPUs')
    print(f'Which means total {world_size} GPUs')

    os.environ['MASTER_ADDR'] = '127.0.0.1'  #local IP       #
    os.environ['MASTER_PORT'] = str(args.port)                      #
    #########################################################

    data_dir = args.data_dir
    if args.dataset == 'reddit':
        dataset = Reddit(f'{data_dir}/reddit')
    elif args.dataset in ['products', 'papers100M']:
        transform = None
        dataset = PygNodePropPredDataset(
            name=f'ogbn-{args.dataset}', root=data_dir, transform=transform)
    else:
        assert 0, f'Unknown dataset {args.dataset}'


    #########################################################
    mp.spawn(run, args=(world_size, dataset, args), nprocs=args.gpus, join=True)
    #########################################################

if __name__ == '__main__':
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("LeyGraph: hello, no bug until now!")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    main()

