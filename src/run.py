import argparse
import datetime
import numpy as np
import os
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
from src import ley_agg, ley_sage_conv, ley_gin_conv, ley_module, origin_module, sample_par, model_par
from src.leygraph_shared_in import LeyGraph_Shared_In
from src.leygraph_shared_cp import LeyGraph_Shared_Cp

from torch.nn import Sequential, Linear, ReLU

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run(rank, world_size, dataset, args):
    ############################################################
    rank_ley = args.nr * args.gpus + rank  
    print(f'Total_rank {rank_ley} >> local_rank {rank} : I run')
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        #init_method='tcp://127.0.0.1:27027',                                   
    	world_size=world_size,                              
    	rank=rank_ley                                               
    )                                                          
    ############################################################
  
    #1.Env
    yy_env = ley_env.Ley_Env()
    yy_env.total_rank = world_size
    yy_env.rank = rank_ley
    yy_env.time_num = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    yy_env.model_name=args.model1

    #2.Logger basic information
    yy_logger = ley_logger.Ley_Logger(yy_env)
    if rank==0: #local_rank=0
        yy_logger.log(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        yy_logger.log(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        if args.model_num==2:
            yy_logger.log(f'Two models joint-training. model 0: {args.model1}, model 1: {args.model2}')
        elif args.model_num==3:
            yy_logger.log(f'Three models joint-training. model 0: {args.model1}, model 1: {args.model2}, model 2: {args.model3}')
        elif args.model_num==4:
            yy_logger.log(f'Four models joint-training. model 0: {args.model1}, model 1: {args.model2}, model 2: {args.model3}, model 4: {args.model4}')
        else:
            yy_logger.log(f'One model training. model 0: {args.model1}')
        yy_logger.log(f'node{args.nodes}_gpu{args.gpus}_dataset[{args.dataset}]_batch{args.batchsize1}_dim{args.hidden}_epochs{args.epochs}')
        if args.print_time:    
            yy_logger.log(f'Printing time of each part of Training')
        yy_logger.log("<<<<<<<<<<<<<<<<<<<<<<<<<<")
        yy_logger.log("<<<<<<<<<<<<<<<<<<<<<<<<<<")

    #3.Dataset
    if args.dataset == 'reddit':
        data = dataset[0]
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0]) 
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        x, y = data.x, data.y.squeeze() 
    elif args.dataset in ['products', 'papers100M']:
        data = dataset[0]
        if args.dataset == 'papers100M':
            rowptr, col, _ = torch.load('6666.pt')
            data.edge_index = SparseTensor(rowptr=rowptr, col=col)
        else:
            data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        
        '''
        #when processing papers100M for the first time
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        torch.save(data.edge_index.csr(), '6666.pt')
        '''
        split_idx = dataset.get_idx_split() 
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        full_data_cuda = False
        x, y = data.x, data.y.squeeze()
    else:
        full_data_cuda = False
        assert 0
        
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank_ley]
   
    #4. Shared_Cp
    model_list = []
    cp_exist = 0
    if args.model_num==2:
        if args.user_defined_cp:
            M1 = model_par.Model_Par(args.model1, dataset.num_features, args.hidden, dataset.num_classes)
            M2 = model_par.Model_Par(args.model2, dataset.num_features, args.hidden, dataset.num_classes)
            mM1, mM2, cp_exist, sh_cp = LeyGraph_Shared_Cp(M1, M2, 1)
            sh_cp = sh_cp.to(rank)
            model_list.append(sh_cp)
            mM1 = mM1.to(rank)
            mM2 = mM2.to(rank)
            mM1 = DistributedDataParallel(mM1, device_ids=[rank])
            mM2 = DistributedDataParallel(mM2, device_ids=[rank])
            model_list.append(mM1)
            model_list.append(mM2)
        else: #args.user_defined==0
            M1 = model_par.Model_Par(args.model1, dataset.num_features, args.hidden, dataset.num_classes)
            M2 = model_par.Model_Par(args.model2, dataset.num_features, args.hidden, dataset.num_classes)
            mM1, mM2, cp_exist, sh_cp = LeyGraph_Shared_Cp(M1, M2, 0)
            if cp_exist:
                sh_cp = sh_cp.to(rank)
                model_list.append(sh_cp)
            mM1 = mM1.to(rank)
            mM2 = mM2.to(rank)
            mM1 = DistributedDataParallel(mM1, device_ids=[rank])
            mM2 = DistributedDataParallel(mM2, device_ids=[rank])
            model_list.append(mM1)
            model_list.append(mM2)
    elif args.model_num==3: 
        mM1 = origin_module.Origin_Module(args.model1, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM2 = origin_module.Origin_Module(args.model2, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM3 = origin_module.Origin_Module(args.model3, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM1 = DistributedDataParallel(mM1, device_ids=[rank])
        mM2 = DistributedDataParallel(mM2, device_ids=[rank])
        mM3 = DistributedDataParallel(mM3, device_ids=[rank])
        model_list.append(mM1)
        model_list.append(mM2)
        model_list.append(mM3)
    elif args.model_num==4: 
        mM1 = origin_module.Origin_Module(args.model1, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM2 = origin_module.Origin_Module(args.model2, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM3 = origin_module.Origin_Module(args.model3, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM4 = origin_module.Origin_Module(args.model4, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM1 = DistributedDataParallel(mM1, device_ids=[rank])
        mM2 = DistributedDataParallel(mM2, device_ids=[rank])
        mM3 = DistributedDataParallel(mM3, device_ids=[rank])
        mM4 = DistributedDataParallel(mM4, device_ids=[rank])
        model_list.append(mM1)
        model_list.append(mM2)
        model_list.append(mM3)
        model_list.append(mM4)
    else: #1 model training
        mM1 = origin_module.Origin_Module(args.model1, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
        mM1 = DistributedDataParallel(mM1, device_ids=[rank])
        model_list.append(mM1)
        
            

    #5. Shared_In
    tiny_exist = 1
    if args.model_num==2:
        sp1 = sample_par.Sample_Par(args.batchsize1, args.dataset, args.num_neighbors)
        sp2 = sample_par.Sample_Par(args.batchsize2, args.dataset, args.num_neighbors)
        sp1, sp2, tiny_exist = LeyGraph_Shared_In(sp1, sp2)
    else:
        sp1 = sample_par.Sample_Par(args.batchsize1, args.dataset, args.num_neighbors)
        
    if tiny_exist:
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                       sizes=sp1.sampler_setting, batch_size=sp1.batch_size, return_e_id=False,
                                       shuffle=True, num_workers=args.sample_worker)
    else:
        raise Exception("not same dataset or sampling setting")

    print_time = args.print_time
    set_random_seed(12345)

    optimizer_list = []
    for i in range(args.model_num):
        if cp_exist:
            tmp_optimizer = torch.optim.Adam(model_list[i+1].parameters(), lr=0.01)
            optimizer_list.append(tmp_optimizer)
        else:
            tmp_optimizer = torch.optim.Adam(model_list[i].parameters(), lr=0.01)
            optimizer_list.append(tmp_optimizer)
                
    full_data_cuda = False
    if full_data_cuda:
        x, y = x.to(rank), y.to(rank)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # start to run 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    yy_logger.log(f'Total_rank {rank_ley} >> local_rank {rank} : I\'m ready, finally start to train the model.')

    for epoch in range(1, args.epochs+1): 

        for i in range(args.model_num):
            if cp_exist:
                model_list[i+1].train()
            else:
                model_list[i].train()
        
        if rank==0:
            yy_logger.log(f'--------------Epoch {epoch}---------------')

        if print_time:
            sample_time, h2d_time, forward_time, backward_time = [], [], [], []
            torch.cuda.synchronize()   
            tic = datetime.datetime.now()

        count = 0

        for batch_size, n_id, adjs in train_loader: 
            count += 1
            if batch_size == 1: continue

            if print_time:
                torch.cuda.synchronize()   
                toc = datetime.datetime.now()
                sample_time.append(toc-tic) # sample_time 
                tic = datetime.datetime.now()

            for adj in adjs:
                adj.adj_t.storage._value = None 

            adjs = [adj.to(rank) for adj in adjs]
            batch_x = x[n_id]
            batch_y = y[n_id[:batch_size]].squeeze().long()
            if not full_data_cuda:
                batch_x = batch_x.to(rank)
                batch_y = batch_y.to(rank)

            if print_time:
                torch.cuda.synchronize()
                toc = datetime.datetime.now()
                h2d_time.append(toc-tic)    # h2d_time 
                tic = datetime.datetime.now()
            
            if cp_exist:
                cp_out = model_list[0](batch_x, adjs)

            out_list = []
            loss_list = []
            for i in range(args.model_num):
                if cp_exist:
                    out_list.append(model_list[i+1](batch_x, adjs, cp_out))
                    loss_list.append(F.nll_loss(out_list[i], batch_y))
                else:
                    out_list.append(model_list[i](batch_x, adjs))
                    loss_list.append(F.nll_loss(out_list[i], batch_y))
                
            if print_time:
                torch.cuda.synchronize()
                toc = datetime.datetime.now()
                forward_time.append(toc-tic)    # forward_time
                tic = datetime.datetime.now()
            
            if args.model_num == 2 and (sp1.num != 1 or sp2.num!=1):
                loss_list[0].backward()
                if count%sp1.num==0:
                    optimizer_list[0].step()
                    optimizer_list[0].zero_grad()
                loss_list[1].backward()
                if count%sp2.num==0:
                    optimizer_list[1].step()
                    optimizer_list[1].zero_grad()
            else:
                for i in range(args.model_num):
                    loss_list[i].backward()
                    optimizer_list[i].step()
                    optimizer_list[i].zero_grad()
                     
            if print_time:
                torch.cuda.synchronize()   
                toc = datetime.datetime.now()
                backward_time.append(toc-tic)   # backward_time
                tic = datetime.datetime.now()
        
        #sum time in each step
        if print_time:
            yy_epoch_time = 0.0

            tmp_t=0.0
            for a in sample_time:
                tmp_t = tmp_t + a.total_seconds()
            yy_sample_time = tmp_t
            yy_epoch_time += tmp_t

            tmp_t=0.0
            for a in h2d_time:
                tmp_t = tmp_t + a.total_seconds()
            yy_h2d_time = tmp_t
            yy_epoch_time += tmp_t

            tmp_t=0.0
            for a in forward_time:
                tmp_t = tmp_t + a.total_seconds()
            yy_forward_time = tmp_t
            yy_epoch_time += tmp_t

            tmp_t=0.0
            for a in backward_time:
                tmp_t = tmp_t + a.total_seconds()
            yy_backward_time = tmp_t
            yy_epoch_time += tmp_t
         
            #log Time
            yy_logger.log(f'Epoch Time: {yy_epoch_time:.4f}')
            yy_logger.log(f'Time Breakdown:  sampling {yy_sample_time:.4f}, data loading {yy_h2d_time:.4f}, forward propagation {yy_forward_time:.4f}, backward propagation {yy_backward_time:.4f}')

            #log GPU memory
            gpu_memory_used = torch.cuda.memory_allocated(rank)/1024/1024
            gpu_memory_reserved = torch.cuda.memory_reserved(rank)/1024/1024
            total_gpu_memory = gpu_memory_used + gpu_memory_reserved
            gpu_max_allocated = torch.cuda.max_memory_allocated(rank)/1024/1024
            yy_logger.log(f'GPU memory total: {total_gpu_memory:.4f} MB')
            yy_logger.log(f'used {gpu_memory_used:.4f} MB, reserved {gpu_memory_reserved:.4f} MB || max memory used {gpu_max_allocated:.4f} MB')
       
           
    dist.destroy_process_group()

