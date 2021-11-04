import datetime
import math
import random
import ipdb
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

from blocked_sae import Decoder, GumbelConvAE, gumbel_softmax
from config import Args_domain_no_completion, Args_domain_probability
from custom_rnn import GumbelVAE_Transition, SigmoidTransition_RNN,Probability
from masked_puzzle import manmade_blocks
from masked_puzzle import manmade_blocks_new
from network import HeurNet, DistNet
from util import append_file, write_file, batch_generator, save_image

from vae_train import VAE
from vae_train_no_completion import VAE_no_completion
from rnn_train import RNN
from rnn_train_no_completion import RNN_no_completion
from probability_train import Prob
from distnet_train import Dist
from checker_domain_train import check
import pickle as pkl
import os
import argparse
import sys
import tables

import pynvml



class Train():
    def __init__(self,domain,num_blocks,mask_length,skip_prob,skip_vae,skip_rnn,check_1,skip_dist,gpu_id,completion):
        print(completion)
        if completion == 0:
            self.args = Args_domain_no_completion(domain, num_blocks,mask_length)
        else:
            self.args = Args_domain_probability(domain, num_blocks,mask_length)
        args = self.args
        self.args.domain_name = domain
        
        if 'hanoi' not in domain and 'blocks' not in domain:
            data = np.load(args.data_path)
            images = torch.from_numpy(data['images'])
            actions = torch.from_numpy(np.eye(4,dtype=np.float32)[data['actions']-1]) # one-hotpip 
            actions[:,0] = 0
            n_train = int(images.shape[0] * 0.85)
            
        else:
            with open(args.data_path, "rb") as f:
                data = pkl.load(f,encoding='urf-8')
            
            images = []
            actions = []
            all_actions = []
            for i in range(len(data)):
                images.append(data[i][1])
                actions.append(data[i][2])
                all_actions += data[i][2]
            actions = np.stack(actions)
            actions_set = list(set(all_actions))
            actions_set.sort(reverse=False)
            args.action_dim = len(actions_set)

            images = torch.round(torch.from_numpy(np.stack(images)))
            images = torch.abs(images - torch.ones_like(images))

            new_actions = []
            for x in range(actions.shape[0]):
                if actions.shape[1] == 7:
                    new_actions.append([0] + [actions_set.index(actions[x][y]) for y in range(actions.shape[1])])
                else:
                    new_actions.append([actions_set.index(actions[x][y]) for y in range(actions.shape[1])])
            actions = torch.from_numpy(np.eye(len(actions_set),dtype=np.float32)[np.stack(new_actions)])
            n_train = 5000
            if 'blocks' in domain:
                n_train = int(args.trace_num * 0.85)
            
            print("All "+str(images.shape[0])+' traces and All ' + str(len(actions_set)) + ' actions')
        
        print(str(images[:n_train].shape[0])+" for training")
        # start_time = time.time()
        # print('Masks generating...')

        # mask_mat = torch.tensor([[manmade_blocks_new(num_blocks,mask_length) for l in range(args.trace_len)] for n in range(args.trace_num)])
        data_mask = np.load(args.mask_path)
        mask_mat = torch.from_numpy(data_mask['mask'])
        # end_time =  time.time()
        # print('time:',end_time - start_time)
        mask_mat[n_train:args.trace_num] = 1
        # print('Masks generating End')

        actions = actions[0:args.trace_num]
        images = images[0:args.trace_num]
        mask_mat = mask_mat[0:args.trace_num]
        if 'path' in domain or 'hanoi' in domain or 'blocks' in domain:
            images = images.float()
            actions = actions.float()
            mask_mat = mask_mat.float()
        self.train_loader = DataLoader(dataset=TensorDataset(images[:n_train], mask_mat[:n_train], actions[:n_train]), 
                            batch_size=args.batch_size,
                            shuffle=False)

        self.test_loader = DataLoader(dataset=TensorDataset(images[n_train:args.trace_num+1], mask_mat[n_train:args.trace_num+1], actions[n_train:args.trace_num+1]), 
                                batch_size=args.batch_size,
                                shuffle=False)
        print(str(images[n_train:args.trace_num+1].shape[0])+' for validation')

        if completion == 1:
            model_probability = nn.DataParallel(Probability(args).cuda())
            if skip_prob == 0:    
                self.gpu_check(gpu_id,1800)
                Prob(self.args,domain,self.train_loader,self.test_loader,model_probability)
            else:
                print('------------Skip Probability------------')
            torch.cuda.empty_cache()
            print("----------Predicting END:{} {} {}---------".format(domain,num_blocks,mask_length))

            
            model = nn.DataParallel(GumbelVAE_Transition(args).cuda())
            if skip_vae == 0:
                if 'hanoi' in domain or 'blocks' in domain:
                    self.gpu_check(gpu_id,8500)
                else:
                    self.gpu_check(gpu_id,9050)
                VAE(self.args,domain,self.train_loader,self.test_loader,model_probability,model)
            else:
                print('------------Skip VAE------------')
            torch.cuda.empty_cache()
            print("----------VAE END:{} {} {}---------".format(domain,num_blocks,mask_length))

            model_rnn = nn.DataParallel(SigmoidTransition_RNN(args,model).cuda())
            if skip_rnn == 0:    
                self.gpu_check(gpu_id,5000)
                RNN(self.args,domain,self.train_loader,self.test_loader,model_probability,model,model_rnn)
            else:
                print('------------Skip RNN------------')
            torch.cuda.empty_cache()
            print("----------RNN END:{} {} {}---------".format(domain,num_blocks,mask_length))
        
        else:
            model = nn.DataParallel(GumbelVAE_Transition(args).cuda())
            if skip_vae == 0:
                if 'hanoi' in domain or 'blocks' in domain:
                    self.gpu_check(gpu_id,8500)
                else:
                    self.gpu_check(gpu_id,9050)
                VAE_no_completion(self.args,domain,self.train_loader,self.test_loader,model)
            else:
                print('------------Skip VAE------------')
            torch.cuda.empty_cache()
            print("----------VAE END:{} {} {}---------".format(domain,num_blocks,mask_length))


            model_rnn = nn.DataParallel(SigmoidTransition_RNN(args,model).cuda())
            if skip_rnn == 0:    
                self.gpu_check(gpu_id,5000)
                RNN_no_completion(self.args,domain,self.train_loader,self.test_loader,model,model_rnn)
            else:
                print('------------Skip RNN------------')
            torch.cuda.empty_cache()
            print("----------RNN END:{} {} {}---------".format(domain,num_blocks,mask_length))

        self.gpu_check(gpu_id,3000)
        if 'blocks' in domain:
            n_train = 8000
            args.batch_size = 1000
            self.train_loader = DataLoader(dataset=TensorDataset(images[:n_train], mask_mat[:n_train], actions[:n_train]), 
                            batch_size=args.batch_size,
                            shuffle=False)

            self.test_loader = DataLoader(dataset=TensorDataset(images[n_train:args.trace_num+1], mask_mat[n_train:args.trace_num+1], actions[n_train:args.trace_num+1]), 
                                batch_size=args.batch_size,
                                shuffle=False)
            self.gpu_check(gpu_id,7700)

        dist_net = nn.DataParallel(DistNet(args).cuda())
        if skip_dist == 0:
            Dist(self.args,domain,self.train_loader,self.test_loader,model,model_rnn,dist_net)
        print("---------- Dist END:{} {} {}---------".format(domain,num_blocks,mask_length))



    def gpu_check(self,gpu_id,minimum):
        print('Waiting for GPU.......')
        pynvml.nvmlInit()
        time.sleep(5)

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        use = meminfo.used/1024/1024
        remain = 11100 - use
        print('remain:',remain)
        
        while remain < minimum:
            time.sleep(10)
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            use = meminfo.used/1024/1024
            remain = 11100 - use
        print('GPU available')


