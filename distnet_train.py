# %%
import datetime
import math
import random

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

from blocked_sae import Decoder, GumbelConvAE, gumbel_softmax
from config import Args_domain_probability
from custom_rnn import GumbelVAE_Transition, SigmoidTransition_RNN
from network import HeurNet, DistNet
from masked_puzzle import manmade_blocks,manmade_blocks_new
from util import append_file, write_file, batch_generator, save_image

# start_time = datetime.datetime.now().strftime('%m%d%H%M')
###
class Dist():
    def __init__(self,args,domain,train_loader,test_loader,model,model_rnn,dist_net):
        model_rnn.module.load('model_rnn_'+domain+'.pkl')
        model.module.load('model_'+domain+'.pkl')
        
        start_time = datetime.datetime.now().strftime('%m%d%H%M')

        for para in list(model.module.parameters()):
            para.requires_grad=False 

        for para in list(model.module.parameters()):
            para.requires_grad=False 

        for para in list(model_rnn.module.parameters()):
            para.requires_grad=False 

        for para in list(model_rnn.module.parameters()):
            para.requires_grad=False 
        
        # for name, para in dist_net.named_parameters():
        #     if 'bias' not in name:
        #         torch.nn.init.normal_(para, mean=0, std=1)
        #     else:
        #         torch.nn.init.constant(para, 0)

        optimizer_dist = optim.Adam(dist_net.module.net.parameters(), lr=args.learning_rate)
        import json
        param_file = dist_net.module.sub_path('record_'+domain+'_distnet_{}.json'.format(start_time))
        log_file = dist_net.module.sub_path('record_'+domain+'_distnet_{}.csv'.format(start_time))
        # write_file(json.dumps(args.__dict__, indent=4), path=param_file)
        headers = ['epoch','train_mse_dist', 'test_mse_dist', 'test_loss']
        append_file(headers, path=log_file, sep=',')

        # %% training
        best_test_loss = None
        print("------------Dist_Net Traning------------\n")

        for epoch in tqdm(range(1, model.module.args.dist_epoch  +1)):
        # for epoch in tqdm(range(1, 2)):
            last_epoch = epoch == model.module.args.dist_epoch
            train_bce_rec, train_kld_rec, train_bce_pred, train_kld_pred, train_nll_heur, train_acc, train_acc_trace, train_mse_dist = self.train(args,model,model_rnn,dist_net,train_loader,optimizer_dist,epoch)

            # test_bce_rec, test_kld_rec, test_bce_pred, test_kld_pred, test_nll_heur, test_acc, test_acc_trace, test_mse_dist = self.test(args,model,model_rnn,dist_net,test_loader,optimizer_dist,epoch)
            # test_loss = sum([test_mse_dist])
            
            # loss_info = map(lambda x: '%.6f'%x, [train_mse_dist, test_mse_dist, test_loss])
            loss_info = map(lambda x: '%.6f'%x, [train_mse_dist])
            append_file(loss_info, log_file, sep=',', num=epoch)
            if True:
                # heur_net.save('heur_net_'+domain+'_2.pkl')
                dist_net.module.save('dist_net_'+domain+'.pkl')
                # best_test_loss = test_loss
    

    def loss_function(self,pred, target, qy, cate_dim=2):
        # import ipdb
        # ipdb.set_trace()
        BCE = F.binary_cross_entropy(pred, target, size_average=False)
        MSE = F.mse_loss(pred, target, size_average=False)
        log_qy = torch.log(qy+1e-20)
        g = Variable(torch.log(torch.tensor([1.0/cate_dim])).cuda())
        KLD = torch.sum(qy*(log_qy - g),dim=-1).sum()
        # KLD = torch.sum(qy*(log_qy),dim=-1).mean()
        return BCE, KLD

    def dist_loss(self,args,dist_net,states, goal_state, action_best, action_compute, index,seq_len):
        """
            states: b x l x dim
            actions: b x l x action_dim
            dist: b x lxl x action_dim
        """
        batch_size = states.shape[0]

        target = (seq_len - index) + action_best.view(args.batch_size,-1)

        h1 = dist_net(torch.cat([states.view(args.batch_size,-1), 
                        goal_state.view(args.batch_size,-1), 
                        action_compute], dim=-1))

        return F.mse_loss(h1, target.detach(), size_average=True)
    
    def train(self,args,model,model_rnn,dist_net,train_loader,optimizer_dist,epoch):
        model.module.eval()
        model_rnn.module.eval()
        dist_net.module.train()
        domain = args.domain_name

        model.module.set_curr_temp(model_rnn.module.min_temperature)
        model_rnn.module.set_curr_temp(model_rnn.module.min_temperature)

        temp = model_rnn.module.update_temp(epoch)

        bce_loss_rec = 0.0
        kld_loss_rec = 0.0
        bce_loss_pred = 0.0     # 交叉熵
        kld_loss_pred = 0.0     # KL散度
        nll_loss_heur = 0.0     # 负对数似然
        mse_loss_dist = 0.0     # 均方误差
        correct = 0
        correct_trace = 0

        for batch_idx, data in enumerate(train_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
                ## image (batch_size x 8 x 42 x 42)
                ## mask (batch_size x 8 x 42 x 42)
                ## action (bathc_size x 8 x 4)
            # input_tensor = torch.stack([image_tensor * mask_tensor, mask_tensor], dim=2)[:,:-1] 
                ## 被遮挡的图片 + 遮挡区域 
                ## input_tensor (batch_size x 7 x 2 x 42 x 42)
            # input_tensor = torch.cat([input_tensor[:, 0:1], input_tensor], dim=1)
                ## input_tensor (batch_size x 8 x 2 x 42 x 42) 第一个图 + 包括第一个图的所有 
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target = image_tensor[:,:-1] * mask_tensor[:,:-1] 
            pred_target = image_tensor[:,1:] * mask_tensor [:,1:]

            goal_tensor = pred_target[:, -1].view(args.batch_size,1,*args.image_shape)

            # optimizer_heur.zero_grad()
            optimizer_dist.zero_grad()

    ################################################
            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

            hidden_state = model_rnn.module._init_hidden(batch_size = args.batch_size)

            rec_set = []
            pred_set = []
            qy_set = []
            pred_qy_set = []
            action_pred_set = []
            rec_change_set = []
            qy_change_set = []

            z_set = []
            z_change_set = []

            mse_dist = 0
            nll_heur = 0

            goal_pi,goal_z = model.module.Encoder(goal_tensor,goal_tensor.shape[0],1)
            heur_action_tensor = action_tensor[:,1:].view(args.batch_size,-1,args.action_dim)

            before_tensor = image_tensor[:,0].view(args.batch_size,1,*args.image_shape) * mask_tensor[:,0].view(args.batch_size,1,*args.image_shape)
            pi,z = model.module.Encoder(before_tensor,args.batch_size,1)
            z = z.view(args.batch_size,1,args.cate_num,args.cate_dim)
            z_known = z[:,:,:,0:2]
            # z_known = z
            
            for i in range(1,args.problem_trace_len):    
                rec = model.module.Decoder(z_known, args.batch_size,1)
                rec = rec.view(-1,1,*args.image_shape)

                z_set.append(z_known)
                z_known = z_known.view(args.batch_size,args.cate_num,args.cate_dim-1)

                z, _, hidden_state = model_rnn(z.view(args.batch_size,args.cate_num,args.cate_dim), action_tensor[:,i], hidden_state)
                z_now = z.view(args.batch_size,1,args.cate_num,args.cate_dim)
                z_known = z_now[:,:,:,0:2]

                pred_rec = model.module.Decoder(z, args.batch_size,1)
                pred_rec = pred_rec.view(-1,1,*args.image_shape)
                
                if i == args.problem_trace_len - 1:
                    z_set.append(z_known)
                
            z = torch.stack(z_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)

            heur_action_dist = (0.5 - heur_action_tensor) * 2

            for x in range(args.problem_trace_len):
                for y in range(x+1,args.problem_trace_len):
                    for a_i in range(args.action_dim):
                        mse_dist_now = self.dist_loss(args,dist_net,z[:,x].view(args.batch_size,-1,args.cate_num,args.cate_dim-1), 
                                                z[:,y].view(args.batch_size,-1,args.cate_num,args.cate_dim-1), 
                                                heur_action_dist[:,x][:,a_i],
                                                torch.eye(args.action_dim)[a_i].expand(args.batch_size, args.action_dim).cuda(),
                                                x,y)
                        mse_dist += mse_dist_now
            loss = mse_dist
            loss.backward()
            # optimizer_heur.step()
            optimizer_dist.step()

            mse_loss_dist += mse_dist.data.item()
        
        B, M, N = len(train_loader), len(train_loader.dataset), args.trace_len
        
        acc = 0
        acc_trace = 0
        mse_loss_dist /= B
        
        return bce_loss_rec, kld_loss_rec, bce_loss_pred, kld_loss_pred, nll_loss_heur, acc, acc_trace, mse_loss_dist

    def test(self,args,model,model_rnn,dist_net,test_loader,optimizer_dist,epoch):
        model.module.eval()
        model_rnn.module.eval()
        # heur_net.eval()
        dist_net.module.eval()
        domain = args.domain_name

        temp = model_rnn.module.update_temp(epoch)

        model.module.set_curr_temp(model_rnn.module.min_temperature)
        model_rnn.module.set_curr_temp(model_rnn.module.min_temperature)

        bce_loss_rec = 0.0
        kld_loss_rec = 0.0
        bce_loss_pred = 0.0
        kld_loss_pred = 0.0
        nll_loss_heur = 0.0
        mse_loss_dist = 0.0
        correct = 0
        correct_trace = 0

        for batch_idx, data in enumerate(test_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
                ## image (batch_size x 8 x 42 x 42)
                ## mask (batch_size x 8 x 42 x 42)
                ## action (bathc_size x 8 x 4)
            # input_tensor = torch.stack([image_tensor * mask_tensor, mask_tensor], dim=2)[:,:-1] 
                ## 被遮挡的图片 + 遮挡区域 
                ## input_tensor (batch_size x 7 x 2 x 42 x 42)
            # input_tensor = torch.cat([input_tensor[:, 0:1], input_tensor], dim=1)
                ## input_tensor (batch_size x 8 x 2 x 42 x 42) 第一个图 + 包括第一个图的所有 
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target = image_tensor[:,:-1] * mask_tensor[:,:-1] 
            pred_target = image_tensor[:,1:] * mask_tensor [:,1:]

            goal_tensor = pred_target[:, -1].view(args.batch_size,1,*args.image_shape)

            # optimizer_heur.zero_grad()
            optimizer_dist.zero_grad()

    ################################################
            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

            hidden_state = model_rnn.module._init_hidden(batch_size = args.batch_size)

            rec_set = []
            pred_set = []
            qy_set = []
            pred_qy_set = []
            action_pred_set = []
            rec_change_set = []
            qy_change_set = []

            z_set = []
            z_change_set = []

            mse_dist = 0
            nll_heur = 0

            goal_pi,goal_z = model.module.Encoder(goal_tensor,goal_tensor.shape[0],1)
            heur_action_tensor = action_tensor[:,1:].view(args.batch_size,-1,args.action_dim)

            before_tensor = image_tensor[:,0].view(args.batch_size,1,*args.image_shape) * mask_tensor[:,0].view(args.batch_size,1,*args.image_shape)
            pi,z = model.module.Encoder(before_tensor,args.batch_size,1)
            z = z.view(args.batch_size,1,args.cate_num,args.cate_dim)
            z_known = z[:,:,:,0:2]
            
            for i in range(1,args.problem_trace_len):    
                rec = model.module.Decoder(z_known, args.batch_size,1)
                rec = rec.view(-1,1,*args.image_shape)

                z_set.append(z_known)
                z_known = z_known.view(args.batch_size,args.cate_num,args.cate_dim-1)
                

                z, _, hidden_state = model_rnn(z_known, action_tensor[:,i], hidden_state)
                z_now = z.view(args.batch_size,1,args.cate_num,args.cate_dim-1)
                z_known = z_now[:,:,:,0:2]

                pred_rec = model.module.Decoder(z, args.batch_size,1)
                pred_rec = pred_rec.view(-1,1,*args.image_shape)
                
                if i == args.problem_trace_len - 1:
                    z_set.append(z_known)
                
            z = torch.stack(z_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)

            heur_action_dist = (0.5 - heur_action_tensor) * 2

            for x in range(args.problem_trace_len):
                for y in range(x+1,args.problem_trace_len):
                    for a_i in range(args.action_dim):
                        mse_dist_now = self.dist_loss(args,dist_net,z[:,x].view(args.batch_size,-1,args.cate_num,args.cate_dim-1), 
                                                z[:,y].view(args.batch_size,-1,args.cate_num,args.cate_dim-1), 
                                                heur_action_dist[:,x][:,a_i],
                                                torch.eye(args.action_dim)[a_i].expand(args.batch_size, args.action_dim).cuda(),
                                                x,y)
                        mse_dist += mse_dist_now
            loss = mse_dist
            
            mse_loss_dist += mse_dist.data.item()
            
        B, M, N = len(test_loader), len(test_loader.dataset), args.trace_len
        
        acc = 0
        acc_trace = 0
        mse_loss_dist /= B
        
        return bce_loss_rec, kld_loss_rec, bce_loss_pred, kld_loss_pred, nll_loss_heur, acc, acc_trace, mse_loss_dist
