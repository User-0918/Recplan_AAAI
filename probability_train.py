import datetime
import math
import random
import ipdb
import copy

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
from custom_rnn import GumbelVAE_Transition, SigmoidTransition_RNN,Probability
from masked_puzzle import manmade_blocks
from masked_puzzle import manmade_blocks_new
from util import append_file, write_file, batch_generator, save_image


class Prob():
    def __init__(self,args,domain,train_loader,test_loader,model):
        optimizer = optim.Adam([
                                {"params":model.module.parameters(),'lr':0.001},
                                ])
        start_time = datetime.datetime.now().strftime('%m%d%H%M')
        import sys
        import json
        param_file = model.module.sub_path('record_'+domain+'_probability_{}.json'.format(start_time))
        log_file = model.module.sub_path('record_'+domain+'_probability_{}.csv'.format(start_time))

        write_file(json.dumps(args.__dict__, indent=4), path=param_file)
        headers = ['epoch','train_probability','test_probability']
        append_file(headers, path=log_file, sep=',')

        print("------------ Probability Module Training ------------\n")
        for epoch in tqdm(range(1, args.probability_epoch + 1)):
        # for epoch in tqdm(range(1, 2)):
            last_epoch = epoch == model.module.args.probability_epoch
            
            train_mse = self.train(args,model,train_loader,optimizer,epoch)
            test_mse = self.test(args,model,test_loader,optimizer,epoch)

            loss_info = map(lambda x: '%.6f'%x, [train_mse,test_mse])
            
            append_file(loss_info, log_file, num=epoch,sep=',')
            
            if True:
                model.module.save('model_probability_'+domain+'.pkl'.format(epoch, test_mse))



    def loss_function_mse(self,pred, target):
        MSE = F.mse_loss(pred, target, size_average=False)
        return MSE
        
    def train(self,args,model,train_loader,optimizer,epoch):
        model.module.train()
        mce_loss_prob = 0.0
        domain = args.domain_name

        for batch_idx, data in enumerate(train_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim))

            first_mask_set = mask_tensor[:,:-1]
            second_mask_set = mask_tensor[:,1:]
            first_images = image_tensor[:,:-1] * first_mask_set
            second_images = image_tensor[:,1:] * second_mask_set
        
            probability_target = torch.where(torch.eq(first_mask_set,0)|torch.eq(second_mask_set,0),
                                torch.zeros_like(first_mask_set),
                                torch.where(torch.eq(first_images,second_images),
                                        torch.ones_like(first_mask_set),-1 * torch.ones_like(first_mask_set)
                                )
                                ).view(args.batch_size,-1,*args.image_shape)

            prob_target_output = 0.5 * (probability_target + torch.ones_like(probability_target))
        
            optimizer.zero_grad()        

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
            probability_set = []

            # for i in range(1,args.problem_trace_len):
            first_image = (image_tensor[:,:-1]* mask_tensor[:,:-1]).view(args.batch_size,-1,1,*args.image_shape)
            second_image = (image_tensor[:,1:]* mask_tensor[:,1:]).view(args.batch_size,-1,1,*args.image_shape)
            action = action_tensor[:,1:].view(args.batch_size,-1,1,args.action_dim)

            predict_probability = model(first_image,second_image,action)
            probability_predict = predict_probability.view(args.batch_size,-1,*args.image_shape)
            prob_predict_output = 0.5 * (probability_predict + torch.ones_like(probability_predict))
            mse_prob = self.loss_function_mse(probability_predict*torch.abs(probability_target),probability_target)
                
            mce_loss_prob += mse_prob.data.item()

            loss =  mse_prob 
            loss.backward()
            optimizer.step()

            
            if (epoch % 50 == 0  or epoch == model.module.args.probability_epoch or epoch == 1) and (batch_idx %5 == 0):
                title = 'prob'
                comparison = torch.cat([first_image[0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        second_image[0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,1:][0].view(-1,1,*args.image_shape),
                                        prob_predict_output[0].view(-1,1,*args.image_shape),
                                        prob_target_output[0].view(-1,1,*args.image_shape),
                                        ]
                                        )
                filename = 'results_'+domain+'/train/'+title+'/train_'+title+'_%04d_%04d.png' % (epoch,batch_idx) 
                save_image(comparison.data.cpu(),
                            model.module.sub_path(filename), nrow=prob_predict_output.shape[1])
            
        mce_loss_prob /= len(train_loader.dataset) * (args.trace_len)
        
        return mce_loss_prob
        

    def test(self,args,model,test_loader,optimizer,epoch):
        model.module.eval()
        mce_loss_prob = 0.0
        domain = args.domain_name

        for batch_idx, data in enumerate(test_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim))

            first_mask_set = mask_tensor[:,:-1]
            second_mask_set = mask_tensor[:,1:]
            first_images = image_tensor[:,:-1] * first_mask_set
            second_images = image_tensor[:,1:] * second_mask_set
        
            probability_target = torch.where(torch.eq(first_mask_set,0)|torch.eq(second_mask_set,0),
                                torch.zeros_like(first_mask_set),
                                torch.where(torch.eq(first_images,second_images),
                                        torch.ones_like(first_mask_set),-1 * torch.ones_like(first_mask_set)
                                )
                                ).view(args.batch_size,-1,1,*args.image_shape)
        
            # optimizer.zero_grad()        

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
            probability_set = []

            first_image = (image_tensor[:,:-1]* mask_tensor[:,:-1]).view(args.batch_size,-1,1,*args.image_shape)
            second_image = (image_tensor[:,1:]* mask_tensor[:,1:]).view(args.batch_size,-1,1,*args.image_shape)
            action = action_tensor[:,1:].view(args.batch_size,-1,1,args.action_dim)

            predict_probability = model(first_image,second_image,action)
            probability_predict = predict_probability.view(args.batch_size,-1,1,*args.image_shape)
            
            mse_prob = self.loss_function_mse(probability_predict*torch.abs(probability_target),probability_target)
                
            mce_loss_prob += mse_prob.data.item()
        
        mce_loss_prob /= len(test_loader.dataset) * (args.trace_len)
        
        return mce_loss_prob
