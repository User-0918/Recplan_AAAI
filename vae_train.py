# %%
"""
train and test VAE and RNN
"""
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
from custom_rnn import GumbelVAE_Transition,Probability
from masked_puzzle import manmade_blocks
from masked_puzzle import manmade_blocks_new
from util import append_file, write_file, batch_generator, save_image

# 

# %%
class VAE():
    def __init__(self,args,domain,train_loader,test_loader,model_probability,model):
        model_probability.module.load('model_probability_'+domain+'.pkl')
        
        optimizer_model = optim.Adam([
                                {"params":model.module.parameters(),'lr':0.0005},
                                ])
        start_time = datetime.datetime.now().strftime('%m%d%H%M')
        import json
        param_file = model.module.sub_path('record_'+domain+'_vae_{}.json'.format(args.hidden_dim))
        log_file = model.module.sub_path('record_'+domain+'_vae_{}_{}.csv'.format(args.hidden_dim,args.cate_num))
        # write_file(json.dumps(args.__dict__, indent=4), path=param_file)
        # headers = ['epoch', 'train_bce_recover','train_bce_rec', 'train_kld_rec',
        # 'test_bce_recover','test_bce_rec', 'test_kld_rec', 'test_bce_pred', 'test_kld_pred',  'train_bce_trans','test_loss']
        headers = ['epoch','train_bce_rec', 'train_kld_rec', 'train_no_change','train_change',
                'test_bce_rec', 'test_kld_rec', 'test_no_change','test_change','test_loss','best_test_loss']

        append_file(headers, path=log_file, sep=',')

        # # %% training
        best_test_loss_rec = None
        print("------------ VAE Module Traning ------------\n")

        for epoch in tqdm(range(1, args.vae_epoch+ 1)):
        # for epoch in tqdm(range(1, 2)):
            last_epoch = epoch == model.module.args.vae_epoch

            train_bce_rec, train_kld_rec, train_no_change,train_change = self.train(args,model,model_probability,train_loader,optimizer_model,epoch)
            test_bce_rec, test_kld_rec, test_no_change,test_change = self.test(args,model,model_probability,test_loader,optimizer_model,epoch)

            test_loss = test_bce_rec + test_no_change

            
            
            if best_test_loss_rec == None or test_loss < best_test_loss_rec:
                model.module.save('model_'+domain+'.pkl'.format(epoch, test_loss))
                best_test_loss_rec = test_loss
                # model_rnn.module.save('model_rnn_'+domain+'.pkl'.format(epoch, test_loss))
            loss_info = map(lambda x: '%.6f'%x, [train_bce_rec, train_kld_rec, train_no_change,train_change,
                                                test_bce_rec, test_kld_rec, test_no_change,test_change,test_loss,best_test_loss_rec])
            append_file(loss_info, log_file, num=epoch,sep=',')


    def loss_function(self,pred, target, qy, cate_dim=2):
        # import ipdb
        # ipdb.set_trace()
        BCE = F.binary_cross_entropy(pred, target, size_average=False)
        MSE = F.mse_loss(pred, target, size_average=False)
        log_qy = torch.log(qy+1e-20)
        g = Variable(torch.log(torch.tensor([1.0/cate_dim])).cuda())
        KLD = torch.sum(qy*(log_qy - g),dim=-1).sum()
        # KLD = torch.sum(qy*(log_qy),dim=-1).mean()
        ### KLD 相对熵
        return BCE, KLD

    def loss_function_bce(self,pred, target):
        # import ipdb
        # ipdb.set_trace()
        BCE = F.binary_cross_entropy(pred, target, size_average=False)
        return BCE

    def loss_function_mse(self,pred, target):
        # import ipdb
        # ipdb.set_trace()
        # BCE = F.binary_cross_entropy(pred, target, size_average=False)
        MSE = F.mse_loss(pred, target, size_average=False)
        # log_qy = torch.log(qy+1e-20)
        # g = Variable(torch.log(torch.tensor([1.0/cate_dim])).cuda())
        # KLD = torch.sum(qy*(log_qy - g),dim=-1).sum()
        # KLD = torch.sum(qy*(log_qy),dim=-1).mean()
        ### KLD 相对熵
        return MSE

    def loss_function_mean(self,pred, target):
        pred_loss = torch.sum(pred**2)
        target = torch.sum(target**2)

        MSE = torch.sqrt((target - pred_loss)**2)
        return MSE

    def train(self,args,model,model_probability,train_loader,optimizer_model,epoch):
        model.module.train()
        model_probability.module.eval()
        # model_inpainting.module.train()
        bce_loss_rec = 0.0
        kld_loss_rec = 0.0
        domain = args.domain_name

        bce_loss_no_change = 0.0
        mean_loss_change = 0.0

        # if epoch > 1000:
        temp = model.module.update_temp(epoch)

        for batch_idx, data in enumerate(train_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target_1_before = image_tensor[:,:-1] * mask_tensor[:,:-1]
            rec_target_2_before = image_tensor[:,1:] * mask_tensor[:,1:]

            optimizer_model.zero_grad()

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
            rec_set_1 = []
            qy_set_1 = []
            rec_set_2 = []
            qy_set_2 = []

            inpainting_image_set_1 = []
            inpainting_image_set_2 = []

            no_change_set_1 = []
            no_change_set_2 = []

            change_set_1 = []
            change_set_2 = []

    #######################################################################    
            action = action_tensor[:,1:]    
            first_image = image_tensor * mask_tensor
            first_image = first_image[:,:-1].view(args.batch_size,-1,1,*args.image_shape)
            second_image = image_tensor[:,1:].view(args.batch_size,-1,1,*args.image_shape)* mask_tensor[:,1:].view(args.batch_size,-1,1,*args.image_shape)

            rec_1,pi_1,z_1 = model(first_image)
            rec_1 = rec_1.view(args.batch_size,-1,*args.image_shape)
            pi_1 = pi_1.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            
            rec_2,pi_2,z_2 = model(second_image)
            pi_2 = pi_2.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            rec_2 = rec_2.view(args.batch_size,-1,*args.image_shape)

            probability = model_probability(first_image,second_image,action).view(args.batch_size,-1,*args.image_shape)
            probability = torch.where(torch.gt(probability,torch.ones_like(probability) * 0.99 ),
                                        torch.ones_like(probability),
                                        -1 * torch.ones_like(probability))
            probability = -0.5 * (probability - torch.ones_like(probability))

            no_change_image = torch.where(
                                        torch.eq(mask_tensor[:,:-1],torch.ones_like(mask_tensor[:,:-1])),
                                        rec_target_1_before,
                                        torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,1:],torch.ones_like(probability)),
                                        rec_target_2_before,
                                        torch.zeros_like(probability)
                                        )
                                        )
            rec_change,pi_change,z_change = model(no_change_image.view(args.batch_size,-1,1,*args.image_shape))
            pi_change = pi_change.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            rec_change = rec_change.view(args.batch_size,-1,*args.image_shape)

            z_change = z_change.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            z_1 = z_1.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            z_2 = z_2.view(args.batch_size,-1,args.cate_num,args.cate_dim)

            compare_mask_1 = torch.where(
                                        torch.eq(mask_tensor[:,:-1],torch.ones_like(mask_tensor[:,:-1])),
                                        torch.ones_like(probability),
                                        torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,1:],torch.ones_like(probability)),
                                        torch.ones_like(probability),
                                        torch.zeros_like(probability)
                                        )
                                        )

            ################ VAE loss
            bce_rec_1, kld_rec_1 = self.loss_function(pred=rec_1*mask_tensor[:,:-1],
                                            target=rec_target_1_before, 
                                            qy=pi_1)
                                            
            bce_rec_no_change, kld_rec_no_change = self.loss_function(pred=rec_change*compare_mask_1,
                                            target=no_change_image, 
                                            qy=pi_change)
                                            
            bce_rec_change_state = (self.loss_function_mse(pred=z_change[:,:,:,0:2] * (z_1[:,:,:,0:2] ** 2),
                                            target=z_1[:,:,:,0:2])
                                    + self.loss_function_mse(pred= (z_change[:,:,:,2] ** 2) * z_1[:,:,:,2],
                                            target=z_change[:,:,:,2])
                                    # + self.loss_function_bce(pred= z_change,
                                    #         target=pi_change.detach())
                                    # + self.loss_function_bce(pred= z_1,
                                    #         target=pi_1.detach())
                                            )
            bce_rec_change_state_see = (self.loss_function_bce(pred=z_change[:,:,:,0:2] * (z_1[:,:,:,0:2] ** 2),
                                            target=z_1[:,:,:,0:2].detach())
                                    + self.loss_function_bce(pred= (z_change[:,:,:,2] ** 2) * z_1[:,:,:,2],
                                            target=z_change[:,:,:,2].detach())
                                    # + self.loss_function_bce(pred= z_change,
                                    #         target=pi_change.detach())
                                    # + self.loss_function_bce(pred= z_1,
                                    #         target=pi_1.detach())
                                            )

            bce_loss_rec += bce_rec_1.data.item()
            kld_loss_rec += kld_rec_1.data.item()
            bce_loss_no_change += bce_rec_change_state_see.data.item()
            
            # if epoch % 2 == 0:
            loss =  bce_rec_1  + bce_rec_change_state
            # else:
            #     loss = 
            loss.backward()
            optimizer_model.step()        
        
            if ((epoch % 10 == 0 and epoch <= 1000) or
                epoch == model.module.args.vae_epoch or epoch == 1) and (batch_idx %5 == 0):
                title = 'vae'
                comparison = torch.cat([
                                        image_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        rec_1[0].view(-1,1,*args.image_shape),
                                        # inpainting_image_batch_1[0].view(-1,1,*args.image_shape),
                                        rec_target_1_before[0].view(-1,1,*args.image_shape),
                                        probability[0].view(-1,1,*args.image_shape),
                                        no_change_image[0].view(-1,1,*args.image_shape),
                                        compare_mask_1[0].view(-1,1,*args.image_shape),
                                        rec_change[0].view(-1,1,*args.image_shape),
                                        # rec_target_1[0].view(-1,1,*args.image_shape),
                                        ]
                                        )
                filename = 'results_'+domain+'/train/'+title+'/train_'+title+'_%04d_%04d.png' % (epoch,batch_idx)
                save_image(comparison.data.cpu(),
                            model.module.sub_path(filename), nrow=rec_1.shape[1])
        
        bce_loss_rec /= len(train_loader.dataset) * (args.trace_len)
        kld_loss_rec /= len(train_loader.dataset) * (args.trace_len)
        bce_loss_no_change /= len(train_loader.dataset) * (args.trace_len)
        mean_loss_change /= len(train_loader.dataset) * (args.trace_len)
        
        return bce_loss_rec,kld_loss_rec,bce_loss_no_change,mean_loss_change

    def test(self,args,model,model_probability,test_loader,optimizer_model,epoch):
        model.module.eval()
        model_probability.module.eval()
        bce_loss_rec = 0.0
        kld_loss_rec = 0.0
        bce_loss_no_change = 0.0
        mean_loss_change = 0.0
        domain = args.domain_name

        # if epoch > 1000:
        temp = model.module.update_temp(epoch)

        for batch_idx, data in enumerate(test_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target_1 = image_tensor[:,:-1] * mask_tensor[:,:-1]
            rec_target_2 = image_tensor[:,1:] * mask_tensor[:,1:]

            rec_target_1_before = image_tensor[:,:-1] * mask_tensor[:,:-1]
            rec_target_2_before = image_tensor[:,1:] * mask_tensor[:,1:]

            optimizer_model.zero_grad()

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
            rec_set_1 = []
            qy_set_1 = []
            rec_set_2 = []
            qy_set_2 = []

            inpainting_image_set_1 = []
            inpainting_image_set_2 = []

            no_change_set_1 = []
            no_change_set_2 = []

            change_set_1 = []
            change_set_2 = []

    #######################################################################        
            action = action_tensor[:,1:]    
            first_image = image_tensor * mask_tensor
            first_image = first_image[:,:-1].view(args.batch_size,-1,1,*args.image_shape)
            second_image = image_tensor[:,1:].view(args.batch_size,-1,1,*args.image_shape)* mask_tensor[:,1:].view(args.batch_size,-1,1,*args.image_shape)

            rec_1,pi_1,z_1 = model(first_image)
            rec_1 = rec_1.view(args.batch_size,-1,*args.image_shape)
            pi_1 = pi_1.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            
            rec_2,pi_2,z_2 = model(second_image)
            pi_2 = pi_2.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            rec_2 = rec_2.view(args.batch_size,-1,*args.image_shape)

            probability = model_probability(first_image,second_image,action).view(args.batch_size,-1,*args.image_shape)
            probability = torch.where(torch.gt(probability,torch.ones_like(probability) * 0.99 ),
                                        torch.ones_like(probability),
                                        -1 * torch.ones_like(probability))
            probability = -0.5 * (probability - torch.ones_like(probability))

            no_change_image = torch.where(
                                        torch.eq(mask_tensor[:,:-1],torch.ones_like(mask_tensor[:,:-1])),
                                        rec_target_1_before,
                                        torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,1:],torch.ones_like(probability)),
                                        rec_target_2_before,
                                        torch.zeros_like(probability)
                                        )
                                        )
            rec_change,pi_change,z_change = model(no_change_image.view(args.batch_size,-1,1,*args.image_shape))
            pi_change = pi_change.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            rec_change = rec_change.view(args.batch_size,-1,*args.image_shape)

            z_change = z_change.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            z_1 = z_1.view(args.batch_size,-1,args.cate_num,args.cate_dim)
            z_2 = z_2.view(args.batch_size,-1,args.cate_num,args.cate_dim)

            compare_mask_1 = torch.where(
                                        torch.eq(mask_tensor[:,:-1],torch.ones_like(mask_tensor[:,:-1])),
                                        torch.ones_like(probability),
                                        torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,1:],torch.ones_like(probability)),
                                        torch.ones_like(probability),
                                        torch.zeros_like(probability)
                                        )
                                        )

            ################ VAE loss
            bce_rec_1, kld_rec_1 = self.loss_function(pred=rec_1*mask_tensor[:,:-1],
                                            target=rec_target_1_before, 
                                            qy=pi_1)
                                            
            bce_rec_no_change, kld_rec_no_change = self.loss_function(pred=rec_change*compare_mask_1,
                                            target=no_change_image, 
                                            qy=pi_change)
                                            
            bce_rec_change_state = (self.loss_function_mse(pred=z_change[:,:,:,0:2] * (z_1[:,:,:,0:2] ** 2),
                                            target=z_1[:,:,:,0:2])
                                    + self.loss_function_mse(pred= (z_change[:,:,:,2] ** 2) * z_1[:,:,:,2],
                                            target=z_change[:,:,:,2])
                                    # + self.loss_function_bce(pred= z_change,
                                    #         target=pi_change.detach())
                                    # + self.loss_function_bce(pred= z_1,
                                    #         target=pi_1.detach())
                                            )
            bce_rec_change_state_see = (self.loss_function_bce(pred=z_change[:,:,:,0:2] * (z_1[:,:,:,0:2] ** 2),
                                            target=z_1[:,:,:,0:2].detach())
                                    + self.loss_function_bce(pred= (z_change[:,:,:,2] ** 2) * z_1[:,:,:,2],
                                            target=z_change[:,:,:,2].detach())
                                    # + self.loss_function_bce(pred= z_change,
                                    #         target=pi_change.detach())
                                    # + self.loss_function_bce(pred= z_1,
                                    #         target=pi_1.detach())
                                            )

            bce_loss_rec += bce_rec_1.data.item()
            kld_loss_rec += kld_rec_1.data.item()
            bce_loss_no_change += bce_rec_change_state_see.data.item()
            
            # loss =  bce_rec_1
        
            if ((epoch % 10 == 0 and epoch <= 1000) or
                epoch == model.module.args.vae_epoch or epoch == 1) and (batch_idx %5 == 0):
                title = 'vae'
                comparison = torch.cat([
                                        image_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        rec_1[0].view(-1,1,*args.image_shape),
                                        # inpainting_image_batch_1[0].view(-1,1,*args.image_shape),
                                        rec_target_1[0].view(-1,1,*args.image_shape),
                                        ]
                                        )
                filename = 'results_'+domain+'/test/'+title+'/test_'+title+'_%04d_%04d.png' % (epoch,batch_idx) 
                save_image(comparison.data.cpu(),
                            model.module.sub_path(filename), nrow=rec_1.shape[1])
        
        bce_loss_rec /= len(test_loader.dataset) * (args.trace_len)
        kld_loss_rec /= len(test_loader.dataset) * (args.trace_len)
        bce_loss_no_change /= len(test_loader.dataset) * (args.trace_len)
        mean_loss_change /= len(test_loader.dataset) * (args.trace_len)
        
        return bce_loss_rec,kld_loss_rec,bce_loss_no_change,mean_loss_change

