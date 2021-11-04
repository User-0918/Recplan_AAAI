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
from custom_rnn import GumbelVAE_Transition,Probability,SigmoidTransition_RNN
from masked_puzzle import manmade_blocks
from masked_puzzle import manmade_blocks_new
from util import append_file, write_file, batch_generator, save_image

class RNN_no_completion():
    def __init__(self,args,domain,train_loader,test_loader,model,model_rnn):
        
        # model_probability.module.load('model_probability_'+domain+'.pkl')
        model.module.load('model_'+domain+'.pkl')

        for para in list(model.module.parameters()):
            para.requires_grad=False 
        # for para in list(model_probability.module.parameters()):
        #     para.requires_grad=False 

        start_time = datetime.datetime.now().strftime('%m%d%H%M')

        # if 'blocks' not in args.domain_name:
        optimizer = optim.Adam([
                            # {"params":model_probability.module.parameters(),'lr':0.00001},
                            {"params":model_rnn.module.rnn_1_h.parameters(),'lr':0.001},
                            {"params":model_rnn.module.rnn_1_x.parameters(),'lr':0.001},
                            {"params":model_rnn.module.rnn_1_o.parameters(),'lr':0.001},
                            {"params":model_rnn.module.linear.parameters(),'lr':0.001},
                            ])
        
        import json
        # param_file = model_rnn.module.sub_path('record_'+domain+'_transition_{}.json'.format(start_time))
        log_file = model_rnn.module.sub_path('record_'+domain+'_transition_hiddenx{}_rnn_hiddenx{}_rnn_linearx{}.csv'.format(args.hidden_dim,args.rnn_hidden_dim,args.rnn_linear_dim))
        # write_file(json.dumps(args.__dict__, indent=4), path=param_file)
        # headers = ['epoch', 'train_bce_recover','train_bce_rec', 'train_kld_rec',
        # 'test_bce_recover','test_bce_rec', 'test_kld_rec', 'test_bce_pred', 'test_kld_pred',  'train_bce_trans','test_loss']
        headers = ['epoch','train_bce_rec', 'train_kld_rec', 'train_bce_pred','train_kld_rec','train_bce_no_change','train_bce_state',
                'test_bce_rec', 'test_kld_rec', 'test_bce_pred','test_kld_pred','test_bce_state','test_loss']

        append_file(headers, path=log_file, sep=',')

        # # %% training
        best_test_loss_rec = None
        print("------------ Transition Module Traning ------------\n")
        for epoch in tqdm(range(1, args.rnn_num_epoch+ 1)):
        # for epoch in tqdm(range(1, 200)):
            last_epoch = epoch == model.module.args.rnn_num_epoch

            train_bce_rec, train_kld_rec, train_bce_pred,train_kld_pred,train_bce_no_change,train_bce_state = self.train(args,model,model_rnn,train_loader,optimizer,epoch)
            test_bce_rec, test_kld_rec, test_bce_pred,test_kld_pred,test_bce_state = self.test(args,model,model_rnn,test_loader,optimizer,epoch)

            test_loss = sum([test_bce_pred, test_bce_rec,test_bce_state])
            
            if best_test_loss_rec == None or best_test_loss_rec > test_loss:

                model_rnn.module.save('model_rnn_'+domain+'.pkl'.format(epoch, test_loss))
                best_test_loss_rec = test_loss
                # model_probability.module.save('model_probability_'+domain+'_finetuning.pkl'.format(epoch, test_loss))
            loss_info = map(lambda x: '%.6f'%x, [train_bce_rec, train_kld_rec, train_bce_pred,train_kld_pred,train_bce_no_change,train_bce_state,
                                                test_bce_rec, test_kld_rec,test_bce_pred,test_kld_pred,test_bce_state,test_loss,best_test_loss_rec])
            append_file(loss_info, log_file, num=epoch,sep=',')

        


    # %%
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



    def train(self,args,model,model_rnn,train_loader,optimizer,epoch):
        model.module.eval()
        model_rnn.module.train()
        domain = args.domain_name

        bce_loss_rec = 0.0
        kld_loss_rec = 0.0

        bce_loss_pred = 0.0
        kld_loss_pred = 0.0

        bce_loss_no_change = 0.0
        mean_loss_change = 0.0

        bce_loss_state = 0.0

        # if epoch > 1000:
        # temp = model.module.update_temp(epoch)
        temp = model_rnn.module.update_temp(epoch)

        for batch_idx, data in enumerate(train_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target_1_before= image_tensor[:,:-1] * mask_tensor[:,:-1]
            rec_target_2_before = image_tensor[:,1:] * mask_tensor[:,1:]

            rec_target = image_tensor[:,:-1] * mask_tensor[:,:-1]
            pred_target = image_tensor[:,1:] * mask_tensor[:,1:]

            first_image = image_tensor[:,:-1].view(args.batch_size,-1,*args.image_shape) * mask_tensor[:,:-1].view(args.batch_size,-1,*args.image_shape)
            second_image = image_tensor[:,1:].view(args.batch_size,-1,*args.image_shape) * mask_tensor[:,1:].view(args.batch_size,-1,*args.image_shape)

            action = action_tensor[:,1:]    
            # probability = model_probability(first_image,second_image,action).view(args.batch_size,-1,*args.image_shape)
            # probability = torch.where(torch.gt(probability,torch.ones_like(probability) * 0.99 ),
            #                             torch.ones_like(probability),
            #                             -1 * torch.ones_like(probability))
            # probability = -0.5 * (probability - torch.ones_like(probability))

            # no_change_image = torch.where(
            #                             torch.eq(mask_tensor[:,:-1],torch.ones_like(mask_tensor[:,:-1])),
            #                             rec_target_1_before,
            #                             torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,1:],torch.ones_like(probability)),
            #                             rec_target_2_before,
            #                             torch.zeros_like(probability)
            #                             )
            #                             )


            # pred_target_no_change = torch.where(
            #                             torch.eq(mask_tensor[:,1:],torch.ones_like(mask_tensor[:,1:])),
            #                             rec_target_2_before,
            #                             torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,:-1],torch.ones_like(probability)),
            #                             rec_target_1_before,
            #                             torch.zeros_like(probability)
            #                             )
            #                             )

            
            # compare_mask_2 = torch.where(
            #                             torch.eq(mask_tensor[:,1:],torch.ones_like(mask_tensor[:,1:])),
            #                             torch.ones_like(probability),
            #                             torch.where(torch.eq(probability,torch.zeros_like(probability)) & torch.eq(mask_tensor[:,:-1],torch.ones_like(probability)),
            #                             torch.ones_like(probability),
            #                             torch.zeros_like(probability)
            #                             )
            #                             )
            # compare_mask_pred = torch.cat([compare_mask_2[:,0].view(args.batch_size,1,*args.image_shape),
            #                         (compare_mask_2[:,1:] * compare_mask_2[:,:-1]).view(args.batch_size,-1,*args.image_shape)],dim=1)
            
            optimizer.zero_grad()

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            hidden_state = model_rnn.module._init_hidden(batch_size = args.batch_size)
            
            rec_set = []
            qy_set = []
            pred_set = []
            pred_qy_set = []
            predict_state_set = []
            target_state_set = []
            rec_state = []

            no_change_set_1 = []
            no_change_set_2 = []
    ###################################################################
            init_tensor = first_image[:,0].view(args.batch_size,1,*args.image_shape)
            pi,z = model.module.Encoder(init_tensor,args.batch_size,1)
            for i in range(1,args.trace_len):
                rec = model.module.Decoder(z[:,:,0:2], args.batch_size,1)
                pi,_ = model.module.Encoder(rec,args.batch_size,1)
                rec = rec.view(-1,1,*args.image_shape)
                rec_state.append(z)

                # _,_,z_target = model(pred_target_no_change[:,i-1].view(args.batch_size,1,*args.image_shape))
                # z_target = z_target.view(args.batch_size,1,args.cate_num,args.cate_dim)

                z, pred_qy, hidden_state = model_rnn(z, action_tensor[:,i], hidden_state)
                z = z.view(args.batch_size,args.cate_num,args.cate_dim)
                pred_rec = model.module.Decoder(z[:,:,0:2], args.batch_size,1)
                pred_rec = pred_rec.view(-1,1,*args.image_shape)

                # z_target_know = z_target[:,:,:,0:2]
                # z_target_unknow = z_target[:,:,:,2].view(args.batch_size,1,args.cate_num,1)
                

                # target_state_set.append(z_target_know)
                predict_state_set.append(z)

                rec_set.append(rec)
                qy_set.append(F.softmax(pi, dim=-1).view(-1,1,args.cate_num,args.cate_dim))
                pred_set.append(pred_rec)
                pred_qy_set.append(pred_qy.view(-1,1,args.cate_num,args.cate_dim-1))
            
            rec_batch = torch.stack(rec_set,dim = 1).view(args.batch_size,-1,*args.image_shape)
            pred_batch = torch.stack(pred_set,dim = 1).view(args.batch_size,-1,*args.image_shape)
            qy = torch.stack(qy_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim)
            pred_qy = torch.stack(pred_qy_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)
            # rec_state = torch.stack(rec_state,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim)
            # predict_state = torch.stack(predict_state_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)
            # target_state = torch.stack(target_state_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)

            
            ################ VAE loss
            bce_rec, kld_rec = self.loss_function(pred=rec_batch*mask_tensor[:,:-1],
                                            target=rec_target, 
                                            qy=qy)
            bce_pred, kld_pred = self.loss_function(pred=pred_batch*mask_tensor[:,1:],
                                            target=pred_target, 
                                            qy=pred_qy)
            bce_pred_mse = self.loss_function_mse(pred=pred_batch*mask_tensor[:,1:],
                                            target=pred_target)
            # bce_pred_mse = self.loss_function_mse(pred=pred_batch*compare_mask_2,
            #                                 target=pred_target_no_change)
            # # kld_pred = 
            # bce_state,kld_state = self.loss_function(pred = predict_state,
            #                         target = target_state.detach(),
            #                         qy=pred_qy
            #                         )
            # state_mse = self.loss_function_mse(pred = predict_state,
            #                         target = target_state,
            #                         # qy=pred_qy
            #                         )
            

            bce_loss_rec += bce_rec.data.item()
            kld_loss_rec += kld_rec.data.item()
            bce_loss_pred += bce_pred.data.item()
            kld_loss_pred += kld_pred.data.item()
            # bce_loss_no_change += bce_no_change.data.item()
            # bce_loss_state += bce_state.data.item()

            loss = bce_pred_mse
            # loss = state_mse
            # loss = bce_state + bce_no_change + bce_loss_pred
            loss.backward()
            optimizer.step()        
        
            if (epoch % 50 == 0 or epoch == model.module.args.rnn_num_epoch or epoch == 1) and (batch_idx %5 == 0):
                title = 'rnn'
                comparison = torch.cat([
                                        image_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        rec_batch[0].view(-1,1,*args.image_shape),
                                        rec_target[0].view(-1,1,*args.image_shape),
                                        pred_batch[0].view(-1,1,*args.image_shape),
                                        pred_target[0].view(-1,1,*args.image_shape),
                                        # compare_mask_2[0].view(-1,1,*args.image_shape),
                                        # compare_mask_pred[0].view(-1,1,*args.image_shape),
                                        # pred_target_no_change[0].view(-1,1,*args.image_shape),
                                        ]
                                        )
                filename = 'results_'+domain+'/train/'+title+'/train_'+title+'_%04d_%04d.png' % (epoch,batch_idx) 
                save_image(comparison.data.cpu(),
                            model.module.sub_path(filename), nrow=rec_batch.shape[1])
        
        bce_loss_rec /= len(train_loader.dataset) * (args.trace_len)
        kld_loss_rec /= len(train_loader.dataset) * (args.trace_len)
        bce_loss_pred /= len(train_loader.dataset) * (args.trace_len)
        kld_loss_pred /= len(train_loader.dataset) * (args.trace_len)
        bce_loss_no_change /= len(train_loader.dataset) * (args.trace_len)
        bce_loss_state /= len(train_loader.dataset) * (args.trace_len)
        
        return bce_loss_rec,kld_loss_rec,bce_loss_pred,kld_loss_pred,bce_loss_no_change,bce_loss_state

    def test(self,args,model,model_rnn,test_loader,optimizer,epoch):
        model.module.eval()
        # model_probability.module.eval()
        model_rnn.module.eval()
        domain = args.domain_name

        bce_loss_rec = 0.0
        kld_loss_rec = 0.0

        bce_loss_pred = 0.0
        kld_loss_pred = 0.0

        bce_loss_no_change = 0.0
        mean_loss_change = 0.0

        bce_loss_state = 0.0

        # if epoch > 1000:
        temp = model.module.update_temp(epoch)
        temp = model_rnn.module.update_temp(epoch)

        for batch_idx, data in enumerate(test_loader):
            image_tensor, mask_tensor, action_tensor = data[0].cuda(), data[1].cuda(), data[2].cuda()
            action_tensor = action_tensor
                ## action tensor （batch_size x 8 x 4(action_dim)）
            rec_target = image_tensor[:,:-1] * mask_tensor[:,:-1]
            pred_target = image_tensor[:,1:] * mask_tensor[:,1:]

            if not args.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            hidden_state = model_rnn.module._init_hidden(batch_size = args.batch_size)
            
            rec_set = []
            qy_set = []
            pred_set = []
            pred_qy_set = []
            predict_state_set = []
            target_state_set = []

    ###################################################################
            init_tensor = image_tensor[:,0].view(args.batch_size,1,*args.image_shape) * mask_tensor[:,0].view(args.batch_size,1,*args.image_shape)
            pi,z = model.module.Encoder(init_tensor,args.batch_size,1)
            
            for i in range(1,args.problem_trace_len):
                rec = model.module.Decoder(z[:,:,0:2], args.batch_size,1)
                pi,_ = model.module.Encoder(rec,args.batch_size,1)
                rec = rec.view(-1,1,*args.image_shape)

                rec_next = image_tensor[:,i].view(args.batch_size,1,*args.image_shape) * mask_tensor[:,i].view(args.batch_size,1,*args.image_shape)
                _,z_next = model.module.Encoder(rec_next,args.batch_size,1)
                target_state_set.append(z_next[:,:,0:2])

                z, pred_qy, hidden_state = model_rnn(z, action_tensor[:,i], hidden_state)
                z = z.view(args.batch_size,args.cate_num,args.cate_dim)
                pred_rec = model.module.Decoder(z[:,:,0:2], args.batch_size,1)
                pred_rec = pred_rec.view(-1,1,*args.image_shape)

                predict_state_set.append(z)
                rec_set.append(rec)
                qy_set.append(F.softmax(pi, dim=-1).view(-1,1,args.cate_num,args.cate_dim))
                pred_set.append(pred_rec)
                pred_qy_set.append(pred_qy.view(-1,1,args.cate_num,args.cate_dim-1))

            rec_batch = torch.stack(rec_set,dim = 1).view(args.batch_size,-1,*args.image_shape)
            pred_batch = torch.stack(pred_set,dim = 1).view(args.batch_size,-1,*args.image_shape)
            qy = torch.stack(qy_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim)
            pred_qy = torch.stack(pred_qy_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim-1)
            # target_state = torch.stack(target_state_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim)
            # predict_state = torch.stack(predict_state_set,dim = 1).view(args.batch_size,-1,args.cate_num,args.cate_dim)
            
            ################ VAE loss
            bce_rec, kld_rec = self.loss_function(pred=rec_batch*mask_tensor[:,:-1],
                                            target=rec_target, 
                                            qy=qy)
            bce_pred, kld_pred = self.loss_function(pred=pred_batch*mask_tensor[:,1:],
                                            target=pred_target, 
                                            qy=pred_qy)
            # bce_state,kld_state = self.loss_function(pred = predict_state,
            #                         target = target_state,
            #                         qy = pred_qy)
            

            bce_loss_rec += bce_rec.data.item()
            kld_loss_rec += kld_rec.data.item()
            bce_loss_pred += bce_pred.data.item()
            kld_loss_pred += kld_pred.data.item()
            # bce_loss_state += bce_state.data.item()

            loss = bce_pred 
        
            if (epoch % 50 == 0 or epoch == model.module.args.rnn_num_epoch or epoch == 1) and (batch_idx %5 == 0):
                title = 'rnn'
                comparison = torch.cat([
                                        image_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        mask_tensor[:,:-1][0].view(-1,1,*args.image_shape),
                                        rec_batch[0].view(-1,1,*args.image_shape),
                                        rec_target[0].view(-1,1,*args.image_shape),
                                        pred_batch[0].view(-1,1,*args.image_shape),
                                        pred_target[0].view(-1,1,*args.image_shape),
                                        ]
                                        )
                filename = 'results_'+domain+'/test/'+title+'/test_'+title+'_%04d_%04d.png' % (epoch,batch_idx) 
                save_image(comparison.data.cpu(),
                            model.module.sub_path(filename), nrow=rec_batch.shape[1])
        
        bce_loss_rec /= len(test_loader.dataset) * (args.trace_len)
        kld_loss_rec /= len(test_loader.dataset) * (args.trace_len)
        bce_loss_pred /= len(test_loader.dataset) * (args.trace_len)
        kld_loss_pred /= len(test_loader.dataset) * (args.trace_len)
        bce_loss_state /= len(test_loader.dataset) * (args.trace_len)
        
        return bce_loss_rec,kld_loss_rec,bce_loss_pred,kld_loss_pred,bce_loss_state
