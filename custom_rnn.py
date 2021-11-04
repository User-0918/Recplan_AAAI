import math
import os
import ipdb
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from util import gumbel_softmax, activation_func, Flatten
from blocked_sae import GumbelConvAE

class Probability(nn.Module):
    def __init__(self, args):
        super(Probability, self).__init__()
        self.args = args
        self.args.probability_hidden_dim = 400

        if self.args.image_shape == (42,42):
            linear_dim = 800
        if self.args.image_shape == (32,96):
            linear_dim = 1410

        self.image_1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=2, kernel_size=(3,3), bias=False),
                                            #  nn.Tanh(),                                                            
                                             nn.Dropout2d(p=self.args.dropout),                                                            
                                             nn.BatchNorm2d(num_features=2, eps=1e-3),                                                            
                                             nn.MaxPool2d(kernel_size=(2,2)),                                                            
                                             Flatten(),
                                             
                                             nn.Linear(linear_dim, 400 ,bias=True),                      
                                             )
        self.image_2 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=2, kernel_size=(3,3), bias=False),
                                            #  nn.Tanh(),                                                            
                                             nn.Dropout2d(p=self.args.dropout),                                                            
                                             nn.BatchNorm2d(num_features=2, eps=1e-3),                                                            
                                             nn.MaxPool2d(kernel_size=(2,2)),                                                            
                                             Flatten(),

                                             nn.Linear(linear_dim, 400 ,bias=True),                       
                                             )

        self.action = nn.Sequential(nn.Linear(self.args.action_dim , 400 ,bias=True),                                                            
                                    )
                                    
        self.output_1 = nn.Sequential(
                                    nn.BatchNorm1d(400),
                                    nn.Dropout(p=self.args.dropout),

                                    nn.Linear(400 , int(np.prod(self.args.input_size)) ,bias=True),                                                            
                                    nn.Tanh(),    )
        self.path_setting()
    
    def path_setting(self):
        self.path = self.args.model_path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
    
    def sub_path(self, path):
        return os.path.join(self.path, path)
    
    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))

    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))

    def forward(self,first_image,second_image,action):
        # print(first_image.shape)
        # print(second_image.shape)
        # print(action.shape)
        # inputs_image = torch.stack([first_image,second_image],dim = 1).view(-1,2,*self.args.image_shape)

        layer_1 = self.image_1(first_image.contiguous().view(-1,1,*self.args.image_shape))
        layer_2 = self.image_2(second_image.contiguous().view(-1,1,*self.args.image_shape))
        layer_3 = self.action(action.contiguous().view(-1,self.args.action_dim))

        layer_4 = torch.mul(layer_1,layer_3) + torch.mul(layer_2,layer_3)

        Probability = self.output_1(layer_4).view(-1,first_image.shape[1],*self.args.image_shape)
        
        return Probability

class GumbelVAE_Transition(nn.Module):
    def __init__(self, args):
        super(GumbelVAE_Transition, self).__init__()
        self.args = args
        self.create_encoder()
        # self.create_transition()
        self.create_decoder()
        self.temperature_setting()
        self.path_setting()
        # input_dim = args.cate_num * args.cate_dim
        # output_dim = input_dim
        # hidden_dim = 400
        # dropout = args.dropout

        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Dropout(p=dropout),
            
        #     nn.Linear(hidden_dim, output_dim),
        # )

    def path_setting(self):
        self.path = self.args.model_path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
    
    def sub_path(self, path):
        return os.path.join(self.path, path)
    
    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))

    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))
    
    def temperature_setting(self):
        self.min_temperature = self.args.min_temperature
        self.max_temperature = self.args.max_temperature
        self.num_epoch = self.args.num_epoch
        self.anneal_rate = math.log(self.max_temperature/self.min_temperature) / self.num_epoch
        self.temperature = self.max_temperature

    def set_curr_temp(self, temp):
        self.temperature = temp
    
    def update_temp(self, epoch):
        self.temperature = max([self.min_temperature, self.max_temperature * math.exp(-self.anneal_rate*epoch)])
        return self.temperature

    def create_encoder(self):
        input_size = self.args.input_size
        hidden_dim = self.args.hidden_dim
        conv_channels = self.args.conv_channels
        output_dim = self.args.cate_num * self.args.cate_dim
        dropout = self.args.dropout

        if self.args.image_shape == (42,42):
            linear_dim = conv_channels*(((input_size[-1]-2)//2-2)//2)**2
        if self.args.image_shape == (32,96):
            linear_dim = 2112

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=conv_channels, kernel_size=(3,3), bias=False),
            nn.Tanh(),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(num_features=conv_channels, eps=1e-3),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(in_channels=conv_channels,out_channels=conv_channels, kernel_size=(3,3), bias=False),
            nn.Tanh(),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(num_features=conv_channels, eps=1e-3),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            Flatten(),
            
            # manually calculated
            nn.Linear(linear_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            # nn.softmax(),

        )
    
    def create_decoder(self):
        input_dim = self.args.cate_num * (self.args.cate_dim-1)
        hidden_dim = self.args.hidden_dim
        conv_channels = self.args.conv_channels
        output_dim = int(np.prod(self.args.input_size))
        dropout = self.args.dropout
        dropout_z = self.args.dropout_z

        self.decoder = nn.Sequential(
            Flatten(),
            *([nn.Dropout(p=dropout)] if dropout_z else []),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            # nn.Softmax()
        )
    def forward(self,input):
        # input_size = self.args.input_size
        # cate_num = self.args.cate_num
        # cate_dim = self.args.cate_dim
        # output_size = input_size

        # self.batch_size = input_tensor.size(0)
        # self.seq_len = input_tensor.size(1)

        # pi = self.encoder(input_tensor.reshape(-1, *input_tensor.shape[2:])).view(self.batch_size, self.seq_len, cate_num, cate_dim)
        # z = gumbel_softmax(pi, self.temperature)[:,:,:,0]
        # rec = self.decoder(z.view(-1, cate_num)).view(self.batch_size, self.seq_len, *output_size)

        # return rec, F.softmax(pi,dim=-1), z
        q = self.encoder(input.reshape(-1, *input.shape[-3:]))
        q_y = q.view([q.size(0), self.args.cate_num, self.args.cate_dim])
        z = gumbel_softmax(q_y, self.temperature)
        # z_t = z[:,:,0]  # from latplan
        z_t = z[:,:,0:2].contiguous()  # new
        # z_t = F.softmax(q_y)

        rec = self.decoder(z_t)
        # .view(*input.shape[:-3], *self.args.input_size)

        # return rec, F.softmax(q_y, dim=-1).view(*input.shape[:-3], *q_y.shape[-2:]), z_t.view(*input.shape[:-3], self.args.cate_num)
        return rec, F.softmax(q_y, dim=-1).view(*input.shape[:-3], *q_y.shape[-2:]), z.view(*input.shape[:-3],self.args.cate_num,self.args.cate_dim)
        # return rec,q_y,z_t

    def Encoder(self,input_tensor,batch_size,seq_len):
        input_size = self.args.input_size
        cate_num = self.args.cate_num
        cate_dim = self.args.cate_dim
        output_size = input_size

        pi = self.encoder(input_tensor).view(batch_size, cate_num, cate_dim)
        z = gumbel_softmax(pi, self.temperature).view(-1,self.args.cate_num,self.args.cate_dim)
        
        return pi,z

    def Decoder(self,z,batch_size,seq_len):
        input_size = self.args.input_size
        cate_num = self.args.cate_num
        cate_dim = self.args.cate_dim
        output_size = input_size
        
        
        z = z.view(batch_size,self.args.cate_num,-1)
        z_t = z[:,:,0:2].contiguous()
        
        rec = self.decoder(z_t).view(batch_size,1, *output_size)
        return rec

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.args.rnn_num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

class LinearRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias, activation='tanh'):
        super(LinearRNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation

        self.linear = nn.Linear(in_features=input_dim+hidden_dim,
                                out_features=2*hidden_dim,
                                bias=bias)
        self.activate = activation_func(activation)
    

    def forward(self, input_tensor, cur_state):

        input_tensor = input_tensor.view(input_tensor.size()[0], -1)    # flatten

        h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_linear = self.linear(combined)
        c_o, c_h = torch.split(combined_linear, self.hidden_dim, dim=1)
        
        o_next = self.activate(c_o)
        h_next = self.activate(c_h)
        
        return o_next, h_next
    
    def init_hidden(self, batch_size):

        return Variable(torch.zeros(batch_size, self.hidden_dim)).cuda()

class SigmoidTransition_RNN(nn.Module):
    def __init__(self, args,model):
        super(SigmoidTransition_RNN, self).__init__()
        self.args = args
        self.Gumbel = model
        self.create_transition()
        # self.create_linear()
        self.temperature_setting()
        self.path_setting()


    def path_setting(self):
        self.path = self.args.model_path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
    
    def sub_path(self, path):
        return os.path.join(self.path, path)
    
    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))

    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))
    
    def temperature_setting(self):
        self.min_temperature = self.args.min_temperature
        self.max_temperature = self.args.max_temperature
        self.num_epoch = self.args.num_epoch
        self.anneal_rate = math.log(self.max_temperature/self.min_temperature) / self.num_epoch
        self.temperature = self.max_temperature


    def set_curr_temp(self, temp):
        self.temperature = temp
    
    def update_temp(self, epoch):
        self.temperature = max([self.min_temperature, self.max_temperature * math.exp(-self.anneal_rate*epoch)])
        return self.temperature

    def create_transition(self):
        # input_dim = self.args.cate_num * (self.args.cate_dim-1)  + self.args.action_dim
        # hidden_dim = self.args.hidden_dim
        # output_dim = self.args.cate_num * (self.args.cate_dim-1)
        if 'blocks' in self.args.domain_name or 'hanoi' in self.args.domain_name :
            input_dim = self.args.cate_num * (self.args.cate_dim-1)  + self.args.action_dim
            hidden_dim = self.args.hidden_dim
            output_dim = self.args.cate_num * (self.args.cate_dim-1)
        else:
            input_dim = self.args.cate_num * (self.args.cate_dim)  + self.args.action_dim
            hidden_dim = self.args.hidden_dim
            output_dim = self.args.cate_num * (self.args.cate_dim)

        cell_list = []
        cur_dim = input_dim

        # if 'blocks' not in self.args.domain_name:
        self.rnn_1_h = nn.Linear(self.args.rnn_hidden_dim,self.args.rnn_hidden_dim)
        self.rnn_1_x = nn.Linear(input_dim,self.args.rnn_hidden_dim)
        self.rnn_1_o = nn.Linear(self.args.rnn_hidden_dim,self.args.rnn_linear_dim)

        self.linear = nn.Sequential(
                    nn.Linear(self.args.rnn_linear_dim,self.args.rnn_linear_dim,bias = True),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.args.rnn_linear_dim),
                    nn.Dropout(p=self.args.dropout),

                    nn.Linear(self.args.rnn_linear_dim, output_dim,bias = True),
        )   

    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))

    def _init_hidden(self, batch_size):
        # init_states = []
        # for i in range(self.args.rnn_num_layers):
        #     init_states.append(self.cell_list[i].init_hidden(batch_size))
        # if 'blocks' in self.args.domain_name:
        #     init_states = [Variable(torch.zeros(batch_size, self.args.rnn_hidden_dim)).cuda() for x in range(2)]
        # else:
        init_states = Variable(torch.zeros(batch_size, self.args.rnn_hidden_dim)).cuda() 
        return init_states

    def forward(self, z, action_tensor, hidden_state):
        # import ipdb
        # ipdb.set_trace()
        input_size = self.args.input_size
        cate_num = self.args.cate_num
        cate_dim = self.args.cate_dim
        
        seq_len = 1
        batch_size = z.size(0)

        # if 'blocks' not in self.args.domain_name:
        if 'blocks' in self.args.domain_name or 'hanoi' in self.args.domain_name :
            output_size = self.args.cate_num * (self.args.cate_dim-1)
            inputs = torch.cat([z[:,:,0:2].contiguous().view(batch_size, -1), action_tensor], dim=-1)
        else:
            output_size = self.args.cate_num * (self.args.cate_dim)
            inputs = torch.cat([z.contiguous().view(batch_size, -1), action_tensor], dim=-1)
        
        next_hidden_state = torch.relu(self.rnn_1_h(hidden_state) + self.rnn_1_x(inputs))
        output = torch.relu(self.rnn_1_o(next_hidden_state))

        if 'blocks' in self.args.domain_name or 'hanoi' in self.args.domain_name :
            pred_z = gumbel_softmax(self.linear(output).view(-1, seq_len, cate_num, cate_dim-1),self.temperature)
            pred_pi = torch.stack([pred_z, 1-pred_z], dim=-1).view(-1, seq_len, cate_num, cate_dim-1) # 无意义
        else:
            pred_z = gumbel_softmax(self.linear(output).view(-1, seq_len, cate_num, cate_dim),self.temperature)
            pred_pi = torch.stack([pred_z, 1-pred_z], dim=-1).view(-1, seq_len, cate_num, cate_dim) # 无意义
        return pred_z, F.softmax(pred_pi, dim=-1), next_hidden_state
