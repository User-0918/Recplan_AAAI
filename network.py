import torch
import torch.nn as nn
import os

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        self.build()
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
    
    def forward(self, input_tensor):
        state = input_tensor[:,0:self.args.cate_num * (self.args.cate_dim-1) *2]
        action = input_tensor[:,self.args.cate_num * (self.args.cate_dim-1) * 2:]

        # if 'hanoi' in self.args.domain_name:
        #     layer1 = torch.sigmoid( self.action_net(action) + self.state_net(state))
        #     return self.net(layer1)
    
        #     print(input_tensor.shape)

        #     input_tensor = input_tensor.view(1,2,int(input_tensor.shape[0]/2),162)
        # print(self.net(input_tensor).shape)
        # import sys
        # sys.exit()
        return self.net(input_tensor) 
        # else:
        #     a = self.action_net(input_tensor[:,self.args.cate_num * (self.args.cate_dim-1) *  2:])
        #     return  self.net(torch.cat([input_tensor[:,0:self.args.cate_num * (self.args.cate_dim-1) *  2],
        #                                   a],dim =-1))
            
class HeurNet(Network):
    def build(self):
        input_dim = self.args.cate_num * 3  # (hidden_states, curr_latent_stete, goal_latent_state)
        latent_dim = input_dim
        output_dim = self.args.action_dim

        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, output_dim),
        #     nn.LogSoftmax(dim=-1)
        # )
        ## (batch_size*8 x action_dim(4)) ??

class DistNet(Network):
    def build(self):
        input_dim = self.args.cate_num * (self.args.cate_dim-1) *  2 + self.args.action_dim 
        output_dim = 1 
        if 'hanoi' not in self.args.domain_name:
            latent_dim = input_dim * 2
            self.net = nn.Sequential(
                nn.Linear(input_dim,latent_dim),
                nn.LeakyReLU(),
            
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU(),
                
                nn.Linear(latent_dim, output_dim)
            )
        else:
            latent_dim = 1000
            self.net = nn.Sequential(
                nn.Linear(input_dim,latent_dim),
                nn.LeakyReLU(),
            
                nn.Linear(latent_dim,50),
                nn.LeakyReLU(),
                
                # nn.Linear(latent_dim, 50),
                nn.Linear(50, output_dim)
            )
        
        
    

#%%
class StateDiscriminator(nn.Module):
    def __init__(self, args):
        super(StateDiscriminator, self).__init__()
        self.args = args
        self.c = 1
        self.input_dim = self.args.cate_num
        self.build()
        
        self.path = self.args.model_path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def sub_path(self, path):
        return os.path.join(self.path, path)

    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))
    
    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))

    def forward(self, input_tensor):
        self.s = self.discriminator(input_tensor)
        return self.s, self.s / self.c

    def build(self):
        self.discriminator = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.args.sd_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.args.sd_dropout),

            nn.Linear(self.args.sd_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def set_c(self, c):
        self.c = c

#%%
class ActionDiscriminator(nn.Module):
    def __init__(self, args):
        super(ActionDiscriminator, self).__init__()
        self.args = args
        self.c = 1
        self.input_dim = self.args.cate_num + self.args.cate_num
        self.build()
        
        self.path = self.args.model_path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def sub_path(self, path):
        return os.path.join(self.path, path)

    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))
    
    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))

    def forward(self, input_tensor):
        self.s = self.discriminator(input_tensor)
        return self.s, self.s / self.c

    def build(self):
        self.discriminator = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.args.ad_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.args.ad_dropout),

            nn.Linear(self.args.ad_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def set_c(self, c):
        self.c = c