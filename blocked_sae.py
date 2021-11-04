#%%
import datetime
import math
import os
import random

import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.patches import Rectangle
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import args
from masked_puzzle import manmade_blocks, mask, traces
from util import (append_file, batch_generator, gumbel_softmax, save_image,
                  write_file)


#%%
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def forward(self, input, input_shape):
        return input.view(input.size(0), input_shape)

class GumbelConvAE(nn.Module):
    def __init__(self, parameters):
        super(GumbelConvAE, self).__init__()
        self.params = parameters
        self.create_encoder(self.params['input_shape'])
        self.create_decoder(self.params['input_shape'])
        self.temperature_setting()
        
        self.path = self.params['model_path']
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def forward(self, input):
        q = self.encoder(input)
        q_y = q.view([q.size(0), self.params['N'], self.params['M']])
        z = gumbel_softmax(q_y, self.temperature)
        z_t = z[:,:,0]  # from latplan
        rec = self.decoder(z_t).view(-1, *self.params['input_shape'])
        return rec, F.softmax(q_y, dim=-1), z_t

    def sub_path(self, path):
        return os.path.join(self.path, path)

    def save(self, path):
        torch.save(self.state_dict(), self.sub_path(path))
    
    def load(self, path):
        self.load_state_dict(torch.load(self.sub_path(path)))

    def create_encoder(self, input_shape):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=self.params['clayer'], kernel_size=(3,3), bias=False),
            nn.Tanh(),
            nn.Dropout2d(p=self.params['dropout']),
            nn.BatchNorm2d(num_features=self.params['clayer'], eps=1e-3),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(in_channels=self.params['clayer'],out_channels=self.params['clayer'], kernel_size=(3,3), bias=False),
            nn.Tanh(),
            nn.Dropout2d(p=self.params['dropout']),
            nn.BatchNorm2d(num_features=self.params['clayer'], eps=1e-3),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            Flatten(),

            nn.Linear(self.params['clayer']*(((input_shape[-1]-2)//2-2)//2)**2, self.params['layer']),
            nn.BatchNorm1d(self.params['layer']),
            nn.Linear(self.params['layer'], self.params['N']*self.params['M'])
        )
    
    def create_decoder(self, input_shape):
        data_dim = int(np.prod(input_shape))
        self.decoder = nn.Sequential(
            Flatten(),
            *([nn.Dropout(p=self.params['dropout'])] if self.params['dropout_z'] else []),
            nn.Linear(self.params['N'], self.params['layer']),
            nn.ReLU(),
            nn.BatchNorm1d(self.params['layer']),
            nn.Dropout(p=self.params['dropout']),

            nn.Linear(self.params['layer'], self.params['layer']),
            nn.ReLU(),
            nn.BatchNorm1d(self.params['layer']),
            nn.Dropout(p=self.params['dropout']),
            
            nn.Linear(self.params['layer'], data_dim),
            nn.Sigmoid()
        )

    def temperature_setting(self):
        self.min_temperature = self.params['min_temperature']
        self.max_temperature = self.params['max_temperature']
        self.num_epoch = self.params['num_epoch']
        self.anneal_rate = math.log(self.max_temperature/self.min_temperature) / self.num_epoch
        self.temperature = self.max_temperature

    def set_curr_temp(self, temp):
        self.temperature = temp

    def update_temp(self, epoch):
        self.temperature = np.max([self.min_temperature, self.max_temperature * np.exp(- self.anneal_rate * epoch)])
        return self.temperature

class Decoder(nn.Module):
    def __init__(self, input_dim, output_shape, hidden_dim, dropout, dropout_z):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        data_dim = int(np.prod(output_shape))
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
            
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x).view([*x.shape[:-1], *self.output_shape])

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, m=1, cate_dim=2):
    # import ipdb
    # ipdb.set_trace()
    BCE = F.binary_cross_entropy(recon_x*m, x*m, size_average=False)
    MSE = F.mse_loss(recon_x*m, x*m, size_average=False)
    log_qy = torch.log(qy+1e-20)
    g = Variable(torch.log(torch.tensor([1.0/cate_dim])).cuda())
    KLD = torch.sum(qy*(log_qy - g),dim=-1).sum()
    # KLD = torch.sum(qy*(log_qy),dim=-1).mean()
    return BCE, KLD
    # return MSE

#%%
def train(model, epoch, train_data, batch_size, optimizer, log_interval):
    model.train()
    train_loss = 0
    train_bce_loss = 0
    train_kld_loss = 0
    temp = model.update_temp(epoch)
    train_loader = batch_generator(train_data, batch_size, False)
    for batch_idx, data in train_loader:
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy, z_t = model(data)
        bce_loss, kld_loss = loss_function(recon_batch, data[:,0], qy, data[:,1])
        loss = bce_loss + kld_loss
        loss.backward()
        train_loss += loss.data.item()
        train_bce_loss += bce_loss.data.item()
        train_kld_loss += kld_loss.data.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                100. * batch_idx * batch_size / len(train_data),
                loss.data.item() / len(data)))
        if batch_idx == 1:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n,0].view(-1, 1, 42, 42), 
                                    F.upsample(qy[:n,:,0].view(-1,1,6,6), size=42),
                                    F.upsample(qy[:n,:,0].view(-1,1,6,6)/temp, size=42),
                                    F.upsample(z_t[:n].view(-1,1,6,6), size=42),
                                    recon_batch.view(-1, 1, 42, 42)[:n]])
            save_image(comparison.data.cpu(),
                        model.sub_path('results/train_' + str(epoch) + '.png'), nrow=n)

    train_loss /= len(train_data)
    train_bce_loss /= len(train_data)
    train_kld_loss /= len(train_data)
    print('====> Epoch: {} Average bce loss: {:.4f}, Average kld loss: {:.4f}'.format(
          epoch, train_bce_loss, train_kld_loss))
    return train_bce_loss, train_kld_loss


def test(model, epoch, test_data, batch_size):
    model.eval()
    test_loss = 0
    test_bce_loss = 0
    test_kld_loss = 0
    temp = model.update_temp(epoch)
    test_loader = batch_generator(test_data, batch_size, False)
    for batch_idx, data in test_loader:
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, qy, z_t = model(data)
        bce_loss, kld_loss = loss_function(recon_batch, data[:,0], qy, data[:,1])
        test_bce_loss += bce_loss.data.item()
        test_kld_loss += kld_loss.data.item()
        test_loss += test_bce_loss + test_kld_loss
        if batch_idx == 1:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n,0].view(-1, 1, 42, 42),
                                  F.upsample(qy[:n,:,0].view(-1,1,6,6), size=42),
                                  F.upsample(qy[:n,:,0].view(-1,1,6,6)/temp, size=42),
                                  F.upsample(z_t[:n].view(-1,1,6,6), size=42),
                                  recon_batch.view(-1, 1, 42, 42)[:n]])
            save_image(comparison.data.cpu(),
                     model.sub_path('results/reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_data)
    test_bce_loss /= len(test_data)
    test_kld_loss /= len(test_data)
    print('====> Test set bce loss: {:.4f}, kld loss: {:.4f}'.format(test_bce_loss, test_kld_loss))
    return test_bce_loss, test_kld_loss

#%%
def run():
    parameters = {
        'layer'             :1000,# [400,4000],
        'clayer'            :16,# [400,4000],
        'dropout'           :0.4, #[0.1,0.4],
        'noise'             :0.4,
        'dropout_z'         :False,
        'activation'        :'tanh',
        'epoch'             :300,#150,
        'batch_size'        :4000,#4000,
        'lr'                :0.001,
        'N'                 :36,
        'M'                 :2,
        'input_shape'       :(42,42),
        'num_epoch'         :300,#150,
        'min_temperature'   :0.7,
        'max_temperature'   :5.0,
    }
    parameters['model_path'] = os.path.join(os.getcwd(), 'models', 'GumbelConvAE')

    log_interval = 3

    model = GumbelConvAE(parameters).cuda()
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])

    log_file = model.sub_path('record_{}.txt'.format(datetime.datetime.now().strftime('%m%d%H%M')))
    import json
    write_file(json.dumps(parameters), path=log_file)

    trace_num = 3750
    trace_len = 8
    imgs, _ = traces(3, 3, trace_num=trace_num, trace_len=trace_len)
    n_train = int(trace_num*trace_len * 0.85)

    mask_mat = torch.tensor([manmade_blocks(mask_pos=random.sample(range(0,9), 2)) for l in range(trace_len*trace_num)]).view(-1, 1, 42, 42)
    mask_mat[n_train:] = 1
    data = (torch.from_numpy(imgs).view(-1, 1, 42, 42) * mask_mat)
    data_mask = torch.cat([data, mask_mat], dim=1)

    train_data = data_mask[:n_train]
    test_data = data_mask[n_train:]
    #%%
    best_test_loss = None
    for epoch in range(1, parameters['num_epoch'] + 1):
        batch_size = parameters['batch_size']
        train_data=train_data[torch.randperm(train_data.size()[0])]
        test_data=test_data[torch.randperm(test_data.size()[0])]
        train_bce_loss, train_kld_loss = train(model, epoch, train_data, batch_size, optimizer, log_interval)
        test_bce_loss, test_kld_loss = test(model, epoch, test_data, batch_size)
        if best_test_loss is None or best_test_loss > test_loss:
            model.save('model.pkl')
            # model.save('ep_%03d' % epoch)
        
        loss_info = map(lambda x: '%.6f'%x, [train_bce_loss, train_kld_loss, test_bce_loss, test_kld_loss])
        append_file(loss_info, log_file, num=epoch)

        sample_z = Variable(torch.randn(8, 36).round()).cuda()
        sample_m = mask_mat[:8].view(-1,1,42,42).cuda()
        sample_x = model.decoder(sample_z).view(-1,1,42,42)
        sample_rec, qy, z_t = model(torch.cat([sample_x, sample_m], dim=1))
        sample_rec2, qy2, z_t2 = model(torch.cat([sample_rec.view(-1, 1, 42, 42), sample_m], dim=1))
        sample_rec3, qy3, z_t3 = model(torch.cat([sample_rec2.view(-1, 1, 42, 42), sample_m], dim=1))
        sample = torch.cat([F.upsample(sample_z.view(-1,1,6,6), size=42),
                            sample_x.view(-1, 1, 42, 42),
                            sample_m.view(-1, 1, 42, 42),
                            F.upsample(qy[:,:,0].view(-1,1,6,6), size=42),
                            F.upsample(z_t.view(-1,1,6,6), size=42),
                            sample_rec.view(-1, 1, 42, 42),
                            F.upsample(qy2[:,:,0].view(-1,1,6,6), size=42),
                            F.upsample(z_t2.view(-1,1,6,6), size=42),
                            sample_rec2.view(-1, 1, 42, 42),
                            F.upsample(qy3[:,:,0].view(-1,1,6,6), size=42),
                            F.upsample(z_t3.view(-1,1,6,6), size=42),
                            sample_rec3.view(-1, 1, 42, 42)], dim=0).cpu()
        save_image(sample.data.view(-1, 1, 42, 42),
                    model.sub_path('results/sample_' + str(epoch) + '.png'))

def test_sae(path):
    
    parameters = {
        'layer'             :1000,# [400,4000],
        'clayer'            :16,# [400,4000],
        'dropout'           :0.4, #[0.1,0.4],
        'noise'             :0.4,
        'dropout_z'         :False,
        'activation'        :'tanh',
        'epoch'             :150,
        'batch_size'        :4000,
        'lr'                :0.001,
        'N'                 :36,
        'M'                 :2,
        'input_shape'       :(42,42),
        'num_epoch'         :150,
        'min_temperature'   :0.7,
        'max_temperature'   :5.0,
    }
    parameters['model_path'] = os.path.join(os.getcwd(), 'models', 'GumbelConvAE')

    ae = GumbelConvAE(parameters)
    ae.load(path)
    sample_z = Variable(torch.randn(8, 36).round())
    sample_z = sample_z.cuda()
    sample_x = ae.decoder(sample_z).view(-1,1,42,42)
    sample_rec = ae(sample_x)[0].view(-1,1,42,42)
    sample_rec2 = ae(sample_rec)[0].view(-1,1,42,42)
    sample_rec3 = ae(sample_rec2)[0].view(-1,1,42,42)
    sample = torch.stack((sample_x, sample_rec, sample_rec2, sample_rec3), dim=0).cpu()
    save_image(sample.data.view(-1, 1, 42, 42),'test.png')


#%%
if __name__ == '__main__':
    run()

# %%
