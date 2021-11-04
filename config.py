import math
import os
import random

import numpy as np
import torch
import time

def set_seed(seed=45):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()      # fixed seed for reproduction
class Args:
    def __init__(self):

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}x{}.npz'.format(self.trace_num, self.trace_len))
        self.problem_path = os.path.join('data', '8puzzle_{}x{}_problem.npz'.format(self.trace_num, self.trace_len))
        
        # training args
        self.batch_size = 1000
        self.learning_rate = 0.003
        self.num_epoch = 150
        self.shuffle = True,

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_channels = [4,2,2]
        self.kernel_size = [(3,3),(3,3),(3,3)]
        self.num_latent_layers = 1
        self.latent_dim = 72
        self.num_transition_layers = 1
        self.action_dim = 4
        self.transition_dim = 72
        self.num_decoder_layers = 2
        self.decoder_dim = [128,42*42]
        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/HeurTransition'
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

args = Args()

class Args2:
    def __init__(self):
        
        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}x{}.npz'.format(self.trace_num, self.trace_len))
        self.problem_path = os.path.join('data', '8puzzle_{}x{}_problem.npz'.format(self.trace_num, self.trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 300
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 1000
        self.rnn_hidden_dim = 400
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN'
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain_no_imageloss:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 1000
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN_{}_{}_noimageloss'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        if domain == 'mnist':
            self.problem_trace_num = 1000
        else:
            self.problem_trace_num = 100
        self.problem_trace_num = 1000
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN_{}_{}'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain_change:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 1000
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN_{}_{}_change'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain_probability:
    def __init__(self, domain, num_blocks,mask_length):
        
        self.probability_epoch = 500
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = mask_length

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        if 'hanoi' in domain:
            self.trace_num = 6000
            self.image_shape = (32,96)
        self.trace_len = 8
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 1000
        if domain == 'hanoi' or 'blocks' in domain:
            self.problem_trace_num = 100
        self.problem_trace_len = self.trace_len
        if 'path' not in domain:
            if 'hanoi' in domain or 'blocks' in domain:
                self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.pkl'.format(self.domain_name, self.trace_num, self.trace_len))
                self.mask_path = os.path.join('data', 'mask_{}_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size,self.trace_len))
                self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_test.pkl'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
                self.fixed_goal_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
            else:
                self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
                self.mask_path = os.path.join('data', 'mask_{}_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size,self.trace_len))
                self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
                self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        else:
            self.data_path = os.path.join('data', '{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
            self.mask_path = os.path.join('data', 'mask_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size))
            self.problem_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
            self.fixed_goal_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        # training args
        self.batch_size = 500
        if  'blocks' in domain:
            self.batch_size = 250
        if 'hanoi' in domain:
            self.batch_size = 250
        self.learning_rate = 0.001
        self.num_epoch = 2000
        self.image_epoch = 500
        self.vae_epoch = 1000
        self.shuffle = True
        self.dist_epoch = 1500
        if 'hanoi' in self.domain_name or 'blocks' in self.domain_name:
            self.dist_epoch = 1200

        # model args
        self.input_channels = 2
        self.input_size = self.image_shape
        self.num_encoder_layers = 3
        
        self.hidden_dim = 1000
        self.conv_channels = 16
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 1500
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 1000 ## RNN input hidden state
        if 'blocks' in domain:
            self.hidden_dim = 1500
            self.rnn_linear_dim = 250
            # self.rnn_hidden_dim = 1000
        if 'hanoi' in domain:
            self.rnn_linear_dim = 250
            # self.rnn_hidden_dim = 1000

        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        if 'blocks' in self.domain_name or 'hanoi' in self.domain_name:
            self.model_path = 'models_new/{}_{}_{}'.format(self.domain_name, self.num_blocks,self.blocks_size)
        else:
            self.model_path = 'models/{}_{}_{}'.format(self.domain_name, self.num_blocks,self.blocks_size)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        # if  'hanoi' in domain:
        #     self.cate_num = 18
        # if 'blocks' in domain:
        #     self.cate_dim = 18
        # self.cate_num = 50 ## size of encoded state of observation done by VAE
        self.cate_dim = 3 ## 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain_no_completion:
    def __init__(self, domain, num_blocks,mask_length):
        
        self.probability_epoch = 500
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = mask_length

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        if 'hanoi' in domain:
            self.trace_num = 6000
            self.image_shape = (32,96)
        self.trace_len = 8
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 1000
        if domain == 'hanoi' or 'blocks' in domain:
            self.problem_trace_num = 100
        self.problem_trace_len = self.trace_len
        if 'path' not in domain:
            if 'hanoi' in domain or 'blocks' in domain:
                self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.pkl'.format(self.domain_name, self.trace_num, self.trace_len))
                self.mask_path = os.path.join('data', 'mask_{}_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size,self.trace_len))
                self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_test.pkl'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
                self.fixed_goal_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
            else:
                self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
                self.mask_path = os.path.join('data', 'mask_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size))
                self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
                self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        else:
            self.data_path = os.path.join('data', '{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
            self.mask_path = os.path.join('data', 'mask_{}_{}_{}.npz'.format(self.domain_name, self.num_blocks, self.blocks_size))
            self.problem_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
            self.fixed_goal_path = os.path.join('data', '{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        # training args
        self.batch_size = 500
        if  'blocks' in domain:
            self.batch_size = 250
        if 'hanoi' in domain:
            self.batch_size = 250
        self.learning_rate = 0.001
        self.num_epoch = 2000
        # self.num_epoch = 1
        self.image_epoch = 500
        self.vae_epoch = 1000
        # self.image_epoch = 1
        # self.vae_epoch = 1
        self.shuffle = True
        self.dist_epoch = 1500
        # self.dist_epoch = 1
        if 'hanoi' in self.domain_name or 'blocks' in self.domain_name:
            self.dist_epoch = 1200

        # model args
        self.input_channels = 2
        self.input_size = self.image_shape
        self.num_encoder_layers = 3
        
        self.hidden_dim = 1000
        self.conv_channels = 16
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 1500
        # self.rnn_num_epoch = 1
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 1000 ## RNN input hidden state
        if 'blocks' in domain:
            self.hidden_dim = 1500
            self.rnn_linear_dim = 250
            # self.rnn_hidden_dim = 1000
        if 'hanoi' in domain:
            self.rnn_linear_dim = 250
            # self.rnn_hidden_dim = 1000

        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        if 'blocks' in self.domain_name or 'hanoi' in self.domain_name:
            self.model_path = 'models_new/{}_{}_{}'.format(self.domain_name, self.num_blocks,self.blocks_size)
        else:
            self.model_path = 'models_no_completion/{}_{}_{}'.format(self.domain_name, self.num_blocks,self.blocks_size)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        # if  'hanoi' in domain:
        #     self.cate_num = 18
        # if 'blocks' in domain:
        #     self.cate_dim = 18
        # self.cate_num = 50 ## size of encoded state of observation done by VAE
        self.cate_dim = 3 ## 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch


class Args_domain_inpainting:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/{}_{}_inpainting'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch


class Args_domain_together_inpainting:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN_{}_{}_inpainting'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class Args_domain_state:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/{}_{}'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch


class Args_domain_classifier:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks
        self.blocks_size = 3

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # if domain == 'mnist':
        #     self.problem_trace_num = 1000
        # else:
        #     self.problem_trace_num = 100
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 1000
        self.image_epoch = 500
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 3000
        self.rnn_linear_dim = 400
        self.rnn_hidden_dim = 400 ## RNN input hidden state
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_RNN_{}_{}_state'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36 ## size of encoded state of observation done by VAE
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch


class Args_domain_nn:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        
        # training args
        self.batch_size = 500
        self.learning_rate = 0.001
        self.num_epoch = 300
        self.shuffle = True

        # model args
        self.input_channels = 2
        self.input_size = (42,42)
        self.num_encoder_layers = 3
        self.hidden_dim = 1000
        self.conv_channels = 4
        self.action_dim = 4

        self.dropout =  0.4
        self.dropout_z = False

        self.rnn_learning_rate = 0.001
        self.rnn_num_epoch = 1000
        self.rnn_hidden_dim = 400
        self.rnn_num_layers = 2

        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.model_path = 'models/SigmoidTransition_NN_{}_{}'.format(self.domain_name, self.num_blocks)
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.cate_num = 36
        self.cate_dim = 2
        self.anneal_rate = math.log(self.max_temperature / self.min_temperature) / self.num_epoch

class LatplanArgs:
    def __init__(self):
        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}x{}.npz'.format(self.trace_num, self.trace_len))
        self.problem_path = os.path.join('data', '8puzzle_{}x{}_problem.npz'.format(self.trace_num, self.trace_len))

        # training args
        self.batch_size = 500 #4000
        self.learning_rate = 0.001
        self.num_epoch = 300
        self.shuffle = True

        # model args
        self.hidden_dim = 1000
        self.conv_channels = 16
        self.dropout = 0.4
        self.noise = 0.4
        self.dropout_z = False
        self.activation = 'tanh'
        self.cate_num = 36
        self.cate_dim = 2
        self.input_size = (42,42)
        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.action_dim = 4
        self.model_path = 'models/latplan'
    
        # training args (aae)
        self.aae_batch_size = 250 # 2000
        self.aae_learning_rate = 0.001
        self.aae_num_epoch = 1000
        self.aae_shuffle = True

        # model args (aae)
        self.aae_cate_num = 1
        self.aae_cate_dim = 4
        self.aae_hidden_dim = 400
        self.aae_encoder_layers = 2
        self.aae_decoder_layers = 2
        self.aae_dropout = 0.4
        self.aae_dropout_z = False
        self.aae_full_epoch = 1000
        self.aae_encoder_activation = 'relu'
        self.aae_decoder_activation = 'relu'

class LatplanArgs_domain:
    def __init__(self, domain, num_blocks):
        
        self.domain_name = domain
        self.num_blocks = num_blocks

        # data args
        self.image_shape = (42,42)
        self.trace_num = 10000
        self.trace_len = 8
        self.data_path = os.path.join('data', '8puzzle_{}_{}x{}.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        # self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_problem.npz'.format(self.domain_name, self.trace_num, self.trace_len))
        self.problem_trace_num = 100
        self.problem_trace_len = 8
        self.problem_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))
        self.fixed_goal_path = os.path.join('data', '8puzzle_{}_{}x{}_fixed_goal.npz'.format(self.domain_name, self.problem_trace_num, self.problem_trace_len))

        # training args
        self.batch_size = 500 #4000
        self.learning_rate = 0.001
        self.num_epoch = 300
        self.shuffle = True

        # model args
        self.hidden_dim = 1000
        self.conv_channels = 16
        self.dropout = 0.4
        self.noise = 0.4
        self.dropout_z = False
        self.activation = 'tanh'
        self.cate_num = 36
        self.cate_dim = 2
        self.input_size = (42,42)
        self.min_temperature = 0.7
        self.max_temperature = 5.0
        self.action_dim = 4
        self.model_path = 'models/latplan_{}_{}'.format(self.domain_name, self.num_blocks)
    
        # training args (aae)
        self.aae_batch_size = 250 # 2000
        self.aae_learning_rate = 0.001
        self.aae_num_epoch = 1000
        self.aae_shuffle = True

        # model args (aae)
        self.aae_cate_num = 1
        self.aae_cate_dim = 4
        self.aae_hidden_dim = 400
        self.aae_encoder_layers = 2
        self.aae_decoder_layers = 2
        self.aae_dropout = 0.4
        self.aae_dropout_z = False
        self.aae_full_epoch = 1000
        self.aae_encoder_activation = 'relu'
        self.aae_decoder_activation = 'relu'

        # training args (sd)
        self.sd_batch_size = 1000 # 2000
        self.sd_learning_rate = 0.0001
        self.sd_num_epoch = 3000
        self.sd_shuffle = True

        # model args (sd)
        self.sd_hidden_dim = 50
        self.sd_dropout = 0.8
        self.sd_dropout_z = False
        self.sd_full_epoch = 1000
        self.sd_activation = 'relu'

        # training args (ad)
        self.ad_batch_size = 1000 # 2000
        self.ad_learning_rate = 0.0001
        self.ad_num_epoch = 3000
        self.ad_shuffle = True

        # model args (ad)
        self.ad_hidden_dim = 300
        self.ad_dropout = 0.8
        self.ad_dropout_z = False
        self.ad_full_epoch = 1000
        self.ad_activation = 'relu'