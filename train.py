import os
import argparse
from main import Train


parser = argparse.ArgumentParser()
parser.add_argument('--domain',type=str,help='domain_name',default='mnist')
parser.add_argument('--mask',type=int,help='mask number',default=0)
parser.add_argument('--mask_length',type=int,help='length of a mask',default=3)
parser.add_argument('--skip_prob',type=int,help='length of a mask',default=0)
parser.add_argument('--skip_vae',type=int,help='length of a mask',default=0)
parser.add_argument('--skip_rnn',type=int,help='length of a mask',default=0)
parser.add_argument('--skip_dist',type=int,help='length of a mask',default=0)
parser.add_argument('--check',type=int,help='check or not, default:False',default=1)
parser.add_argument('--gpu',type=int,help='gpu',default=3)
parser.add_argument('--completion',type=int,help='completion or not',default=0)



args = parser.parse_args()
print('-'*50 + 'Start' + '-'*50)
print('-Domain:',args.domain)
print('-Mask Num:',args.mask)
print('-Mask Size:',args.mask_length,'x',args.mask_length)
print('-gpu:',args.gpu)


Train(args.domain,args.mask,args.mask_length,args.skip_prob,args.skip_vae,args.skip_rnn,args.check,args.skip_dist,args.gpu,args.completion)
print('-'*50 + 'End' + '-'*50)