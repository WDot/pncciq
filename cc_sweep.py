import torch
import numpy as np
import argparse
import os
import os.path
from util import IOStream
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from estimators.fcc import fcc
from estimators.cc import cc
from estimators.cfo import cfo
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset,random_split

torch.backends.cuda.enable_flash_sdp(True)
device = 'cuda'

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/models'):
        os.makedirs('checkpoints/' + args.exp_name + '/models')

    os.system('cp main.py checkpoints/'+args.exp_name + '/main.py.backup')
    os.system('cp model.py checkpoints/'+args.exp_name + '/model.py.backup')
    os.system('cp tfdoa_analytic.py checkpoints/'+args.exp_name + '/tfdoa_analytic.py.backup')

def get_correlations(func,signal1,signal2,taus1d):
    correlations = func(signal1,signal2,taus1d)
    ind = torch.unravel_index(torch.argmax(correlations,axis=-1),correlations.shape)
    maxtau = taus1d[ind[0],ind[1],0]
    return maxtau

def train(args):
    
    total_size = 200000
    train_size = int(0.8 * total_size)
    val_size = int((total_size - train_size) / 2.0)

    with np.load('tdelays20250506.npz') as data:
        tensors = [torch.from_numpy(data[key]) for key in ['noisy_signals_passband','noisy_signals_passband_shifted','noisy_signals_passband_delayed', 'mod_labels','tdelays','snrs','fcs','fc2s']]


    full_dataset = TensorDataset(*tensors)    
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    valtest_size = total_size - train_size
    val_size = int(0.1*total_size)
    test_size = valtest_size - val_size
    _, valtest_dataset = random_split(full_dataset,[train_size,valtest_size], generator=torch.Generator().manual_seed(args.seed))

    _, test_dataset = random_split(valtest_dataset,[val_size,test_size], generator=torch.Generator().manual_seed(args.seed))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,num_workers=0)
    if args.task == 'td':
        estimator = fcc
    elif args.task == 'ds':
        estimator = cc
    else:
        estimator = cfo

    if args.task == 'td':
        scale = args.time_scale
    else:
        scale = args.freq_scale

    denoiser = torch.load('checkpoints/%s/denoiser.pth' % args.denoiser_exp).cuda()
    denoiser.eval()

    test_gt_lambdas = []
    test_pred_lambdas = []
    test_snrs = []
    io.cprint('Test Mode!')

    lambdahats = torch.reshape(torch.linspace(-scale,scale,args.num_hypotheses,device=device),[1,-1,1])
    with torch.no_grad():
        for noisy_passband,noisy_passband_shifted,noisy_passband_delayed,label,true_tau,snr,fc,fc2 in test_loader:
            label = label.to(device)
            if args.task == 'td':
                signal1 = torch.unsqueeze(noisy_passband,1).to(device)
                signal2 = torch.unsqueeze(noisy_passband_delayed,1).to(device)
                true_lambda = true_tau.to(device)
            elif args.task == 'ds':
                signal1 = torch.unsqueeze(noisy_passband,1).to(device)
                signal2 = torch.unsqueeze(noisy_passband_shifted,1).to(device)
                true_lambda = fc2.to(device)
            else:
                signal1 = torch.unsqueeze(noisy_passband,1).to(device)
                signal2 = torch.ones([signal1.shape[0],1,signal1.shape[-1]],device=device,dtype=torch.complex64)
                true_lambda = fc.to(device)

            cleansignal1raw = denoiser(torch.cat((torch.real(signal1),\
                                    torch.imag(signal1)),1))
            if args.task != 'cf':
                cleansignal2raw = denoiser(torch.cat((torch.real(signal2),\
                                        torch.imag(signal2)),1))
            else:
                cleansignal2raw = torch.cat((torch.real(signal2),\
                                        torch.imag(signal2)),1)
            
            cleansignal1 = torch.complex(cleansignal1raw[:,0:1,:],cleansignal1raw[:,1:2,:])
            cleansignal2 = torch.complex(cleansignal2raw[:,0:1,:],cleansignal2raw[:,1:2,:])

            est_lambda = get_correlations(estimator,cleansignal1,cleansignal2,lambdahats)

            tdloss = torch.mean(1 - estimator(signal1,signal2,torch.reshape(est_lambda,[-1,1,1])))
            loss = tdloss

            loss =  loss.detach().cpu().numpy()
            test_pred_lambdas.append(est_lambda.detach().cpu().numpy())
            test_gt_lambdas.append(true_lambda.detach().cpu().numpy())
            test_snrs.append(snr.detach().cpu().numpy())

    test_pred_lambdas = np.concatenate(test_pred_lambdas)
    test_gt_lambdas = np.concatenate(test_gt_lambdas)
    test_snrs = np.concatenate(test_snrs)
    test_mse = mean_squared_error(test_gt_lambdas,test_pred_lambdas)

    io.cprint('Test: Loss: {0} EstLambda: {1} TruLambda: {2} MSE: {3}'.format(
                                                        loss,\
                                                        (est_lambda[0]).detach().cpu().numpy(),\
                                                        (true_lambda[0]).detach().cpu().numpy(),\
                                                        test_mse))
    STEP = 2
    for i in range(-20,20,STEP):
        indices = np.where((test_snrs> i) & (test_snrs < i + STEP))
        cur_snr_mse = mean_squared_error(test_gt_lambdas[indices],test_pred_lambdas[indices])
        io.cprint('Test SNR < {0}: MSE: {1}'.format(i + STEP, cur_snr_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TFDOA Estimator')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--time_scale',type=float)
    parser.add_argument('--freq_scale',type=float)
    parser.add_argument('--transform',choices=['identity','neural','gccphat','gccht'])
    parser.add_argument('--task', choices=['td', 'ds', 'cf'])
    parser.add_argument('--denoiser_exp',type=str,default='exp')
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--num_hypotheses',type=int)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    train(args)
