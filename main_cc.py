import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_predictor import Predictor
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
from model_denoiser import Denoiser
import matplotlib.pyplot as plt

torch.backends.cuda.enable_flash_sdp(True)
device = 'cuda'

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/models'):
        os.makedirs('checkpoints/' + args.exp_name + '/models')
    if not os.path.exists('checkpoints/' + args.exp_name + '/plots'):
        os.makedirs('checkpoints/' + args.exp_name + '/plots')

    os.system('cp main.py checkpoints/'+args.exp_name + '/main.py.backup')
    os.system('cp model.py checkpoints/'+args.exp_name + '/model.py.backup')
    os.system('cp tfdoa_analytic.py checkpoints/'+args.exp_name + '/tfdoa_analytic.py.backup')

def train(args):
    label_dict = {'16qam': 0, '64qam': 1, '256qam': 2, 'bpsk': 3, 'qpsk': 4, '8psk': 5, 'dqpsk': 6, 'msk': 7}

    with np.load('tdelays20250506.npz') as data:
        tensors = [torch.from_numpy(data[key]) for key in ['noisy_signals_passband','noisy_signals_passband_shifted','noisy_signals_passband_delayed', 'mod_labels','tdelays','snrs','fcs','fc2s']]

    full_dataset = TensorDataset(*tensors)    
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    valtest_size = total_size - train_size
    val_size = int(0.1*total_size)
    test_size = valtest_size - val_size
    print(train_size)
    print(val_size)
    print(test_size)
    train_dataset, valtest_dataset = random_split(full_dataset,[train_size,valtest_size], generator=torch.Generator().manual_seed(args.seed))

    val_dataset, test_dataset = random_split(valtest_dataset,[val_size,test_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, drop_last=True,num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,num_workers=1)
    if args.task == 'td':
        scale = args.time_scale
    else:
        scale = args.freq_scale

    predictor = Predictor(2*2,\
                            len(label_dict),\
                            scale).cuda()
    predictor = nn.DataParallel(predictor)

    if args.transform == 'neural':
        denoiser = Denoiser(device,\
                    2,\
                    2).cuda()
        
    else:
        denoiser = nn.Identity().cuda()
    denoiser = nn.DataParallel(denoiser)

    if args.loss == 'cc':
        if args.task == 'td':
            estimator = fcc
        elif args.task == 'ds':
            estimator = cc
        else:
            estimator = cfo
    else:
        estimator = nn.DataParallel(nn.MSELoss()).cuda()

    mse_loss = nn.DataParallel(nn.MSELoss()).cuda()

    EPOCHS = args.epochs

    
    opt = optim.AdamW(list(predictor.parameters()) + list(denoiser.parameters()),lr=args.lr)

    #scheduler = CosineAnnealingLR(opt,EPOCHS,eta_min=args.lr)
    scheduler = ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=25)

    best_mse = 100000
    predictor.train()
    
    plt.figure(figsize=(15,10))
            
    for epoch in range(EPOCHS):
        predictor.train()
        denoiser.train()
        train_gt_lambdas = []
        train_pred_lambdas = []
        train_snrs = []
        val_gt_lambdas = []
        val_pred_lambdas = []
        val_snrs = []
        t = time.time()
        opt.zero_grad()
        for noisy_passband,noisy_passband_shifted,noisy_passband_delayed,label,true_tau,snr,fc,fc2 in train_loader:
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
            
            mask = F.dropout(torch.ones(size=signal1.shape,device=device),0.5)

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
            signals_total = torch.cat((torch.real(cleansignal1),\
                                    torch.imag(cleansignal1),\
                                    torch.real(cleansignal2),\
                                    torch.imag(cleansignal2)),1)
            
            est_lambda = predictor(signals_total.type(torch.float32))

            
            #print(est_class.shape)
            if args.loss == 'mse':
                loss = torch.mean(estimator(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
                
            elif args.transform == 'neural':
                #est_lambda = predictor(signals_total.type(torch.float32))
                tdloss1 = torch.mean(1 - estimator(cleansignal1,signal2,torch.reshape(true_lambda,[-1,1,1]),device,mask))
                tdloss2 = torch.mean(1 - estimator(signal1,cleansignal2,torch.reshape(true_lambda,[-1,1,1]),device,mask))
                loss = tdloss1 + tdloss2 + torch.mean(mse_loss(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
            else:
                loss = torch.mean(1 - estimator(cleansignal1,cleansignal2,torch.reshape(true_lambda,[-1,1,1]),device,mask)) #+ torch.mean(mse_loss(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
            loss.backward()
            opt.step()
            
            opt.zero_grad()

            loss =  loss.detach().cpu().numpy()
            train_pred_lambdas.append(est_lambda.detach().cpu().numpy())
            train_gt_lambdas.append(true_lambda.detach().cpu().numpy())
            train_snrs.append(snr.detach().cpu().numpy())

        train_pred_lambdas = np.concatenate(train_pred_lambdas)
        train_gt_lambdas = np.concatenate(train_gt_lambdas)
        train_mse = mean_squared_error(train_gt_lambdas,train_pred_lambdas)
        train_snrs = np.concatenate(train_snrs)
        

        io.cprint('Train Epoch {0}: Loss: {1} EstLambda: {2} TruLambda: {3} MSE: {4} LR: {5}'.format(epoch,\
                                                            loss,\
                                                            (est_lambda[0]).detach().cpu().numpy(),\
                                                            (true_lambda[0]).detach().cpu().numpy(),\
                                                            train_mse,\
                                                            scheduler.get_last_lr()))
        predictor.eval()
        denoiser.eval()
        with torch.no_grad():
            for noisy_passband,noisy_passband_shifted,noisy_passband_delayed,label,true_tau,snr,fc,fc2 in val_loader:
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
                
                #mask = F.dropout(torch.ones(size=signal1.shape,device=device),0.5)

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
                signals_total = torch.cat((torch.real(cleansignal1),\
                                        torch.imag(cleansignal1),\
                                        torch.real(cleansignal2),\
                                        torch.imag(cleansignal2)),1)
                
                est_lambda = predictor(signals_total.type(torch.float32))

                if args.loss == 'mse':
                    loss = torch.mean(estimator(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
                    
                elif args.transform == 'neural':
                    tdloss1 = torch.mean(1 - estimator(cleansignal1,signal2,torch.reshape(est_lambda,[-1,1,1]),device))
                    tdloss2 = torch.mean(1 - estimator(signal1,cleansignal2,torch.reshape(est_lambda,[-1,1,1]),device))
                    loss = tdloss1 + tdloss2
                else:
                    loss = torch.mean(1 - estimator(cleansignal1,cleansignal2,torch.reshape(est_lambda,[-1,1,1]),device))# + torch.mean(mse_loss(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
                    loss =  loss.detach().cpu().numpy()
                val_pred_lambdas.append(est_lambda.detach().cpu().numpy())
                val_gt_lambdas.append(true_lambda.detach().cpu().numpy())
                val_snrs.append(snr.detach().cpu().numpy())

        val_pred_lambdas = np.concatenate(val_pred_lambdas)
        val_gt_lambdas = np.concatenate(val_gt_lambdas)
        val_snrs = np.concatenate(val_snrs)
        val_mse = mean_squared_error(val_gt_lambdas,val_pred_lambdas)

        scheduler.step(val_mse)

        io.cprint('Val Epoch {0}: Loss: {1} EstTime: {2} TruTime: {3} MSE: {4} LR: {5}'.format(epoch,\
                                                            loss,\
                                                            (est_lambda[0]).detach().cpu().numpy(),\
                                                            (true_lambda[0]).detach().cpu().numpy(),\
                                                            val_mse,
                                                            scheduler.get_last_lr()))
        
        STEP = 2
        for i in range(-20,20,STEP):
            indices = np.where((val_snrs> i) & (val_snrs < i + STEP))
            cur_snr_mse = mean_squared_error(val_gt_lambdas[indices],val_pred_lambdas[indices])
            io.cprint('Val SNR < {0} MSE: {1}'.format(i + STEP, cur_snr_mse))                                                           
    
        #break
        io.cprint('Elapsed: {0}s'.format(time.time() - t))
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(predictor.module,'checkpoints/%s/model.pth' % args.exp_name)
            torch.save(denoiser.module,'checkpoints/%s/denoiser.pth' % args.exp_name)
            io.cprint('Model Saved!')
        io.cprint('Best MSE: {0}'.format(best_mse))

    predictor.eval()
    test_gt_lambdas = []
    test_pred_lambdas = []
    test_snrs = []
    io.cprint('Test Mode!')
    predictor = nn.DataParallel(torch.load('checkpoints/%s/model.pth' % args.exp_name).cuda())
    predictor.eval()
    denoiser = nn.DataParallel(torch.load('checkpoints/%s/denoiser.pth' % args.exp_name).cuda())
    denoiser.eval()
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
            signals_total = torch.cat((torch.real(cleansignal1),\
                                    torch.imag(cleansignal1),\
                                    torch.real(cleansignal2),\
                                    torch.imag(cleansignal2)),1)
            
            est_lambda = predictor(signals_total.type(torch.float32))

            if args.loss == 'mse':
                loss = torch.mean(estimator(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)
                
            elif args.transform == 'neural':
                tdloss1 = torch.mean(1 - estimator(cleansignal1,signal2,torch.reshape(est_lambda,[-1,1,1]),device))
                tdloss2 = torch.mean(1 - estimator(signal1,cleansignal2,torch.reshape(est_lambda,[-1,1,1]),device))
                loss = tdloss1 + tdloss2
            else:
                loss = torch.mean(1 - estimator(cleansignal1,cleansignal2,torch.reshape(est_lambda,[-1,1,1]),device)) #+ torch.mean(mse_loss(est_lambda/scale,torch.unsqueeze(true_lambda/scale,-1)),-1)

            loss =  loss.detach().cpu().numpy()
            test_pred_lambdas.append(est_lambda.detach().cpu().numpy())
            test_gt_lambdas.append(true_lambda.detach().cpu().numpy())
            test_snrs.append(snr.detach().cpu().numpy())

    test_pred_lambdas = np.concatenate(test_pred_lambdas)
    test_gt_lambdas = np.concatenate(test_gt_lambdas)
    test_snrs = np.concatenate(test_snrs)
    test_mse = mean_squared_error(test_gt_lambdas,test_pred_lambdas)

    io.cprint('Test Epoch {0}: Loss: {1} EstTime: {2} TruTime: {3} MSE: {4} LR: {5}'.format(epoch,\
                                                        loss,\
                                                        (est_lambda[0]).detach().cpu().numpy(),\
                                                        (true_lambda[0]).detach().cpu().numpy(),\
                                                        test_mse,\
                                                        scheduler.get_last_lr()))
    
    STEP = 2
    for i in range(-20,20,STEP):
        indices = np.where((test_snrs> i) & (test_snrs < i + STEP))
        cur_snr_mse = mean_squared_error(test_gt_lambdas[indices],test_pred_lambdas[indices])
        io.cprint('Test SNR < {0}: MSE: {1}'.format(i + STEP, cur_snr_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TFDOA Estimator')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--time_scale',type=float)
    parser.add_argument('--freq_scale',type=float)
    parser.add_argument('--transform',choices=['identity','neural'])
    parser.add_argument('--loss', choices=['cc','mse'])
    parser.add_argument('--task', choices=['td', 'ds', 'cf'])
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--seed',type=int,default=100)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    train(args)
