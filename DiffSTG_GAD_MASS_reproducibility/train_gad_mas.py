from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np
import torch
try:
    from easydict import EasyDict as edict
except Exception:
    class edict(dict):
        def __getattr__(self, name):
            try: return self[name]
            except KeyError as exc: raise AttributeError(name) from exc
        def __setattr__(self, name, value): self[name] = value

from utils.eval import Metric
from utils.common_utils import dir_check, Logger, to_device
from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG
from algorithm.diffstg.schedule_diagnostics import schedule_report

SPECTRAL_SCHEDULERS = ('mas_snr', 'fcps')


def str2bool(v):
    if isinstance(v, bool): return v
    s = str(v).lower()
    if s in {'yes','true','t','1','y'}: return True
    if s in {'no','false','f','0','n'}: return False
    raise argparse.ArgumentTypeError('Boolean expected')


def setup_seed(seed: int):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    # Paths / dataset
    p.add_argument('--data', default='PEMS08')
    p.add_argument('--data_root', default='data/dataset')
    p.add_argument('--output_root', default='output_clean')
    p.add_argument('--exp_name', default='mas_clean')
    p.add_argument('--resume_path', default='')
    p.add_argument('--device', default='auto')

    # Author architecture/diffusion dimensions
    p.add_argument('--epsilon_theta', default='UGnet')
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--N', type=int, default=100)
    p.add_argument('--beta_schedule', default='quad', choices=['uniform','quad'])
    p.add_argument('--beta_end', type=float, default=0.4)
    p.add_argument('--T_h', type=int, default=12)
    p.add_argument('--T_p', type=int, default=12)

    # Scheduler
    p.add_argument('--diffusion_scheduler', default='mas_snr', choices=['ddpm','mas_snr','fcps'])
    p.add_argument('--sample_steps', type=int, default=25, help='Validation sampler NFE target')
    p.add_argument('--ss', default='mas_ddim_multi')
    p.add_argument('--mas_gamma', type=float, default=1e-2)
    p.add_argument('--mas_sigma', type=float, default=1.0)
    p.add_argument('--mas_r_max', type=float, default=1e2)
    p.add_argument('--mas_r_min', type=float, default=1e-3)
    p.add_argument('--mas_s_min', type=float, default=None)
    p.add_argument('--mas_s_min_quantile', type=float, default=0.05)
    p.add_argument('--mas_power_mode', default='conditional_residual', choices=['second_moment','centered','conditional_residual'])
    p.add_argument('--mas_conditional_ridge', type=float, default=1e-3)
    p.add_argument('--mas_conditional_stride', type=int, default=1)

    # FCPS (scalar GAD schedule) baseline; used only with --diffusion_scheduler fcps.
    # cbar_t = c_min t + (cbar_T - c_min) t^(alpha+1) on t in [t_0, 1].
    p.add_argument('--fcps_alpha', type=float, default=2.0, help='Polynomial exponent (GAD paper: alpha > 1)')
    p.add_argument('--fcps_c_min', type=float, default=0.5, help='Rate floor per unit normalized time (GAD paper: 0 < c_min < 1)')
    p.add_argument('--fcps_endpoints', default='match_mass', choices=['match_mass','delta'],
                   help='match_mass: worst-mode a^2 at t_0 and T equal to MASS with the same r_max/r_min; delta: use --fcps_delta')
    p.add_argument('--fcps_delta', type=float, default=1e-3, help='Tolerance for --fcps_endpoints delta')
    p.add_argument('--fcps_cbar_T', type=float, default=None, help='Override terminal accumulated schedule')
    p.add_argument('--fcps_cbar_t0', type=float, default=None, help='Override accumulated schedule at the first level')

    # Theory/task alignment
    p.add_argument('--mas_prediction_type', default='x0', choices=['x0','eps'])
    p.add_argument('--mas_loss_scope', default='future', choices=['future','all'])
    p.add_argument('--mas_x0_loss_weighting', default='none', choices=['none','inv_sqrt_power','inv_power'])
    p.add_argument('--mas_time_loss_weighting', default='none', choices=['none','snr_fraction','min_snr'])
    p.add_argument('--mas_min_snr_gamma', type=float, default=5.0)
    p.add_argument('--mas_profile_condition', type=str2bool, default=True)
    p.add_argument('--gnn_graph_operator', default='mass_symmetric', choices=['author_directed','mass_symmetric'])
    p.add_argument('--mas_history_projection', type=str2bool, default=True)
    p.add_argument('--mas_sampler_grid', default='logsnr', choices=['logsnr','uniform'],
                   help="MASS: both are identical. FCPS: 'uniform' = same indices as MASS; 'logsnr' = uniform in mode-mean log-SNR")
    p.add_argument('--mas_eta', type=float, default=0.0)
    p.add_argument('--mas_clip_x0', type=float, default=0.0, help='0 disables clipping; x0 prediction normally does not need it')

    # Optimization / validation
    p.add_argument('--is_train', type=str2bool, default=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--min_epochs', type=int, default=60)
    p.add_argument('--val_every', type=int, default=5)
    p.add_argument('--early_stop_checks', type=int, default=8, help='8 checks x val_every=5 -> ~40 epochs patience')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--mask_ratio', type=float, default=0.0)
    p.add_argument('--select_metric', default='crps', choices=['crps','mae','rmse','mape'])
    p.add_argument('--val_n_samples', type=int, default=8)
    p.add_argument('--val_max_batches', type=int, default=0, help='0 = full validation set')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)

    # Locked test: OFF in all sweeps.
    p.add_argument('--run_locked_test', type=str2bool, default=False)
    p.add_argument('--test_n_samples', type=int, default=8)
    p.add_argument('--test_sample_steps', default='100,50,25')
    return p.parse_args()


def default_config(args):
    config = edict()
    root = Path(args.output_root).expanduser().resolve()
    config.PATH_MOD = str(root/'model') + '/'; config.PATH_LOG = str(root/'log') + '/'
    config.PATH_FORECAST = str(root/'forecast') + '/'; config.PATH_JSON = str(root/'json') + '/'
    for d in [config.PATH_MOD,config.PATH_LOG,config.PATH_FORECAST,config.PATH_JSON]: os.makedirs(d,exist_ok=True)

    config.data = edict(); config.data.name=args.data; config.data.path=str(Path(args.data_root).expanduser().resolve())+'/'
    config.data.feature_file=config.data.path+args.data+'/flow.npy'; config.data.spatial=config.data.path+args.data+'/adj.npy'; config.data.num_recent=1
    if args.data != 'PEMS08': raise NotImplementedError('clean package is PEMS08-specific')
    config.data.num_features=1; config.data.num_vertices=170; config.data.points_per_hour=12
    config.data.val_start_idx=int(17856*0.6); config.data.test_start_idx=int(17856*0.8)

    config.device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    m=config.model=edict(); m.T_p=args.T_p; m.T_h=args.T_h; m.V=170; m.F=1; m.week_len=7; m.day_len=288; m.device=config.device
    m.d_h=args.hidden_size; m.C=args.hidden_size; m.n_channels=args.hidden_size; m.N=args.N; m.sample_steps=args.sample_steps
    m.epsilon_theta=args.epsilon_theta; m.is_label_condition=True; m.beta_start=0.0001; m.beta_end=args.beta_end; m.beta_schedule=args.beta_schedule
    m.sample_strategy=args.ss; m.channel_multipliers=[1,2]; m.supports_len=2; m.diffusion_scheduler=args.diffusion_scheduler
    m.mas_gamma=args.mas_gamma; m.mas_sigma=args.mas_sigma; m.mas_r_max=args.mas_r_max; m.mas_r_min=args.mas_r_min
    m.mas_s_min=args.mas_s_min; m.mas_s_min_quantile=args.mas_s_min_quantile; m.mas_power_mode=args.mas_power_mode
    m.mas_conditional_ridge=args.mas_conditional_ridge; m.mas_conditional_stride=args.mas_conditional_stride
    m.mas_prediction_type=args.mas_prediction_type; m.mas_loss_scope=args.mas_loss_scope; m.mas_x0_loss_weighting=args.mas_x0_loss_weighting
    m.mas_time_loss_weighting=args.mas_time_loss_weighting; m.mas_min_snr_gamma=args.mas_min_snr_gamma
    m.mas_profile_condition=args.mas_profile_condition; m.gnn_graph_operator=args.gnn_graph_operator
    m.mas_history_projection=args.mas_history_projection; m.mas_sampler_grid=args.mas_sampler_grid; m.mas_eta=args.mas_eta; m.mas_clip_x0=args.mas_clip_x0
    if args.diffusion_scheduler == 'fcps':
        m.fcps_alpha=args.fcps_alpha; m.fcps_c_min=args.fcps_c_min; m.fcps_endpoints=args.fcps_endpoints
        m.fcps_delta=args.fcps_delta; m.fcps_cbar_T=args.fcps_cbar_T; m.fcps_cbar_t0=args.fcps_cbar_t0

    config.model_name='DiffSTG'; config.epoch=args.epochs; config.lr=args.lr; config.batch_size=args.batch_size; config.wd=args.weight_decay
    config.grad_clip=args.grad_clip; config.mask_ratio=args.mask_ratio; config.select_metric=args.select_metric; config.seed=args.seed; config.exp_name=args.exp_name
    config.val_n_samples=args.val_n_samples; config.test_n_samples=args.test_n_samples; config.logger=Logger()
    return config


def make_jsonable(obj):
    if isinstance(obj,(str,int,float,bool)) or obj is None: return obj
    if isinstance(obj,torch.device): return str(obj)
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,(dict,edict)):
        return {k:make_jsonable(v) for k,v in obj.items() if k not in {'logger','mas_train_data','A'}}
    if isinstance(obj,(list,tuple)): return [make_jsonable(x) for x in obj]
    return str(obj)


def save_json(path,payload):
    dir_check(path)
    with open(path,'w',encoding='utf-8') as f: json.dump(make_jsonable(payload),f,indent=2,sort_keys=True)


def metric_dict(metric):
    return {k:float(v) for k,v in metric.metrics.items() if isinstance(v,(int,float,np.number))}


def eval_model(model, loader, metric, config, clean_data, mode='Val', n_samples=8, max_batches=0):
    # Fixed evaluation seed makes validation comparisons low-noise and reproducible.
    setup_seed(2022)
    model.eval(); y_true=[]; y_pred=[]; elapsed=0.0
    with torch.no_grad():
        for i,batch in enumerate(loader):
            if max_batches and i>=max_batches: break
            start=timer(); future,history_batch,pos_w,pos_d=to_device(batch,config.device)
            x=torch.cat((history_batch,future),dim=1).to(config.device)
            x_masked=torch.cat((history_batch,torch.zeros_like(future)),dim=1).to(config.device)
            x=x.transpose(1,3); x_masked=x_masked.transpose(1,3)
            x_hat=model((x_masked,pos_w,pos_d),n_samples)
            elapsed += timer()-start
            x=clean_data.reverse_normalization(x); x_hat=clean_data.reverse_normalization(x_hat).detach()
            truth=x[:,:,:,-config.model.T_p:].transpose(1,3).cpu().numpy()
            pred=x_hat[:,:,:,:,-config.model.T_p:].transpose(2,4).cpu().numpy()
            y_true.append(truth); y_pred.append(np.clip(pred,0,np.inf))
    metric.update_metrics(np.concatenate(y_true),np.concatenate(y_pred)); metric.metrics['time']=elapsed
    metric.metrics['nfe']=int(model.last_nfe or model.sample_steps)
    return metric


def load_state(model,path,device):
    ckpt=torch.load(path,map_location=device)
    if isinstance(ckpt,dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict']); return int(ckpt.get('epoch',-1)),ckpt
    model=ckpt.to(device); return -1,{'legacy_whole_model':True}


def main(args):
    if args.diffusion_scheduler in SPECTRAL_SCHEDULERS and not args.ss.startswith('mas_'):
        raise ValueError(f"--diffusion_scheduler {args.diffusion_scheduler} needs a spectral sampler (--ss mas_ddim_multi), got {args.ss}")
    setup_seed(args.seed); torch.set_num_threads(2); config=default_config(args)
    config.trial_name=args.exp_name; config.log_path=config.PATH_LOG+args.exp_name+'.log'; config.logger.open(config.log_path,mode='w')

    clean=CleanDataset(config); config.model.A=clean.adj
    # Only first 60% enters MASS power estimation; val/test never enter p_k.
    config.model.mas_train_data=np.asarray(clean.feature[:config.data.val_start_idx],dtype=np.float32)
    model=DiffSTG(config.model).to(config.device)

    diag_tag='fcps' if args.diffusion_scheduler=='fcps' else 'mas'
    diag_path=config.PATH_JSON+args.exp_name+f'_{diag_tag}_diagnostics.json'
    if model.mas is not None:
        save_json(diag_path,model.mas.diagnostics()); print(f'{diag_tag.upper()} diagnostics:',json.dumps(model.mas.diagnostics(),indent=2))
        # Same forward-channel report for MASS and FCPS (SNR spread, informative levels, schedule-only stiffness).
        report_path=config.PATH_JSON+args.exp_name+'_schedule_report.json'
        save_json(report_path,schedule_report(model.mas)); print('schedule report:',report_path)

    train_ds=TrafficDataset(clean,(config.model.T_p,config.data.val_start_idx-config.model.T_p+1),config)
    val_ds=TrafficDataset(clean,(config.data.val_start_idx+config.model.T_p,config.data.test_start_idx-config.model.T_p+1),config)
    test_ds=TrafficDataset(clean,(config.data.test_start_idx+config.model.T_p,-1),config)
    train_loader=torch.utils.data.DataLoader(train_ds,config.batch_size,shuffle=True,pin_memory=True,num_workers=args.num_workers)
    val_loader=torch.utils.data.DataLoader(val_ds,64,shuffle=False,num_workers=args.num_workers)
    test_loader=torch.utils.data.DataLoader(test_ds,64,shuffle=False,num_workers=args.num_workers)

    model_path=config.PATH_MOD+args.exp_name+'+'+model.model_file_name(); config.model_path=model_path
    cfg_path=config.PATH_JSON+args.exp_name+'_config.json'; hist_path=config.PATH_JSON+args.exp_name+'_validation_history.json'
    summary_path=config.PATH_JSON+args.exp_name+'_summary.json'; save_json(cfg_path,{'config':config,'args':vars(args)})

    best=float('inf'); best_epoch=-1; bad_checks=0; validation_history=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)

    if args.resume_path:
        best_epoch,_=load_state(model,args.resume_path,config.device)

    if args.is_train:
        for epoch in range(args.epochs):
            model.train(); losses=[]
            for batch in train_loader:
                future,history_batch,pos_w,pos_d=to_device(batch,config.device)
                if args.mask_ratio>0:
                    mask=torch.rand_like(history_batch)<args.mask_ratio
                    history_batch=history_batch.clone()
                    history_batch[mask]=0
                x=torch.cat((history_batch,future),dim=1).transpose(1,3)
                xm=torch.cat((history_batch,torch.zeros_like(future)),dim=1).transpose(1,3)
                loss=model.loss(x,(xm,pos_w,pos_d))
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                optimizer.step(); losses.append(float(loss.detach()))
            train_loss=float(np.mean(losses)); print(f'epoch={epoch} train_loss={train_loss:.6f}',flush=True)

            do_val=((epoch+1)%args.val_every==0) or epoch==0 or epoch==args.epochs-1
            if not do_val: continue
            met=Metric(T_p=args.T_p); eval_model(model,val_loader,met,config,clean,'Val',args.val_n_samples,args.val_max_batches)
            cur=float(met.metrics[args.select_metric]); scheduler.step(cur)
            row={'epoch':epoch,'train_loss':train_loss,'lr':float(optimizer.param_groups[0]['lr']),**metric_dict(met)}
            validation_history.append(row); print('VAL',row,flush=True)
            if cur < best:
                best=cur; best_epoch=epoch; bad_checks=0
                torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                            'best_value':best,'select_metric':args.select_metric,'config':make_jsonable(config),'args':vars(args)},model_path)
            else:
                bad_checks += 1
            save_json(hist_path,{'select_metric':args.select_metric,'best_value':best,'best_epoch':best_epoch,'bad_checks':bad_checks,'history':validation_history})
            if (epoch+1)>=args.min_epochs and bad_checks>=args.early_stop_checks:
                print(f'early stop epoch={epoch}; best={best_epoch}',flush=True); break
    elif not args.resume_path:
        raise ValueError('--is_train false requires --resume_path')

    if os.path.exists(model_path): best_epoch,_=load_state(model,model_path,config.device)
    elif args.resume_path: model_path=args.resume_path
    else: raise RuntimeError('No checkpoint available')

    payload={'exp_name':args.exp_name,'diffusion_scheduler':args.diffusion_scheduler,'best_epoch':best_epoch,'best_validation_value':best,'select_metric':args.select_metric,
             'checkpoint':model_path,'config_path':cfg_path,'validation_path':hist_path,'mas_diagnostics_path':diag_path if model.mas else None,
             'locked_test_ran':bool(args.run_locked_test),'test_results':[]}

    # Locked test is opt-in only.  Sweeps should leave this false.
    if args.run_locked_test:
        for steps in [int(s) for s in args.test_sample_steps.split(',') if s.strip()]:
            if steps>config.model.N: continue
            model.set_ddim_sample_steps(steps)
            model.set_sample_strategy('mas_ddim_multi' if args.diffusion_scheduler in SPECTRAL_SCHEDULERS else 'ddim_multi')
            met=Metric(T_p=args.T_p); eval_model(model,test_loader,met,config,clean,'Test',args.test_n_samples,0)
            result={'requested_steps':steps,'actual_nfe':int(model.last_nfe or steps),'sample_sequence':model.last_sample_sequence,**metric_dict(met)}
            payload['test_results'].append(result); print('LOCKED TEST',result,flush=True)

    save_json(summary_path,payload); print('summary:',summary_path); return payload


if __name__=='__main__': main(parse_args())
