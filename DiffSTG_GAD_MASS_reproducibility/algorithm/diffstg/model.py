try:
    import easydict
except Exception:
    class _EasyDict(dict):
        def __getattr__(self, name):
            try: return self[name]
            except KeyError as exc: raise AttributeError(name) from exc
        def __setattr__(self, name, value): self[name] = value
    class easydict:
        EasyDict = _EasyDict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ugnet import UGnet
from .mas_snr import MASSNR
from .fcps import FCPS
from utils.common_utils import gather


class DiffSTG(nn.Module):
    def __init__(self, config: easydict):
        super().__init__()
        self.config = config
        self.N = config.N
        self.sample_steps = config.sample_steps
        self.sample_strategy = config.sample_strategy
        self.device = config.device
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.beta_schedule = config.beta_schedule
        self.diffusion_scheduler = config.get('diffusion_scheduler', 'ddpm')
        self.mas_clip_x0 = float(config.get('mas_clip_x0', 0.0))
        self.mas_prediction_type = config.get('mas_prediction_type', 'x0')
        self.mas_loss_scope = config.get('mas_loss_scope', 'future')
        self.mas_x0_loss_weighting = config.get('mas_x0_loss_weighting', 'none')
        self.mas_time_loss_weighting = config.get('mas_time_loss_weighting', 'none')
        self.mas_min_snr_gamma = float(config.get('mas_min_snr_gamma', 5.0))
        self.mas_sampler_grid = config.get('mas_sampler_grid', 'logsnr')
        self.mas_eta = float(config.get('mas_eta', 0.0))
        self.mas_history_projection = bool(config.get('mas_history_projection', True))
        self.last_nfe = None
        self.last_sample_sequence = None

        if config.epsilon_theta != 'UGnet':
            raise NotImplementedError(config.epsilon_theta)
        self.eps_model = UGnet(config).to(self.device)

        # Author DDPM coefficients: kept exactly on the baseline path.
        if self.beta_schedule == 'uniform':
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N).to(self.device)
        elif self.beta_schedule == 'quad':
            self.beta = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.N) ** 2
            self.beta = self.beta.to(self.device)
        else:
            raise NotImplementedError(self.beta_schedule)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

        # Spectral (GAD) scheduler handle; named ``mas`` for backward
        # compatibility with published MASS checkpoints.  Holds MASS or FCPS.
        self.mas = None
        spectral_kwargs = dict(
            adj=config.A,
            train_data=config.mas_train_data,
            num_steps=self.N,
            device=self.device,
            gamma=config.get('mas_gamma', 1e-2),
            sigma=config.get('mas_sigma', 1.0),
            r_max=config.get('mas_r_max', 1e2),
            r_min=config.get('mas_r_min', 1e-3),
            s_min=config.get('mas_s_min', None),
            s_min_quantile=config.get('mas_s_min_quantile', 0.05),
            power_mode=config.get('mas_power_mode', 'centered'),
            conditional_history=config.get('T_h', 12),
            conditional_horizon=config.get('T_p', 12),
            conditional_ridge=config.get('mas_conditional_ridge', 1e-3),
            conditional_stride=config.get('mas_conditional_stride', 1),
        ) if self.diffusion_scheduler in ('mas_snr', 'fcps') else None
        if self.diffusion_scheduler == 'mas_snr':
            self.mas = MASSNR(**spectral_kwargs)
        elif self.diffusion_scheduler == 'fcps':
            if self.mas_time_loss_weighting != 'none':
                raise ValueError("FCPS has no common SNR trajectory; use mas_time_loss_weighting='none'")
            self.mas = FCPS(
                fcps_alpha=config.get('fcps_alpha', 2.0),
                fcps_c_min=config.get('fcps_c_min', 0.5),
                fcps_endpoints=config.get('fcps_endpoints', 'match_mass'),
                fcps_delta=config.get('fcps_delta', 1e-3),
                fcps_cbar_T=config.get('fcps_cbar_T', None),
                fcps_cbar_t0=config.get('fcps_cbar_t0', None),
                **spectral_kwargs,
            )
        elif self.diffusion_scheduler != 'ddpm':
            raise NotImplementedError(self.diffusion_scheduler)

    def _profile(self, t):
        if self.mas is None or not bool(self.config.get('mas_profile_condition', False)):
            return None
        return self.mas.profile_features(t)

    def q_xt_x0(self, x0, t, eps=None):
        if eps is None: eps = torch.randn_like(x0)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean + eps * var ** 0.5

    def p_sample(self, xt, t, c):
        eps_theta = self.eps_model(xt, t, c)
        alpha_coef = 1. / gather(self.alpha, t) ** 0.5
        eps_coef = gather(self.beta, t) / (1 - gather(self.alpha_bar, t)) ** 0.5
        mean = alpha_coef * (xt - eps_coef * eps_theta)
        var = (1 - gather(self.alpha_bar, t - 1)) / (1 - gather(self.alpha_bar, t)) * gather(self.beta, t)
        return mean + torch.randn_like(xt) * var ** 0.5

    def p_sample_loop(self, c):
        x_masked, _, _ = c
        B, _, V, T = x_masked.shape
        with torch.no_grad():
            x = torch.randn([B, self.config.F, V, T], device=self.device)
            for t in range(self.N, 0, -1):
                t -= 1
                if t > 0: x = self.p_sample(x, x.new_full((B,), t, dtype=torch.long), c)
        return x

    def _ddim_sequence(self):
        """Original author sequence; used only by DDPM/DDIM baseline."""
        N, timesteps, skip_type = self.N, self.sample_steps, self.beta_schedule
        if timesteps <= 0 or timesteps > N: raise ValueError("bad sample_steps")
        if skip_type == 'uniform':
            skip = max(N // timesteps, 1); seq = list(range(0, N, skip))[:timesteps]
        elif skip_type == 'quad':
            seq = [int(s) for s in (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)]
        else: raise NotImplementedError(skip_type)
        out = []
        for s in seq:
            s = int(max(0, min(N - 1, s)))
            if not out or out[-1] != s: out.append(s)
        if out[-1] != N - 1: out.append(N - 1)
        return out

    def p_sample_loop_ddim(self, c):
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape
        seq = self._ddim_sequence()
        self.last_sample_sequence, self.last_nfe = seq, len(seq)
        x = torch.randn([B, self.config.F, V, T], device=self.device)
        return generalized_steps(x, seq, self.eps_model, self.beta, c, eta=1)

    def _predict_x0_and_eps(self, xt, t, c):
        xt_spec = self.mas.to_spec(xt)
        a_t, nu_t, _ = self.mas.coeffs(t)
        profile = self._profile(t)
        pred = self.eps_model(xt, t, c, noise_profile=profile)
        if self.mas_prediction_type == 'x0':
            x0_node = pred
            if self.mas_clip_x0 > 0:
                x0_node = torch.clamp(x0_node, -self.mas_clip_x0, self.mas_clip_x0)
            x0_spec = self.mas.to_spec(x0_node)
            eps_spec = (xt_spec - a_t[:, None, :, None] * x0_spec) / (
                torch.sqrt(nu_t)[:, None, :, None] + 1e-8
            )
        elif self.mas_prediction_type == 'eps':
            eps_spec = self.mas.to_spec(pred)
            x0_spec = (xt_spec - torch.sqrt(nu_t)[:, None, :, None] * eps_spec) / (
                a_t[:, None, :, None] + 1e-8
            )
            x0_node = self.mas.from_spec(x0_spec)
            if self.mas_clip_x0 > 0:
                x0_node = torch.clamp(x0_node, -self.mas_clip_x0, self.mas_clip_x0)
                x0_spec = self.mas.to_spec(x0_node)
        else:
            raise ValueError(f"Unknown mas_prediction_type={self.mas_prediction_type}")
        return x0_node, x0_spec, eps_spec

    def p_sample_loop_mas_ddim(self, c):
        if self.mas is None: raise RuntimeError("MASS not initialized")
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape
        seq = self.mas.sample_sequence(self.sample_steps, self.mas_sampler_grid)
        self.last_sample_sequence, self.last_nfe = seq, len(seq)
        seq_next = [-1] + list(seq[:-1])
        x = self.mas.terminal_sample((B, F, V, T), self.device)

        hist_len = int(self.config.get('T_h', 12))
        hist_clean = x_masked[..., :hist_len]
        hist_eps = torch.randn_like(self.mas.to_spec(hist_clean)) if self.mas_history_projection else None
        if self.mas_history_projection:
            terminal_t = torch.full((B,), self.N - 1, dtype=torch.long, device=self.device)
            aT, nuT, _ = self.mas.coeffs(terminal_t)
            histT = aT[:, None, :, None] * self.mas.to_spec(hist_clean) + torch.sqrt(nuT)[:, None, :, None] * hist_eps
            x[..., :hist_len] = self.mas.from_spec(histT)

        xs, x0_preds = [x], []
        eta = min(max(self.mas_eta, 0.0), 1.0)
        with torch.no_grad():
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((B,), int(i), dtype=torch.long, device=self.device)
                next_t = torch.full((B,), int(j), dtype=torch.long, device=self.device)
                xt = xs[-1].to(self.device)
                x0_node, x0_spec, eps_spec = self._predict_x0_and_eps(xt, t, c)
                x0_preds.append(x0_node.detach().cpu())
                a_next, nu_next = self.mas.coeffs_for_next_t(next_t)
                if eta > 0 and j >= 0:
                    z = torch.randn_like(eps_spec)
                    eps_next = math.sqrt(1.0 - eta ** 2) * eps_spec + eta * z
                else:
                    eps_next = eps_spec
                x_next_spec = a_next[:, None, :, None] * x0_spec + torch.sqrt(nu_next)[:, None, :, None] * eps_next
                x_next = self.mas.from_spec(x_next_spec)

                if self.mas_history_projection:
                    if j < 0:
                        x_next[..., :hist_len] = hist_clean
                    else:
                        hist_next_spec = (
                            a_next[:, None, :, None] * self.mas.to_spec(hist_clean)
                            + torch.sqrt(nu_next)[:, None, :, None] * hist_eps
                        )
                        x_next[..., :hist_len] = self.mas.from_spec(hist_next_spec)
                if self.mas_clip_x0 > 0:
                    x_next = torch.clamp(x_next, -self.mas_clip_x0, self.mas_clip_x0)
                xs.append(x_next.detach().cpu())
        return xs, x0_preds

    def set_sample_strategy(self, sample_strategy): self.sample_strategy = sample_strategy
    def set_ddim_sample_steps(self, sample_steps): self.sample_steps = sample_steps

    def evaluate(self, input, n_samples=2):
        x_masked, pos_w, pos_d = input
        B, F, V, T = x_masked.shape
        if self.sample_strategy in ['ddim_multi', 'mas_ddim_multi']:
            xmr = x_masked.unsqueeze(1).repeat(1,n_samples,1,1,1).reshape(-1,F,V,T)
            pwr = pos_w.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,pos_w.shape[-1]) if pos_w.dim()==2 else pos_w
            pdr = pos_d.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,pos_d.shape[-1]) if pos_d.dim()==2 else pos_d
            if self.sample_strategy == 'mas_ddim_multi': xs, _ = self.p_sample_loop_mas_ddim((xmr,pwr,pdr))
            else: xs, _ = self.p_sample_loop_ddim((xmr,pwr,pdr))
            return xs[-1].to(self.device).reshape(B,n_samples,F,V,T)
        if self.sample_strategy in ['ddim_one','mas_ddim_one']:
            xs, _ = self.p_sample_loop_mas_ddim((x_masked,pos_w,pos_d)) if self.sample_strategy=='mas_ddim_one' else self.p_sample_loop_ddim((x_masked,pos_w,pos_d))
            return torch.stack([xx.to(self.device) for xx in xs[-n_samples:]], dim=1)
        if self.sample_strategy == 'ddpm':
            xmr = x_masked.unsqueeze(1).repeat(1,n_samples,1,1,1).reshape(-1,F,V,T)
            return self.p_sample_loop((xmr,pos_w,pos_d)).reshape(B,n_samples,F,V,T)
        raise NotImplementedError(self.sample_strategy)

    def forward(self, input, n_samples=1): return self.evaluate(input, n_samples)

    def _future_slice(self, x):
        return x[..., -int(self.config.get('T_p', 12)):]

    def _x0_loss(self, pred_node, target_node, t):
        if self.mas_loss_scope == 'future':
            pred_node, target_node = self._future_slice(pred_node), self._future_slice(target_node)
        if self.mas_x0_loss_weighting == 'none':
            per = (pred_node - target_node) ** 2
        else:
            pred_spec, target_spec = self.mas.to_spec(pred_node), self.mas.to_spec(target_node)
            err2 = (pred_spec - target_spec) ** 2
            p = self.mas.sk[None,None,:,None]
            if self.mas_x0_loss_weighting == 'inv_sqrt_power':
                w = torch.rsqrt(p + 1e-8)
            elif self.mas_x0_loss_weighting == 'inv_power':
                w = 1.0 / (p + 1e-8)
            else: raise ValueError(self.mas_x0_loss_weighting)
            w = w / w.mean()
            per = err2 * w
        per_sample = per.reshape(per.shape[0], -1).mean(dim=1)
        if self.mas_time_loss_weighting == 'none':
            w_t = torch.ones_like(per_sample)
        else:
            r = self.mas.r_of_t(t)
            if self.mas_time_loss_weighting == 'snr_fraction':
                w_t = r / (1.0 + r)
            elif self.mas_time_loss_weighting == 'min_snr':
                w_t = torch.minimum(r, torch.full_like(r, self.mas_min_snr_gamma)) / (r + 1e-8)
            else: raise ValueError(self.mas_time_loss_weighting)
            w_t = w_t / w_t.mean().clamp_min(1e-8)
        return (per_sample * w_t).mean()

    def loss(self, x0: torch.Tensor, c: Tuple):
        t = torch.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=torch.long)
        if self.diffusion_scheduler in ('mas_snr', 'fcps'):
            xt, eps_spec = self.mas.q_sample(x0, t)
            profile = self._profile(t)
            pred = self.eps_model(xt, t, c, noise_profile=profile)
            if self.mas_prediction_type == 'x0':
                return self._x0_loss(pred, x0, t)
            if self.mas_prediction_type == 'eps':
                eps_pred_spec = self.mas.to_spec(pred)
                if self.mas_loss_scope == 'future':
                    eps_pred_spec = self._future_slice(eps_pred_spec)
                    eps_spec = self._future_slice(eps_spec)
                return F.mse_loss(eps_pred_spec, eps_spec)
            raise ValueError(self.mas_prediction_type)

        eps = torch.randn_like(x0)
        xt = self.q_xt_x0(x0, t, eps)
        eps_theta = self.eps_model(xt, t, c)
        return F.mse_loss(eps, eps_theta)

    def model_file_name(self):
        keys = ['N','T_h','T_p','epsilon_theta']
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in keys])
        if self.config.get('diffusion_scheduler','ddpm') == 'mas_snr':
            file_name += (
                f"+MAS-g{self.config.get('mas_gamma')}"
                f"-rmax{self.config.get('mas_r_max')}-rmin{self.config.get('mas_r_min')}"
                f"-p{self.config.get('mas_power_mode')}-pred{self.config.get('mas_prediction_type')}"
            )
        elif self.config.get('diffusion_scheduler','ddpm') == 'fcps':
            file_name += (
                f"+FCPS-g{self.config.get('mas_gamma')}"
                f"-a{self.config.get('fcps_alpha')}-cmin{self.config.get('fcps_c_min')}"
                f"-{self.config.get('fcps_endpoints')}"
                f"-p{self.config.get('mas_power_mode')}-pred{self.config.get('mas_prediction_type')}"
            )
        return file_name + '.pt'


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1-beta).cumprod(dim=0).index_select(0,t+1).view(-1,1,1,1)


def generalized_steps(x, seq, model, b, c, **kwargs):
    with torch.no_grad():
        n=x.size(0); seq_next=[-1]+list(seq[:-1]); x0_preds=[]; xs=[x]
        for i,j in zip(reversed(seq),reversed(seq_next)):
            t=(torch.ones(n)*i).to(x.device).long(); nt=(torch.ones(n)*j).to(x.device).long()
            at=compute_alpha(b,t); an=compute_alpha(b,nt); xt=xs[-1].to(x.device); et=model(xt,t,c)
            x0_t=(xt-et*(1-at).sqrt())/at.sqrt(); x0_preds.append(x0_t.to('cpu'))
            c1=kwargs.get('eta',0)*((1-at/an)*(1-an)/(1-at)).sqrt(); c2=((1-an)-c1**2).sqrt()
            xs.append((an.sqrt()*x0_t+c1*torch.randn_like(x)+c2*et).to('cpu'))
    return xs,x0_preds

# Compatibility hook retained for original author utilities.
try:
    from utils.common_utils import save2file_meta, ws
    def save2file(params):
        file_name = ws + '/output/metrics/DiffSTG.csv'
        head = ['data.name','model','model.N','model.epsilon_theta','model.d_h','model.T_h','model.T_p',
                'model.sample_strategy','model.sample_steps','model.beta_end','model.diffusion_scheduler','n_samples',
                'epoch','best_epoch','batch_size','lr','wd','early_stop','is_test','log_time','mae','rmse','mape','crps','mis','time','model_path','log_path','forecast_path']
        save2file_meta(params,file_name,head)
except Exception:
    pass
