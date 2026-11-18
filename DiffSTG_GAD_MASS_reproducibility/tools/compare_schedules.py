import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm.diffstg.mas_snr import MASSNR
from algorithm.diffstg.fcps import FCPS
from algorithm.diffstg.schedule_diagnostics import schedule_report

p = argparse.ArgumentParser()
p.add_argument('--data_root', default='data/dataset')
p.add_argument('--N', type=int, default=100)
p.add_argument('--mas_gamma', type=float, default=0.02)
p.add_argument('--mas_sigma', type=float, default=1.0)
p.add_argument('--mas_r_max', type=float, default=100.0)
p.add_argument('--mas_r_min', type=float, default=1e-4)
p.add_argument('--mas_s_min_quantile', type=float, default=0.05)
p.add_argument('--mas_power_mode', default='conditional_residual')
p.add_argument('--mas_conditional_ridge', type=float, default=1e-3)
p.add_argument('--mas_conditional_stride', type=int, default=1)
p.add_argument('--T_h', type=int, default=12)
p.add_argument('--T_p', type=int, default=12)
p.add_argument('--fcps_alpha', default='2.0', help='comma-separated list')
p.add_argument('--fcps_c_min', default='0.5', help='comma-separated list')
p.add_argument('--fcps_endpoints', default='match_mass', choices=['match_mass', 'delta'])
p.add_argument('--fcps_delta', type=float, default=1e-3)
p.add_argument('--eps', type=float, default=0.1, help='informative tolerance: SNR in [eps, 1/eps]')
p.add_argument('--out', default='')
a = p.parse_args()

# Same data preparation as CleanDataset: PEMS08 flow, normalized with training-split statistics.
data = Path(a.data_root) / 'PEMS08'
flow = np.expand_dims(np.load(data / 'flow.npy')[:, :, 0], -1)
val_start = int(17856 * 0.6)
train = flow[:val_start]
x = ((flow - np.mean(train)) / np.std(train)).astype('float32')[:val_start]
adj = np.load(data / 'adj.npy')

common = dict(adj=adj, train_data=x, num_steps=a.N, device=torch.device('cpu'), gamma=a.mas_gamma,
              sigma=a.mas_sigma, r_max=a.mas_r_max, r_min=a.mas_r_min, s_min_quantile=a.mas_s_min_quantile,
              power_mode=a.mas_power_mode, conditional_history=a.T_h, conditional_horizon=a.T_p,
              conditional_ridge=a.mas_conditional_ridge, conditional_stride=a.mas_conditional_stride)
results = []
mass = MASSNR(**common)
results.append(('MASS', mass, mass.diagnostics(), schedule_report(mass, eps=a.eps)))
for alpha in [float(v) for v in a.fcps_alpha.split(',') if v.strip()]:
    for c_min in [float(v) for v in a.fcps_c_min.split(',') if v.strip()]:
        f = FCPS(fcps_alpha=alpha, fcps_c_min=c_min, fcps_endpoints=a.fcps_endpoints, fcps_delta=a.fcps_delta, **common)
        results.append((f'FCPS a={alpha:g} cmin={c_min:g}', f, f.diagnostics(), schedule_report(f, eps=a.eps)))

print(f"cond(L_gamma) = {results[0][3]['cond_L_gamma']:.2f}   informative: SNR in [{a.eps:g}, {1/a.eps:g}]")
hdr = f"{'schedule':<24}{'min a2(t0)':>11}{'max a2(T)':>11}{'SNR spread':>12}{'inf hi-f':>10}{'inf med':>9}{'L_S0':>10}{'L_S0 @T':>10}"
print(hdr); print('-' * len(hdr))
def endpoint_a2(sched):
    with torch.no_grad():
        a0, _, _ = sched.coeffs(torch.zeros(1, dtype=torch.long))
        aT, _, _ = sched.coeffs(torch.full((1,), sched.num_steps - 1, dtype=torch.long))
    return float((a0 ** 2).min()), float((aT ** 2).max())


for name, sched, d, r in results:
    a2_0, a2_T = endpoint_a2(sched)
    inf = r['informative_fraction_of_levels']; st = r['schedule_only_stiffness']
    print(f"{name:<24}{a2_0:>11.4f}{a2_T:>11.4f}{r['log10_snr_spread_across_modes']['max_over_levels']:>12.2f}"
          f"{inf['highest_graph_frequency']:>10.2f}{inf['median_over_modes']:>9.2f}{st['L_S0']:>10.1f}{st['max_over_modes_at_terminal']:>10.1f}")
print("\nSNR spread: max over levels of (max_k - min_k) log10 SNR.  inf hi-f / inf med: fraction of the"
      "\nN levels at which the highest-frequency / median mode is informative.  L_S0: schedule-only stiffness"
      "\n(normalized horizon); a scalar schedule's peak is set by its rate floor c_min near t_0.")
if a.out:
    Path(a.out).write_text(json.dumps([{'schedule': n, 'diagnostics': d, 'report': r} for n, _, d, r in results],
                                      indent=2, default=str))
    print(f'Wrote {a.out}')
