import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--results-root', required=True)
p.add_argument('--nfes', default='3,8')
p.add_argument('--out', default='published_aggregate.csv')
p.add_argument('--scheduler', default='mas_snr', choices=['mas_snr','fcps','both'])
p.add_argument('--seeds', default='42,43,44,45,46')
p.add_argument('--fcps-eval-group', default='fcps_locked_low_nfe',
               help='FCPS evaluation folder under results root, e.g. fcps_locked_low_nfe_logsnr')
a=p.parse_args()
seeds=[int(x) for x in a.seeds.split(',') if x.strip()]
nfes=[int(x) for x in a.nfes.split(',') if x.strip()]
METRICS=('mae','rmse','crps')


def summary_path(scheduler, seed):
    root=Path(a.results_root)
    if scheduler=='mas_snr':
        return root/'final_locked_low_nfe'/f'seed_{seed}'/'json'/f'published_low_nfe_s{seed}_summary.json'
    return root/a.fcps_eval_group/f'seed_{seed}'/'json'/f'low_nfe_fcps_s{seed}_summary.json'


def load_rows(scheduler):
    rows=[]
    for seed in seeds:
        path=summary_path(scheduler, seed)
        if not path.is_file():
            raise SystemExit(f'Missing {path}')
        obj=json.loads(path.read_text())
        by={int(r['requested_steps']):r for r in obj['test_results']}
        for n in nfes:
            if n not in by:
                raise SystemExit(f'{path} has no {n}-NFE result (available: {sorted(by)})')
            r=by[n]
            rows.append((seed,n,float(r['mae']),float(r['rmse']),float(r['crps']),int(r['actual_nfe'])))
    return rows


def summarize(rows, n):
    q=[r for r in rows if r[1]==n]
    arr=np.array([[r[2],r[3],r[4]] for r in q],dtype=float)
    return arr.mean(0), arr.std(0,ddof=1), all(r[5]==n for r in q)


def print_row(n, mean, std, exact, label=''):
    print(f'{label}{n:>2} NFE  MAE {mean[0]:.4f} ± {std[0]:.4f} | RMSE {mean[1]:.4f} ± {std[1]:.4f} | CRPS {mean[2]:.5f} ± {std[2]:.5f} | exact NFE={exact}')


out=Path(a.out)
if a.scheduler in ('mas_snr','fcps'):
    rows=load_rows(a.scheduler)
    with out.open('w') as f:
        f.write('nfe,mae_mean,mae_std,rmse_mean,rmse_std,crps_mean,crps_std,all_actual_nfe_equal_requested\n')
        for n in nfes:
            mean,std,exact=summarize(rows,n)
            f.write(f'{n},{mean[0]:.8f},{std[0]:.8f},{mean[1]:.8f},{std[1]:.8f},{mean[2]:.8f},{std[2]:.8f},{exact}\n')
            print_row(n,mean,std,exact)
    print(f'Wrote {out}')
    raise SystemExit(0)

# --scheduler both
tables={s:load_rows(s) for s in ('mas_snr','fcps')}
with out.open('w') as f:
    f.write('scheduler,nfe,mae_mean,mae_std,rmse_mean,rmse_std,crps_mean,crps_std,all_actual_nfe_equal_requested\n')
    for s,label in (('mas_snr','MASS'),('fcps','FCPS')):
        for n in nfes:
            mean,std,exact=summarize(tables[s],n)
            f.write(f'{s},{n},{mean[0]:.8f},{std[0]:.8f},{mean[1]:.8f},{std[1]:.8f},{mean[2]:.8f},{std[2]:.8f},{exact}\n')
            print_row(n,mean,std,exact,label=f'{label:<5}')
print(f'Wrote {out}')

paired=out.with_name(out.stem+'_paired.csv')
print(f'\nSeed-paired FCPS - MASS (positive = MASS better; seeds {seeds}):')
with paired.open('w') as f:
    f.write('nfe,metric,mass_mean,fcps_mean,diff_mean,diff_std,rel_diff_pct,seeds_mass_better,num_seeds\n')
    for n in nfes:
        m={r[0]:r for r in tables['mas_snr'] if r[1]==n}
        c={r[0]:r for r in tables['fcps'] if r[1]==n}
        for j,name in enumerate(METRICS):
            mv=np.array([m[s][2+j] for s in seeds]); cv=np.array([c[s][2+j] for s in seeds])
            d=cv-mv; rel=100.0*d.mean()/mv.mean(); wins=int((d>0).sum())
            f.write(f'{n},{name},{mv.mean():.8f},{cv.mean():.8f},{d.mean():.8f},{d.std(ddof=1):.8f},{rel:.4f},{wins},{len(seeds)}\n')
            print(f'{n:>2} NFE  {name.upper():<4} diff {d.mean():+.5f} ± {d.std(ddof=1):.5f}  ({rel:+.2f}% of MASS)  MASS better on {wins}/{len(seeds)} seeds')
print(f'Wrote {paired}')
