from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from detailed_training import EvaluationConfig, TrainingConfig, evaluate_model_nfe_sweep, train_paired_denoiser, write_csv, write_json
from diagnostics import compute_forward_plot_arrays, run_forward_diagnostics
from schedules import (
    build_schedules,
    mass_endpoints_from_tolerances,
    scalar_cbar_from_terminal_a2,
    schedule_metadata,
)
from synthetic_problem import FrozenSyntheticProblem
from theory_checks import mass_M, scalar_stiffness_lower_bound, guaranteed_improvement_ratio


def _parse_csv(text: str, cast):
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Controlled learned-GNN synthetic study: isotropic VP vs scalar GAD vs MASS, plus EM/DDIM/DPM++2M NFE sweeps."
    )
    p.add_argument("--problem", required=True, help="Frozen .npz from synthetic_problem.py")
    p.add_argument("--out", required=True)
    p.add_argument("--evaluation_split", default="test", choices=["val", "test"], help="Reference split used for generation metrics. Hyperparameter tuning must use val; final reporting uses test.")
    p.add_argument("--candidate_id", default="", help="Optional hyperparameter-candidate identifier propagated to outputs.")
    p.add_argument("--study_phase", default="", help="Optional workflow phase label propagated to outputs.")
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--terminal_a2", type=float, default=1e-3, help="Scalar slowest-mode terminal attenuation target.")
    p.add_argument("--scalar_cbar_T", type=float, default=None, help="Override automatic scalar endpoint calibration; used for fixed-parameter robustness ablations.")
    p.add_argument("--mass_start_a2_error", type=float, default=1e-3)
    p.add_argument("--mass_terminal_a2", type=float, default=1e-3)
    p.add_argument("--mass_r_max", type=float, default=None, help="Override automatic MASS r_max calibration.")
    p.add_argument("--mass_r_min", type=float, default=None, help="Override automatic MASS r_min calibration.")
    p.add_argument("--fcps_alpha", type=float, default=4.0)
    p.add_argument("--fcps_cmin", type=float, default=0.1)
    p.add_argument("--schedulers", default="Isotropic-VP-Uniform,Isotropic-VP-Cosine,Uniform,FCPS,Cosine,MAS-SNR")
    p.add_argument("--methods", default="em,ddim,dpmpp2m")
    p.add_argument("--steps", default="10,25,50,100,250", help="Fallback step grid used by methods without an explicit method-specific grid.")
    p.add_argument("--em_steps", default=None, help="Optional EM-only step/NFE grid, e.g. 50,100,250,...,3500.")
    p.add_argument("--ddim_steps", default=None, help="Optional DDIM-only step/NFE grid.")
    p.add_argument("--dpm_steps", default=None, help="Optional DPM-Solver++ 2M step/NFE grid; use the same values as MASS for complementarity tests.")
    p.add_argument("--sample_seeds", default="6001,6002,6003,6004,6005")
    p.add_argument("--plot_export_steps", default="50,250,1000,2500,3500", help="Only these sampling steps export heavy per-mode/sample NPZ/CSV plot data.")
    p.add_argument("--plot_export_sample_seeds", default="6001", help="Sampling seeds for which plot-ready generated samples/spectra are retained.")
    p.add_argument("--num_eval_samples", type=int, default=1024)
    p.add_argument("--t0", type=float, default=1e-3)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--sampler_init_mode", default="algorithm2", choices=["algorithm2", "stationary", "rho"])
    p.add_argument("--cosine_s", type=float, default=0.008)
    p.add_argument("--cosine_terminal_epsilon", type=float, default=1e-3)
    p.add_argument("--iso_cosine_s", type=float, default=None, help="Cosine offset of the Isotropic-VP controls; defaults to --cosine_s (previous behaviour).")
    p.add_argument("--iso_cosine_terminal_epsilon", type=float, default=None, help="Terminal truncation of the Isotropic-VP controls; defaults to --cosine_terminal_epsilon.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    # Paired training controls.
    p.add_argument("--model_seed", type=int, default=3001)
    p.add_argument("--train_seed", type=int, default=4001)
    p.add_argument("--validation_seed", type=int, default=5001)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--cheb_order", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--train_t_min", type=float, default=1e-2)
    p.add_argument("--val_every_epochs", type=int, default=2)
    p.add_argument("--val_num_samples", type=int, default=1024)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--score_nu_floor", type=float, default=1e-12, help="Numerical denominator floor for plug-in score; default is effectively exact.")
    p.add_argument("--score_clip", type=float, default=None, help="Optional absolute score clipping. Default None preserves the theoretical vector field.")
    p.add_argument("--skip_forward_diagnostics", action="store_true")
    p.add_argument("--train_only", action="store_true")
    args = p.parse_args()

    if args.gamma <= 0 or args.sigma <= 0:
        raise ValueError("gamma and sigma must be positive")
    sched_names = list(_parse_csv(args.schedulers, str))
    if "FCPS" in sched_names and args.fcps_alpha <= 0:
        raise ValueError(
            "A row labelled FCPS must use alpha>0. alpha=0 is algebraically identical to Uniform; use the Uniform row instead."
        )

    problem = FrozenSyntheticProblem.load(args.problem)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    lam_gamma_np = problem.shifted_eigenvalues(args.gamma)
    s_k_emp = problem.s_k_train
    cbar_T = float(args.scalar_cbar_T) if args.scalar_cbar_T is not None else scalar_cbar_from_terminal_a2(args.gamma, args.terminal_a2)
    auto_r_max, auto_r_min = mass_endpoints_from_tolerances(
        lam_gamma_np,
        s_k_emp,
        args.sigma,
        start_a2_error=args.mass_start_a2_error,
        terminal_a2=args.mass_terminal_a2,
    )
    r_max = float(args.mass_r_max) if args.mass_r_max is not None else auto_r_max
    r_min = float(args.mass_r_min) if args.mass_r_min is not None else auto_r_min
    if not (r_max > r_min > 0):
        raise ValueError("Require mass_r_max > mass_r_min > 0")
    schedules = build_schedules(
        lam_gamma=lam_gamma_np,
        s_k=s_k_emp,
        sigma=args.sigma,
        T=args.T,
        fcps_c0=cbar_T,
        fcps_cmin=args.fcps_cmin,
        fcps_alpha=args.fcps_alpha,
        r_max=r_max,
        r_min=r_min,
        match_terminal_energy=False,
        uniform_cbar_T=cbar_T,
        cosine_cbar_T=cbar_T,
        cosine_s=args.cosine_s,
        cosine_terminal_epsilon=args.cosine_terminal_epsilon,
        include_isotropic_vp=True,
        iso_cosine_s=args.iso_cosine_s,
        iso_cosine_terminal_epsilon=args.iso_cosine_terminal_epsilon,
    )

    for name in sched_names:
        if name not in schedules:
            raise KeyError(f"Unknown scheduler {name}; available={sorted(k for k in schedules if k!='r_min')}")

    V = torch.tensor(problem.V, device=device, dtype=dtype)
    lam_gamma = torch.tensor(lam_gamma_np, device=device, dtype=dtype)
    s_k = torch.tensor(s_k_emp, device=device, dtype=dtype)
    W = torch.tensor(problem.W, device=device, dtype=dtype)
    L = torch.tensor(problem.L, device=device, dtype=dtype)
    X_train = torch.tensor(problem.X_train, device=device, dtype=dtype)
    X_val = torch.tensor(problem.X_val, device=device, dtype=dtype)

    train_cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_embed_dim=args.time_embed_dim,
        cheb_order=args.cheb_order,
        dropout=args.dropout,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        train_t_min=args.train_t_min,
        val_every_epochs=args.val_every_epochs,
        val_num_samples=args.val_num_samples,
        grad_clip_norm=args.grad_clip_norm,
        model_seed=args.model_seed,
        train_seed=args.train_seed,
        validation_seed=args.validation_seed,
        score_nu_floor=args.score_nu_floor,
        score_clip=args.score_clip,
    )
    method_steps = {}
    if args.em_steps:
        method_steps["em"] = _parse_csv(args.em_steps, int)
    if args.ddim_steps:
        method_steps["ddim"] = _parse_csv(args.ddim_steps, int)
    if args.dpm_steps:
        method_steps["dpmpp2m"] = _parse_csv(args.dpm_steps, int)
    eval_cfg = EvaluationConfig(
        methods=_parse_csv(args.methods, str),
        steps=_parse_csv(args.steps, int),
        method_steps=method_steps or None,
        sample_seeds=_parse_csv(args.sample_seeds, int),
        num_samples=args.num_eval_samples,
        t0=args.t0,
        T=args.T,
        ddim_eta=args.ddim_eta,
        init_mode=args.sampler_init_mode,
        plot_export_steps=_parse_csv(args.plot_export_steps, int),
        plot_export_sample_seeds=_parse_csv(args.plot_export_sample_seeds, int),
    )

    config = {
        "problem": str(Path(args.problem).resolve()),
        "evaluation_split": args.evaluation_split,
        "candidate_id": args.candidate_id,
        "study_phase": args.study_phase,
        "dtype": args.dtype,
        "problem_metadata": problem.metadata,
        "gamma": args.gamma,
        "sigma": args.sigma,
        "lambda_gamma_min": float(lam_gamma_np[0]),
        "lambda_gamma_max": float(lam_gamma_np[-1]),
        "lambda_gamma_condition_number": float(lam_gamma_np[-1] / lam_gamma_np[0]),
        "scalar_cbar_T": cbar_T,
        "terminal_a2_target": args.terminal_a2,
        "mass_r_max": r_max,
        "mass_r_min": r_min,
        "fcps_alpha": args.fcps_alpha,
        "fcps_cmin": args.fcps_cmin,
        "cosine_s": args.cosine_s,
        "cosine_terminal_epsilon": args.cosine_terminal_epsilon,
        "iso_cosine_s": args.cosine_s if args.iso_cosine_s is None else args.iso_cosine_s,
        "iso_cosine_terminal_epsilon": args.cosine_terminal_epsilon if args.iso_cosine_terminal_epsilon is None else args.iso_cosine_terminal_epsilon,
        "endpoint_calibration_mode": {
            "scalar": "fixed_override" if args.scalar_cbar_T is not None else "terminal_a2_calibrated",
            "mass": "fixed_override" if (args.mass_r_max is not None or args.mass_r_min is not None) else "attenuation_tolerance_calibrated",
        },
        "mass_endpoint_calibration": {
            "start_a2_error": args.mass_start_a2_error,
            "terminal_a2": args.mass_terminal_a2,
        },
        "paired_invariants": {
            "same_problem_file_all_schedulers": True,
            "same_gamma_sigma_within_this_comparison": True,
            "same_model_initialization_seed": args.model_seed,
            "same_minibatch_time_noise_seed": args.train_seed,
            "sampler_independent_checkpoint_selection": True,
        },
        "training": asdict(train_cfg),
        "evaluation": asdict(eval_cfg),
        "schedules": {name: schedule_metadata(name, schedules[name]) for name in sched_names},
    }
    write_json(out / "run_config.json", config)

    # Forward diagnostics: SNR equality is a sanity check; optimal stiffness is the theory-facing quantity.
    if not args.skip_forward_diagnostics:
        run_forward_diagnostics(
            X_train_np=problem.X_train,
            V_np=problem.V,
            lam_gamma_np=lam_gamma_np,
            sigma=args.sigma,
            schedules=schedules,
            scheduler_names=sched_names,
            dataset_name=problem.metadata.get("family", "frozen_problem"),
            out_dir=out / "forward_diagnostics",
            num_times=512,
            num_extreme_modes=4,
            terminal_num_samples=min(args.num_eval_samples, len(problem.X_train)),
            seed=args.validation_seed,
            pmin=1e-10,
            t_min=max(args.t0, 1e-5),
            stiffness_h_proxy=0.0,
        )

    diag_metrics = {}
    for name in sched_names:
        _, m = compute_forward_plot_arrays(
            lam_gamma_np, s_k_emp, args.sigma, name, schedules[name], num_times=2048, t_min=args.t0
        )
        diag_metrics[name] = m
    theory_rows: list[dict] = []
    if "MAS-SNR" in sched_names:
        mass_upper_h0 = mass_M(schedules["MAS-SNR"])
        mass_measured_h0 = diag_metrics["MAS-SNR"]["max_stiffness_proxy"]
        theory_rows.append(
            {
                "schedule": "MAS-SNR",
                "diagnostic": "schedule_only_h0",
                "measured_stiffness": mass_measured_h0,
                "mass_theory_upper_bound_M": mass_upper_h0,
                "bound_satisfied": bool(mass_measured_h0 <= mass_upper_h0 * (1 + 5e-3)),
                "log_snr_interpretation": "construction/implementation sanity check",
            }
        )
        for name in ["FCPS", "Uniform", "Cosine"]:
            if name not in sched_names:
                continue
            scalar = scalar_stiffness_lower_bound(name, schedules[name], lam_gamma_np, s_k_emp, args.sigma)
            measured = diag_metrics[name]["max_stiffness_proxy"]
            theory_rows.append(
                {
                    "schedule": name,
                    "diagnostic": "schedule_only_h0",
                    "measured_stiffness": measured,
                    **scalar,
                    "lower_bound_satisfied": bool(measured + 1e-9 >= float(scalar["scalar_stiffness_lower_bound"])) if scalar["scalar_theorem_condition_snr_le_1_over_3"] else "condition_not_met",
                    "measured_ratio_Lscalar_over_LMASS": float(measured / mass_measured_h0),
                    "theory_ratio_lower_bound": guaranteed_improvement_ratio(float(scalar["scalar_stiffness_lower_bound"]), mass_upper_h0),
                    "ratio_note": "the theorem gives a conservative lower bound, not an exact point prediction",
                }
            )
    write_csv(out / "theory_stiffness_summary.csv", theory_rows)

    eval_ref_np = problem.X_val if args.evaluation_split == "val" else problem.X_test
    all_rows: list[dict] = []
    training_rows: list[dict] = []
    for name in sched_names:
        schedule_dir = out / name.replace("/", "_")
        model, tr_summary = train_paired_denoiser(
            schedule_name=name,
            schedule_obj=schedules[name],
            X_train=X_train,
            X_val=X_val,
            V=V,
            lam_gamma=lam_gamma,
            s_k=s_k,
            W=W,
            L=L,
            sigma=args.sigma,
            T=args.T,
            cfg=train_cfg,
            out_dir=schedule_dir,
        )
        training_rows.append(
            {
                "schedule": name,
                "best_epoch": tr_summary["best_epoch"],
                "best_denoising_val_loss": tr_summary["best_denoising_val_loss"],
                "gamma": args.gamma,
                "sigma": args.sigma,
                "model_seed": args.model_seed,
                "fcps_alpha": args.fcps_alpha,
                "fcps_cmin": args.fcps_cmin,
                "scalar_cbar_T": cbar_T,
                "mass_r_max": r_max,
                "mass_r_min": r_min,
                "cosine_s": args.cosine_s,
                "cosine_terminal_epsilon": args.cosine_terminal_epsilon,
                "evaluation_split": args.evaluation_split,
                "candidate_id": args.candidate_id,
                "study_phase": args.study_phase,
        "dtype": args.dtype,
                "problem_sha256": problem.metadata.get("problem_sha256", ""),
                "dataset_family": problem.metadata.get("family", ""),
            }
        )
        write_csv(out / "training_summary.csv", training_rows)
        if args.train_only:
            continue
        rows = evaluate_model_nfe_sweep(
            model=model,
            schedule_name=name,
            schedule_obj=schedules[name],
            X_ref_np=eval_ref_np,
            V_np=problem.V,
            L_np=problem.L,
            V=V,
            lam_gamma=lam_gamma,
            s_k=s_k,
            sigma=args.sigma,
            cfg=eval_cfg,
            out_dir=schedule_dir,
            shared_fields={
                "problem_sha256": problem.metadata.get("problem_sha256", ""),
                "dataset_family": problem.metadata.get("family", ""),
                "gamma": args.gamma,
                "sigma": args.sigma,
                "fcps_alpha": args.fcps_alpha,
                "fcps_cmin": args.fcps_cmin,
                "scalar_cbar_T": cbar_T,
                "mass_r_max": r_max,
                "mass_r_min": r_min,
                "cosine_s": args.cosine_s,
                "cosine_terminal_epsilon": args.cosine_terminal_epsilon,
                "evaluation_split": args.evaluation_split,
                "candidate_id": args.candidate_id,
                "study_phase": args.study_phase,
        "dtype": args.dtype,
                "model_seed": args.model_seed,
                "train_seed": args.train_seed,
            },
        )
        all_rows.extend(rows)
        write_csv(out / "results_raw.csv", all_rows)

    print(f"Detailed learned study complete: {out}")


if __name__ == "__main__":
    main()
