from __future__ import annotations

import threading
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_models import diff
from .graph import SpatialGraph
import logging

from .schedules import MASCorridor, build_schedule, schedule_stiffness_report


class GADMASFENCEBase(nn.Module):
    def __init__(self, target_dim: int, config, device, graph: SpatialGraph, spectral_power: np.ndarray):
        super().__init__()
        self.target_dim = int(target_dim)
        self.device = torch.device(device)
        self.config = config
        self.graph = graph
        self.register_buffer("lam_gamma", self.graph.lam_gamma.detach().clone())
        self.register_buffer("s_k", torch.tensor(np.asarray(spectral_power, dtype=np.float32), device=self.device))

        self.emb_time_dim = int(config["model"]["timeemb"])
        self.emb_feature_dim = int(config["model"]["featureemb"])
        self.target_strategy = config["model"]["target_strategy"]
        self.device_cond = torch.device(config["model"].get("cond_device", str(device)))
        self.device_uncond = torch.device(config["model"].get("uncond_device", str(device)))
        self.results = {}

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = str(self.emb_total_dim)
        input_dim = 2
        self.diffmodel_cond = diff(config_diff, input_dim)
        self.diffmodel_uncond = diff(config_diff, input_dim)

        self.train_levels = int(config_diff.get("train_levels", config_diff.get("num_steps", 50)))
        self.sample_nfe = int(config_diff.get("sample_nfe", config_diff.get("num_steps", 50)))
        self.num_steps = self.train_levels
        self.validation_levels = int(config_diff.get("validation_levels", min(self.train_levels, 16)))
        self.sample_grid_power = float(config_diff.get("sample_grid_power", 1.0))
        self.continuous_time_training = str(config_diff.get("continuous_time_training", "1")).lower() in {"1", "true", "yes"}
        self.nsample = int(config_diff.get("nsample", 10))

        gad = config["gad"] if config.has_section("gad") else {}
        self.sigma = float(gad.get("sigma", 1.0))
        self.T_diff = float(gad.get("T_diff", 1.0))
        self.t_eps = float(gad.get("t_eps", 1e-3))
        self.r_max = float(gad.get("r_max", 1e4))
        self.r_min = float(gad.get("r_min", 1e-4))
        self.objective = str(gad.get("objective", "score"))
        self.x0_loss_weight = float(gad.get("x0_loss_weight", 0.0))
        self.sampler = str(gad.get("sampler", "em")).lower()
        self.heun_stochastic = str(gad.get("heun_stochastic", "0")).lower() in {"1", "true", "yes"}
        self.clamp_observed = str(gad.get("clamp_observed", "0")).lower() in {"1", "true", "yes"}
        self.score_clip = float(gad.get("score_clip", 0.0))
        self.posterior_cluster_average = str(gad.get("posterior_cluster_average", "0")).lower() in {"1", "true", "yes"}
        self.final_x0_projection = str(gad.get("final_x0_projection", "1")).lower() in {"1", "true", "yes"}
        self.corridor, self.schedule_calibration = build_schedule(gad, lam_gamma=self.lam_gamma, s_k=self.s_k)
        self.schedule_kind = self.corridor.kind
        if self.schedule_kind != "mass":
            logging.info("forward schedule=%s %s calibration=%s",
                         self.schedule_kind, self.corridor.describe(), self.schedule_calibration)

        self.loss_mse_weight = float(gad.get("loss_mse_weight", 1.0))
        self.loss_huber_weight = float(gad.get("loss_huber_weight", 0.25))
        self.loss_relative_weight = float(gad.get("loss_relative_weight", 0.05))
        self.loss_spectral_weight = float(gad.get("loss_spectral_weight", 0.05))
        self.huber_delta_physical = float(gad.get("huber_delta_physical", 10.0))
        self.relative_eps_physical = float(gad.get("relative_eps_physical", 1e-4))
        self.train_mean = float(config["train"].get("data_train_mean", 0.0))
        self.train_std = float(config["train"].get("data_train_std", 1.0))
        self.register_buffer(
            "spectral_loss_weights",
            (torch.rsqrt(torch.clamp(self.s_k.detach(), min=1e-8)) / torch.rsqrt(torch.clamp(self.s_k.detach(), min=1e-8)).mean()).detach(),
        )

        budgets = [int(x.strip()) for x in str(gad.get("low_nfe_budgets", "6,8,12,20,50")).split(",") if x.strip()]
        probs = [float(x.strip()) for x in str(gad.get("low_nfe_probs", "0.15,0.30,0.30,0.15,0.10")).split(",") if x.strip()]
        if len(budgets) != len(probs) or not budgets:
            raise ValueError("low_nfe_budgets and low_nfe_probs must have the same nonzero length")
        psum = sum(probs)
        if psum <= 0:
            raise ValueError("low_nfe_probs must sum to > 0")
        self.low_nfe_budgets = budgets
        self.low_nfe_probs = [p / psum for p in probs]
        self.low_nfe_curriculum_enabled = False

        self.phase = "cond"
        self.p_drop = float(config["model"].get("p_drop", 0.1))
        self.cfg_scale = float(config["model"].get("cfg_scale", 1.0))
        self.guidance = str(config["model"].get("guidance", "cfg")).lower()

        self.uses_fbg = self.guidance in {"orig_fbg", "fbg", "cov_fbg"}
        self.original_fbg = self.guidance == "orig_fbg"
        self.covariance_aware_fbg = self.guidance == "cov_fbg"

        if self.uses_fbg:
            fbg = config["fbg"]
            self.pi = float(fbg["pi"])
            self.fbg_mode = fbg["mode"]
            self.constant_guidance = float(fbg["constant_guidance"])
            self.max_guidance = float(fbg["max_guidance"])
            self.minimal_log_posterior = np.log(((1 - self.pi) * (self.max_guidance - self.constant_guidance + 1)) / (self.max_guidance - self.constant_guidance))
            if "temp" in fbg and "offset" in fbg:
                self.temp = float(fbg["temp"])
                self.offset = float(fbg["offset"])
            elif "t0" in fbg and "t1" in fbg:
                self.offset = self._calculate_fbg_offset(float(fbg["t0"]))
                self.temp = self._calculate_fbg_temp(self.offset, float(fbg["t1"]))
            else:
                raise ValueError("FBG config must provide either temp/offset or t0/t1")
            self.temp *= float(fbg.get("temp_scale", 1.0))
            self.logp_delta_clip = float(fbg.get("logp_delta_clip", 0.0))
            if self.fbg_mode == "cluster":
                self.n_clusters = int(fbg["n_clusters"])

    def _calculate_fbg_offset(self, t0, verbose=True):
        lambda_ref = self.max_guidance
        log_term = np.log((1 - self.pi) * lambda_ref / (lambda_ref - 1))
        offset = (1 / ((1 - t0) * self.sample_nfe)) * log_term
        return round(float(offset), 5)

    def _calculate_fbg_temp(self, offset, t1, alpha=10.0, verbose=True):
        times = self.corridor.sample_times(
            self.sample_nfe, device=self.device, dtype=torch.float32, grid_power=self.sample_grid_power
        )
        idx_float = t1 * (self.sample_nfe - 1)
        idx_lower = int(np.floor(idx_float))
        idx_lower = int(np.clip(idx_lower, 1, self.sample_nfe - 1))
        coeff = self._reverse_coeff_time(times[idx_lower], times[idx_lower - 1])
        sigma_sq_at_t1 = float(coeff["eta2"].mean().detach().cpu().item())
        temp = abs(2 * sigma_sq_at_t1 / alpha * offset)
        return round(float(temp), 5)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            if num_masked > 0:
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        side_mask = cond_mask.unsqueeze(1)
        return torch.cat([side_info, side_mask], dim=1)

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    def _kmeans_torch_cluster(self, features, n_clusters, num_iter=20):
        B, K, D = features.shape
        device = features.device
        centroids = torch.zeros(B, n_clusters, D, device=device)
        initial_indices = torch.randint(0, K, (B,), device=device)
        centroids[:, 0, :] = features[torch.arange(B), initial_indices, :]
        dists = torch.cdist(features, centroids[:, 0, :].unsqueeze(1)).squeeze(2)
        for i in range(1, n_clusters):
            probs = torch.clamp(dists, min=1e-12)
            next_centroid_indices = torch.multinomial(probs, 1).squeeze(1)
            centroids[:, i, :] = features[torch.arange(B), next_centroid_indices, :]
            new_dists = torch.cdist(features, centroids[:, i, :].unsqueeze(1)).squeeze(2)
            dists = torch.minimum(dists, new_dists)
        for _ in range(num_iter):
            dists = torch.cdist(features, centroids)
            assignments = torch.argmin(dists, dim=2)
            one_hot = nn.functional.one_hot(assignments, n_clusters).float()
            counts = one_hot.sum(dim=1).clamp_(min=1.0)
            centroids = torch.bmm(one_hot.transpose(1, 2), features) / counts.unsqueeze(2)
        return assignments

    def set_sample_nfe(self, nfe: int) -> None:
        nfe = int(nfe)
        if nfe < 2:
            raise ValueError("sample_nfe must be >= 2")
        self.sample_nfe = nfe
        if self.uses_fbg and "t0" in self.config["fbg"] and "t1" in self.config["fbg"]:
            self.offset = self._calculate_fbg_offset(float(self.config["fbg"]["t0"]), verbose=False)
            self.temp = self._calculate_fbg_temp(self.offset, float(self.config["fbg"]["t1"]), verbose=False)
            self.temp *= float(self.config["fbg"].get("temp_scale", 1.0))

    def set_low_nfe_curriculum(self, enabled: bool = True) -> None:
        self.low_nfe_curriculum_enabled = bool(enabled)

    def sample_times(self) -> torch.Tensor:
        return self.corridor.sample_times(
            self.sample_nfe,
            device=self.device,
            dtype=torch.float32,
            grid_power=self.sample_grid_power,
        )

    def index_to_time(self, idx: torch.Tensor) -> torch.Tensor:
        """Legacy training-grid index -> physical time."""
        return self.corridor.time_from_index(
            idx.to(device=self.device, dtype=torch.float32), self.train_levels
        ).to(self.device)

    def diffusion_condition(self, t: torch.Tensor) -> torch.Tensor:
        return self.corridor.condition(t).to(device=self.device, dtype=torch.float32)

    def channel_params_time(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.corridor.channel_params(self.lam_gamma, self.s_k, self.sigma, t)

    def schedule_state(self) -> dict:
        """Provenance record stored in every checkpoint and training summary."""
        state = dict(self.corridor.describe())
        if self.schedule_calibration:
            state["calibration"] = dict(self.schedule_calibration)
        return state

    def stiffness_report(self, nfes=()) -> dict:
        return schedule_stiffness_report(self.corridor, self.lam_gamma, self.s_k, self.sigma, nfes)

    def channel_params_idx(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.channel_params_time(self.index_to_time(idx))

    def q_sample_time(self, x0: torch.Tensor, t: torch.Tensor, noise_spec: Optional[torch.Tensor] = None):
        pars = self.channel_params_time(t)
        x0_spec = self.graph.to_spectral(x0)
        if noise_spec is None:
            noise_spec = torch.randn_like(x0_spec)
        xt_spec = pars["a"][:, :, None] * x0_spec + torch.sqrt(pars["nu"][:, :, None]) * noise_spec
        xt = self.graph.from_spectral(xt_spec)
        return xt, xt_spec, x0_spec, pars, noise_spec

    def q_sample(self, x0: torch.Tensor, idx: torch.Tensor, noise_spec: Optional[torch.Tensor] = None):
        return self.q_sample_time(x0, self.index_to_time(idx), noise_spec=noise_spec)

    def _sample_training_times(self, batch_size: int) -> torch.Tensor:
        if self.low_nfe_curriculum_enabled:
            probs = torch.tensor(self.low_nfe_probs, device=self.device, dtype=torch.float32)
            choices = torch.multinomial(probs, batch_size, replacement=True)
            out = torch.empty(batch_size, device=self.device, dtype=torch.float32)
            for ci, budget in enumerate(self.low_nfe_budgets):
                which = torch.where(choices == ci)[0]
                if which.numel() == 0:
                    continue
                grid = self.corridor.sample_times(
                    budget, device=self.device, dtype=torch.float32, grid_power=self.sample_grid_power
                )
                level = torch.randint(0, budget, (which.numel(),), device=self.device)
                out[which] = grid[level]
            return out
        if self.continuous_time_training:
            u = torch.rand(batch_size, device=self.device, dtype=torch.float32)
            return self.corridor.time_from_u(u)
        idx = torch.randint(0, self.train_levels, (batch_size,), device=self.device)
        return self.index_to_time(idx)

    def score_from_x0_hat(self, xt: torch.Tensor, x0_hat: torch.Tensor, pars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        xt_spec = self.graph.to_spectral(xt)
        x0_hat_spec = self.graph.to_spectral(x0_hat)
        score_spec = (pars["a"][:, :, None] * x0_hat_spec - xt_spec) / torch.clamp(pars["nu"][:, :, None], min=1e-12)
        if self.score_clip > 0:
            score_spec = torch.clamp(score_spec, -self.score_clip, self.score_clip)
        score = self.graph.from_spectral(score_spec)
        return score, score_spec

    def target_score(self, xt_spec: torch.Tensor, x0_spec: torch.Tensor, pars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        score_spec = (pars["a"][:, :, None] * x0_spec - xt_spec) / torch.clamp(pars["nu"][:, :, None], min=1e-12)
        if self.score_clip > 0:
            score_spec = torch.clamp(score_spec, -self.score_clip, self.score_clip)
        return self.graph.from_spectral(score_spec), score_spec

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1):
        times = self.corridor.sample_times(
            self.validation_levels, device=self.device, dtype=torch.float32, grid_power=1.0
        )
        loss_sum = 0.0
        for t_scalar in times:
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info,
                is_train, is_phase1, set_time=float(t_scalar.item())
            )
            loss_sum += loss.detach()
        return loss_sum / float(len(times))

    def _task_aligned_x0_loss(self, x0_hat, observed_data, target_mask, *, is_phase1: bool):
        denom = target_mask.sum().clamp(min=1.0)
        err = (x0_hat - observed_data) * target_mask
        mse = err.pow(2).sum() / denom
        if is_phase1:
            return mse

        beta = max(self.huber_delta_physical / max(abs(self.train_std), 1e-8), 1e-4)
        huber_elem = F.smooth_l1_loss(x0_hat, observed_data, reduction="none", beta=beta)
        huber = (huber_elem * target_mask).sum() / denom

        pred_phys = x0_hat * self.train_std + self.train_mean
        target_phys = observed_data * self.train_std + self.train_mean
        valid = (target_mask > 0) & (target_phys > self.relative_eps_physical)
        rel = torch.zeros_like(err)
        rel[valid] = torch.abs(pred_phys[valid] - target_phys[valid]) / target_phys[valid].clamp(min=self.relative_eps_physical)
        relative = torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).sum() / denom

        err_spec = self.graph.to_spectral(err)
        spectral = (
            err_spec.pow(2) * self.spectral_loss_weights[None, :, None].to(err_spec.dtype)
        ).sum() / denom

        return (
            self.loss_mse_weight * mse
            + self.loss_huber_weight * huber
            + self.loss_relative_weight * relative
            + self.loss_spectral_weight * spectral
        )

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info,
        is_train, is_phase1, set_t=-1, set_time: Optional[float] = None
    ):
        B, N, L = observed_data.shape
        if not is_train:
            if set_time is not None:
                t = torch.full((B,), float(set_time), device=self.device, dtype=torch.float32)
            else:
                idx = torch.full((B,), int(set_t), device=self.device, dtype=torch.long)
                t = self.index_to_time(idx)
        else:
            t = self._sample_training_times(B)
        noisy_data, xt_spec, x0_spec, pars, _ = self.q_sample_time(observed_data, t)
        diffusion_coord = self.diffusion_condition(t)

        if is_phase1:
            input_uncond = torch.cat([torch.zeros_like(noisy_data).unsqueeze(1), noisy_data.unsqueeze(1)], dim=1)
            null_side_info = side_info.clone(); null_side_info[:, -1:, :, :] = 0
            x0_hat, _ = self.diffmodel_uncond(input_uncond, null_side_info, diffusion_coord)
            target_mask = observed_mask
        else:
            input_cond = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
            x0_hat, _ = self.diffmodel_cond(input_cond, side_info, diffusion_coord)
            target_mask = observed_mask - cond_mask

        score_pred, _ = self.score_from_x0_hat(noisy_data, x0_hat, pars)
        score_true, _ = self.target_score(xt_spec, x0_spec, pars)
        denom = target_mask.sum().clamp(min=1.0)
        score_loss = (((score_pred - score_true) * target_mask) ** 2).sum() / denom
        x0_loss = self._task_aligned_x0_loss(x0_hat, observed_data, target_mask, is_phase1=is_phase1)
        if self.objective == "x0":
            return x0_loss
        if self.objective in {"score", "dsm"}:
            return score_loss + self.x0_loss_weight * x0_loss if self.x0_loss_weight > 0 else score_loss
        if self.objective in {"score_x0", "hybrid"}:
            return score_loss + self.x0_loss_weight * x0_loss
        raise ValueError(f"Unknown gad objective={self.objective}")

    def _reverse_coeff_time(self, t_cur: torch.Tensor | float, t_next: torch.Tensor | float):
        return self.corridor.step_coefficients_time(
            self.lam_gamma, self.s_k, self.sigma,
            t_cur, t_next, device=self.device, dtype=torch.float32,
        )

    def _reverse_mean_spec_time(self, xt: torch.Tensor, score_spec: torch.Tensor, t_cur, t_next):
        coeff = self._reverse_coeff_time(t_cur, t_next)
        xt_spec = self.graph.to_spectral(xt)
        dt = coeff["dt"].to(xt.device, xt.dtype)
        c_mode = coeff["c_mode"].to(xt.device, xt.dtype)
        drift = c_mode[None, :, None] * (
            self.lam_gamma[None, :, None].to(xt.dtype) * xt_spec
            + 2.0 * (self.sigma ** 2) * score_spec
        )
        mu_spec = xt_spec + drift * dt
        return mu_spec, coeff

    def _sample_terminal(self, observed_data: torch.Tensor, terminal_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, L = observed_data.shape
        if terminal_time is None:
            terminal_time = self.sample_times()[-1]
        t = torch.full((B,), float(torch.as_tensor(terminal_time).item()), device=self.device, dtype=torch.float32)
        pars = self.channel_params_time(t)
        noise = torch.randn(B, N, L, device=self.device, dtype=observed_data.dtype)
        xt_spec = torch.sqrt(pars["nu"][:, :, None]) * noise
        return self.graph.from_spectral(xt_spec)

    def _build_noisy_cond_history(self, observed_data: torch.Tensor, cond_mask: torch.Tensor, sample_times: torch.Tensor):
        x_obs = observed_data * cond_mask
        hist = []
        for t_scalar in sample_times:
            t = torch.full((observed_data.shape[0],), float(t_scalar.item()), device=self.device, dtype=torch.float32)
            xt, _, _, _, _ = self.q_sample_time(x_obs, t)
            hist.append(xt * cond_mask)
        return hist

    def _guidance_scale(self, log_posterior, cluster_labels: Optional[torch.Tensor], B: int, N: int):
        guidance_scale = torch.exp(log_posterior) / torch.clamp(torch.exp(log_posterior) - (1 - self.pi), min=1e-12)
        guidance_scale = guidance_scale + self.constant_guidance - 1
        if self.fbg_mode == "global":
            guidance_scale = guidance_scale.view(B, 1, 1)
        elif self.fbg_mode == "spatial":
            guidance_scale = guidance_scale.view(B, N, 1)
        elif self.fbg_mode == "cluster":
            if cluster_labels is None:
                raise RuntimeError("cluster_labels is required for cluster FBG")
            guidance_scale = torch.gather(guidance_scale, 1, cluster_labels).view(B, N, 1)
        return torch.clamp(guidance_scale, min=1.0, max=self.max_guidance)

    def _fbg_posterior_step(self, prev_log_posterior, diff):
        delta = -0.5 * self.temp * diff
        if self.logp_delta_clip > 0:
            delta = torch.clamp(delta, -self.logp_delta_clip, self.logp_delta_clip)
        logp = prev_log_posterior + delta + self.offset
        return torch.clamp(logp, min=self.minimal_log_posterior, max=3.0)

    def _update_logp_orig_fbg(
        self,
        prev_log_posterior,
        next_sample,
        mu_u,
        mu_c,
        coeff,
        cluster_labels=None,
    ):
        eta2 = coeff["eta2"].to(next_sample.device, next_sample.dtype)
        sigma2_proxy = torch.clamp(eta2.mean(), min=1e-12)

        if self.fbg_mode == "global":
            du = (((next_sample - mu_u) ** 2) / sigma2_proxy).sum(dim=(1, 2))
            dc = (((next_sample - mu_c) ** 2) / sigma2_proxy).sum(dim=(1, 2))
            diff = dc - du
            logp = self._fbg_posterior_step(prev_log_posterior, diff)

        else:
            du = (((next_sample - mu_u) ** 2) / sigma2_proxy).sum(dim=2)  # [B,N]
            dc = (((next_sample - mu_c) ** 2) / sigma2_proxy).sum(dim=2)
            diff = dc - du

            if self.fbg_mode == "spatial":
                logp = self._fbg_posterior_step(prev_log_posterior, diff)

            elif self.fbg_mode == "cluster":
                if cluster_labels is None:
                    raise RuntimeError("cluster_labels is required for cluster original FBG")

                B = diff.shape[0]
                cluster_diff = torch.zeros(B, self.n_clusters, device=self.device, dtype=diff.dtype)
                cluster_diff.scatter_add_(dim=1, index=cluster_labels, src=diff)

                if self.posterior_cluster_average:
                    counts = torch.zeros(B, self.n_clusters, device=self.device, dtype=diff.dtype)
                    ones = torch.ones_like(diff)
                    counts.scatter_add_(dim=1, index=cluster_labels, src=ones)
                    cluster_diff = cluster_diff / counts.clamp(min=1.0)

                logp = self._fbg_posterior_step(prev_log_posterior, cluster_diff)

            else:
                raise ValueError(self.fbg_mode)

        return torch.clamp(logp, min=self.minimal_log_posterior, max=3.0)


    def _update_logp_gad(self, prev_log_posterior, next_sample, mu_u, mu_c, coeff, cluster_labels=None):
        eta2 = coeff["eta2"].to(next_sample.device, next_sample.dtype)
        omega2 = torch.matmul(self.graph.U.to(next_sample.dtype).pow(2), eta2).clamp(min=1e-12)  # [N]
        if self.fbg_mode == "global":
            du = (((next_sample - mu_u) ** 2) / omega2[None, :, None]).sum(dim=(1, 2))
            dc = (((next_sample - mu_c) ** 2) / omega2[None, :, None]).sum(dim=(1, 2))
            diff = dc - du
            logp = self._fbg_posterior_step(prev_log_posterior, diff)
        else:
            du = (((next_sample - mu_u) ** 2) / omega2[None, :, None]).sum(dim=2)  # [B,N]
            dc = (((next_sample - mu_c) ** 2) / omega2[None, :, None]).sum(dim=2)
            diff = dc - du
            if self.fbg_mode == "spatial":
                logp = self._fbg_posterior_step(prev_log_posterior, diff)
            elif self.fbg_mode == "cluster":
                B = diff.shape[0]
                cluster_diff = torch.zeros(B, self.n_clusters, device=self.device, dtype=diff.dtype)
                cluster_diff.scatter_add_(dim=1, index=cluster_labels, src=diff)
                if self.posterior_cluster_average:
                    counts = torch.zeros(B, self.n_clusters, device=self.device, dtype=diff.dtype)
                    ones = torch.ones_like(diff)
                    counts.scatter_add_(dim=1, index=cluster_labels, src=ones)
                    cluster_diff = cluster_diff / counts.clamp(min=1.0)
                logp = self._fbg_posterior_step(prev_log_posterior, cluster_diff)
            else:
                raise ValueError(self.fbg_mode)
        return torch.clamp(logp, min=self.minimal_log_posterior, max=3.0)

    
    def _update_logp_cov_fbg(
        self,
        prev_log_posterior,
        next_sample,
        mu_u,
        mu_c,
        coeff,
        cluster_labels=None,
    ):

        eta2 = coeff["eta2"].to(next_sample.device, next_sample.dtype)
        inv_eta2 = torch.reciprocal(torch.clamp(eta2, min=1e-12))

        r_u = next_sample - mu_u          # [B, N, T]
        r_c = next_sample - mu_c          # [B, N, T]

        r_u_spec = self.graph.to_spectral(r_u)   # [B, N, T]
        r_c_spec = self.graph.to_spectral(r_c)   # [B, N, T]

        if self.fbg_mode == "global":
            du = (
                r_u_spec.pow(2)
                * inv_eta2[None, :, None]
            ).sum(dim=(1, 2))   # [B]

            dc = (
                r_c_spec.pow(2)
                * inv_eta2[None, :, None]
            ).sum(dim=(1, 2))   # [B]

            diff = dc - du

            logp = self._fbg_posterior_step(prev_log_posterior, diff)


        else:
            qinv_r_u_spec = r_u_spec * inv_eta2[None, :, None]
            qinv_r_c_spec = r_c_spec * inv_eta2[None, :, None]

            qinv_r_u = self.graph.from_spectral(qinv_r_u_spec)  # [B,N,T]
            qinv_r_c = self.graph.from_spectral(qinv_r_c_spec)  # [B,N,T]

            du_node = (r_u * qinv_r_u).sum(dim=2)   # [B,N]
            dc_node = (r_c * qinv_r_c).sum(dim=2)   # [B,N]

            diff_node = dc_node - du_node           # [B,N]

            if self.fbg_mode == "spatial":
                logp = self._fbg_posterior_step(prev_log_posterior, diff_node)

            elif self.fbg_mode == "cluster":
                if cluster_labels is None:
                    raise RuntimeError(
                        "cluster_labels is required for cluster covariance-aware FBG"
                    )

                B = diff_node.shape[0]

                cluster_diff = torch.zeros(
                    B,
                    self.n_clusters,
                    device=self.device,
                    dtype=diff_node.dtype,
                )

                cluster_diff.scatter_add_(
                    dim=1,
                    index=cluster_labels,
                    src=diff_node,
                )

                if self.posterior_cluster_average:
                    counts = torch.zeros(
                        B,
                        self.n_clusters,
                        device=self.device,
                        dtype=diff_node.dtype,
                    )
                    ones = torch.ones_like(diff_node)
                    counts.scatter_add_(
                        dim=1,
                        index=cluster_labels,
                        src=ones,
                    )
                    cluster_diff = cluster_diff / counts.clamp(min=1.0)

                logp = self._fbg_posterior_step(prev_log_posterior, cluster_diff)

            else:
                raise ValueError(self.fbg_mode)

        return torch.clamp(
            logp,
            min=self.minimal_log_posterior,
            max=3.0,
        )
    
    
    def _predict_pair_time(
        self,
        current_sample,
        observed_data,
        cond_mask,
        side_info,
        null_side_info,
        noisy_cond_history,
        level_idx: int,
        t_cur: torch.Tensor | float,
    ):
        noisy_observed_uncond = cond_mask * noisy_cond_history[level_idx] + (1.0 - cond_mask) * current_sample
        input_uncond = torch.cat(
            [torch.zeros_like(noisy_observed_uncond).unsqueeze(1), noisy_observed_uncond.unsqueeze(1)], dim=1
        )
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
        input_cond = torch.cat([cond_obs, noisy_target], dim=1)
        t = torch.full(
            (current_sample.shape[0],), float(torch.as_tensor(t_cur).item()),
            device=self.device, dtype=torch.float32,
        )
        coord = self.diffusion_condition(t)
        x0_u, _ = self.diffmodel_uncond(input_uncond, null_side_info, coord)
        x0_c, attn_c = self.diffmodel_cond(input_cond, side_info, coord)
        x0_c_proj = cond_mask * observed_data + (1.0 - cond_mask) * x0_c
        pars = self.channel_params_time(t)
        score_u, score_u_spec = self.score_from_x0_hat(current_sample, x0_u, pars)
        score_c, score_c_spec = self.score_from_x0_hat(current_sample, x0_c_proj, pars)
        return x0_u, x0_c_proj, score_u, score_u_spec, score_c, score_c_spec, attn_c

    def _em_step_time(self, current_sample, score_g_spec, t_cur, t_next, noise: Optional[torch.Tensor] = None):
        mu_g_spec, coeff = self._reverse_mean_spec_time(current_sample, score_g_spec, t_cur, t_next)
        if noise is None:
            noise = torch.randn_like(mu_g_spec)
        eta = torch.sqrt(coeff["eta2"].to(current_sample.device, current_sample.dtype))[None, :, None]
        next_spec = mu_g_spec + eta * noise
        return self.graph.from_spectral(next_spec), mu_g_spec, coeff

    def _heun_step_time(
        self, current_sample, observed_data, cond_mask, side_info, null_side_info,
        noisy_cond_history, level_cur, level_next, t_cur, t_next, score_g_spec,
        guidance_scale=None,
    ):
        mu_pred_spec, coeff = self._reverse_mean_spec_time(current_sample, score_g_spec, t_cur, t_next)
        pred = self.graph.from_spectral(mu_pred_spec)
        if self.clamp_observed:
            pred = cond_mask * observed_data + (1.0 - cond_mask) * pred
        _, _, score_u2, _, score_c2, _, _ = self._predict_pair_time(
            pred, observed_data, cond_mask, side_info, null_side_info,
            noisy_cond_history, level_next, t_next,
        )
        if guidance_scale is None:
            score_g2 = score_c2
        else:
            score_g2 = score_u2 + guidance_scale * (score_c2 - score_u2)
        score_g2_spec = self.graph.to_spectral(score_g2)
        xt_spec = self.graph.to_spectral(current_sample)
        pred_spec = self.graph.to_spectral(pred)
        dt = coeff["dt"].to(current_sample.device, current_sample.dtype)
        c_mode = coeff["c_mode"].to(current_sample.device, current_sample.dtype)
        drift1 = c_mode[None, :, None] * (
            self.lam_gamma[None, :, None].to(current_sample.dtype) * xt_spec
            + 2.0 * (self.sigma ** 2) * score_g_spec
        )
        drift2 = c_mode[None, :, None] * (
            self.lam_gamma[None, :, None].to(current_sample.dtype) * pred_spec
            + 2.0 * (self.sigma ** 2) * score_g2_spec
        )
        next_spec = xt_spec + 0.5 * (drift1 + drift2) * dt
        if self.heun_stochastic:
            eta = torch.sqrt(coeff["eta2"].to(current_sample.device, current_sample.dtype))[None, :, None]
            next_spec = next_spec + eta * torch.randn_like(next_spec)
        return self.graph.from_spectral(next_spec), coeff

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, N, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, N, L, device=self.device, dtype=observed_data.dtype)
        null_side_info = side_info.clone(); null_side_info[:, -1:, :, :] = 0
        sample_times = self.sample_times()  # ascending: clean -> terminal
        noisy_cond_history = self._build_noisy_cond_history(observed_data, cond_mask, sample_times)

        for s in range(n_samples):
            current_sample = self._sample_terminal(observed_data, sample_times[-1])
            if self.uses_fbg:
                if self.fbg_mode == "global":
                    log_posterior = torch.zeros(B, device=self.device)
                elif self.fbg_mode == "spatial":
                    log_posterior = torch.zeros(B, N, device=self.device)
                elif self.fbg_mode == "cluster":
                    log_posterior = torch.zeros(B, self.n_clusters, device=self.device)
                else:
                    raise ValueError(self.fbg_mode)
            cluster_labels = None

            for level_cur in range(self.sample_nfe - 1, 0, -1):
                level_next = level_cur - 1
                t_cur = sample_times[level_cur]
                t_next = sample_times[level_next]
                x0_u, x0_c_proj, score_u, score_u_spec, score_c, score_c_spec, attn_c = self._predict_pair_time(
                    current_sample, observed_data, cond_mask, side_info, null_side_info,
                    noisy_cond_history, level_cur, t_cur,
                )
                if self.uses_fbg and self.fbg_mode == "cluster" and attn_c is not None:
                    with torch.no_grad():
                        cluster_labels = self._kmeans_torch_cluster(attn_c.detach(), self.n_clusters, num_iter=10)

                guidance_scale_for_heun = None
                if self.guidance == "cfg":
                    score_g = score_u + self.cfg_scale * (score_c - score_u)
                    guidance_scale_for_heun = self.cfg_scale
                elif self.uses_fbg:
                    scale = self._guidance_scale(log_posterior, cluster_labels, B, N)
                    score_g = score_u + scale * (score_c - score_u)
                    guidance_scale_for_heun = scale
                else:
                    score_g = score_c
                score_g_spec = self.graph.to_spectral(score_g)

                mu_u_spec, coeff = self._reverse_mean_spec_time(current_sample, score_u_spec, t_cur, t_next)
                mu_c_spec, _ = self._reverse_mean_spec_time(current_sample, score_c_spec, t_cur, t_next)
                mu_u = self.graph.from_spectral(mu_u_spec)
                mu_c = self.graph.from_spectral(mu_c_spec)

                if self.sampler == "em":
                    next_sample, _, coeff = self._em_step_time(current_sample, score_g_spec, t_cur, t_next)
                elif self.sampler == "heun":
                    next_sample, coeff = self._heun_step_time(
                        current_sample, observed_data, cond_mask, side_info, null_side_info,
                        noisy_cond_history, level_cur, level_next, t_cur, t_next,
                        score_g_spec, guidance_scale_for_heun,
                    )
                else:
                    raise ValueError(f"Unknown sampler={self.sampler}. Use em or heun.")

                if self.clamp_observed:
                    next_sample = cond_mask * observed_data + (1.0 - cond_mask) * next_sample

                if self.uses_fbg:
                    with torch.no_grad():
                        if self.original_fbg:
                            log_posterior = self._update_logp_orig_fbg(
                                log_posterior, next_sample, mu_u, mu_c, coeff, cluster_labels
                            )
                        elif self.covariance_aware_fbg:
                            log_posterior = self._update_logp_cov_fbg(
                                log_posterior, next_sample, mu_u, mu_c, coeff, cluster_labels
                            )
                        else:
                            log_posterior = self._update_logp_gad(
                                log_posterior, next_sample, mu_u, mu_c, coeff, cluster_labels
                            )
                current_sample = next_sample

            if self.final_x0_projection:
                x0_u, x0_c_proj, _, _, _, _, attn_c = self._predict_pair_time(
                    current_sample, observed_data, cond_mask, side_info, null_side_info,
                    noisy_cond_history, 0, sample_times[0],
                )
                if self.uses_fbg and self.fbg_mode == "cluster" and attn_c is not None:
                    with torch.no_grad():
                        cluster_labels = self._kmeans_torch_cluster(attn_c.detach(), self.n_clusters, num_iter=10)
                if self.guidance == "cfg":
                    final_x0 = x0_u + self.cfg_scale * (x0_c_proj - x0_u)
                elif self.uses_fbg:
                    scale = self._guidance_scale(log_posterior, cluster_labels, B, N)
                    final_x0 = x0_u + scale * (x0_c_proj - x0_u)
                else:
                    final_x0 = x0_c_proj
                current_sample = cond_mask * observed_data + (1.0 - cond_mask) * final_x0

            imputed_samples[:, s] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1, is_phase1=True):
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, _ = self.process_data(batch, is_train)
        if self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1)

    def evaluate(self, batch, n_samples):
        observed_data, observed_mask, observed_tp, gt_mask, _, cut_length = self.process_data(batch, 0)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class GADMASFENCE_Traffic(GADMASFENCEBase):
    def __init__(self, config, target_dim, device, graph: SpatialGraph, spectral_power: np.ndarray):
        super().__init__(target_dim, config, device, graph, spectral_power)

    def process_data(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask
        return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length
