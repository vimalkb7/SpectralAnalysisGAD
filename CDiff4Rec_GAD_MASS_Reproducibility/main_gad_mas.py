from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

import data_utils
import evaluate_utils
from config_gad_mas import get_config
from models.DNN import DNN
from models.Attention import Attention
from models.gad_mas_diffusion import GADMASDiffusion
from utils import keep_topk_values, get_fill_value, load_pickle, set_random_seed


def compute_weights(args, train_data, device, pseudo_data=None):
    r_topk = f_topk = None
    if args.using_real:
        sim = cosine_similarity(train_data)
        np.fill_diagonal(sim, 0.0)
        sim = torch.as_tensor(sim, dtype=torch.float32) / args.tau
        r_topk = keep_topk_values(sim.to(device), k=args.topk, fill_value=get_fill_value(args.r_agg, args.topk)).float()
    if args.using_pseudo:
        f_train = pseudo_data["train_data"]
        sim = cosine_similarity(train_data, f_train)[:, :args.topk_pseudo_user]
        # User and pseudo-user indices are different domains: do not zero a diagonal.
        sim = torch.as_tensor(sim, dtype=torch.float32) / args.tau
        f_topk = keep_topk_values(sim.to(device), k=args.topk, fill_value=get_fill_value(args.f_agg, args.topk)).float()
    return r_topk, f_topk


def normalized_aggregation_weights(args):
    if args.using_real and args.using_pseudo:
        total = args.alpha + args.beta + args.gamma
        if total <= 0:
            raise ValueError("alpha + beta + gamma must be positive")
        return args.alpha / total, args.beta / total, args.gamma / total
    if args.using_real:
        return args.alpha, 1.0 - args.alpha, 0.0
    if args.using_pseudo:
        return args.alpha, 0.0, 1.0 - args.alpha
    return 1.0, 0.0, 0.0


def evaluate(
    model, diffusion, loader, data_te, r_topk, f_topk, f_train_tensor,
    mask_his, topN, n_user, n_item, f_n_user, args, device
):
    model.eval()
    targets = [data_te[i, :].nonzero()[1].tolist() for i in range(mask_his.shape[0])]
    r_predictions = torch.zeros((n_user, n_item), dtype=torch.float32, device=device)
    f_predictions = torch.zeros((f_n_user, n_item), dtype=torch.float32, device=device) if args.using_pseudo and args.training_pseudo else None

    with torch.no_grad():
        for batch in loader:
            if args.training_pseudo:
                (r_x, r_idx), (f_x, f_idx) = batch
                f_pred = diffusion.p_sample(model, f_x.to(device), args.sampling_steps, args.sampling_noise, branch="pseudo", inference_mode=args.inference_mode, partial_start_step=args.partial_start_step)
                f_predictions[f_idx] = f_pred
            else:
                r_x, r_idx = batch
            r_pred = diffusion.p_sample(model, r_x.to(device), args.sampling_steps, args.sampling_noise, branch="real", inference_mode=args.inference_mode, partial_start_step=args.partial_start_step)
            r_predictions[r_idx] = r_pred

        r_info = torch.zeros_like(r_predictions)
        f_info = torch.zeros_like(r_predictions)
        if args.using_real:
            weights = (r_topk / args.topk) if args.r_agg == "att" else r_topk
            r_info = weights.to(device) @ r_predictions
        if args.using_pseudo:
            weights = (f_topk / args.topk) if args.f_agg == "att" else f_topk
            if args.training_pseudo:
                f_info = weights.to(device) @ f_predictions
            else:
                f_info = weights.to(device) @ f_train_tensor

        scores = args.alpha * r_predictions + args.beta * r_info + args.gamma * f_info
        rows, cols = mask_his.nonzero()
        scores[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = -float("inf")
        ranked = torch.topk(scores, topN[-1], dim=1).indices.cpu().numpy().tolist()
    return evaluate_utils.computeTopNAccuracy(targets, ranked, topN)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    set_random_seed(args.random_seed)
    started = datetime.now()
    print(f"Start time: {started:%Y-%m-%d %H:%M:%S}")
    print(f"Device: {device}")
    print(f"Scheduler: {args.scheduler}; objective: {args.objective}")

    dataset_path = Path(args.data_path) / args.dataset
    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
        str(dataset_path / 'train_list.npy'), str(dataset_path / 'valid_list.npy'), str(dataset_path / 'test_list.npy'))
    train_tensor = torch.as_tensor(train_data.toarray(), dtype=torch.float32)
    train_dataset = data_utils.DataDiffusion_method(train_tensor)
    print(f"Dataset {args.dataset}: users={n_user}, items={n_item}, train interactions={train_data.nnz}")

    pseudo_data = None
    f_train_tensor = None
    f_n_user = 0
    if args.using_pseudo:
        pseudo_data = load_pickle(str(dataset_path / 'pseudo_data.pickle'))
        f_sparse = pseudo_data['train_data'].tocsr()[:args.topk_pseudo_user]
        f_train_tensor_cpu = torch.as_tensor(f_sparse.toarray(), dtype=torch.float32)
        f_train_tensor = f_train_tensor_cpu.to(device)
        f_n_user = f_train_tensor_cpu.shape[0]
        if args.training_pseudo:
            train_dataset = data_utils.ConcatDataset(train_dataset, data_utils.DataDiffusion_method(f_train_tensor_cpu))
        print(f"Pseudo users in aggregation: {f_n_user}; denoise pseudo users: {args.training_pseudo}")

    r_topk, f_topk = compute_weights(args, train_data, device, pseudo_data)
    args.alpha, args.beta, args.gamma = normalized_aggregation_weights(args)
    print(f"Aggregation weights: own={args.alpha:.4f}, real={args.beta:.4f}, pseudo={args.gamma:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=args.cuda, shuffle=True)
    eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    if args.tst_w_val:
        tv_tensor = train_tensor + torch.as_tensor(valid_y_data.toarray(), dtype=torch.float32)
        tv_dataset = data_utils.DataDiffusion_method(tv_tensor)
        test_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader = eval_loader
    mask_test = train_data + valid_y_data

    diffusion = GADMASDiffusion(
        graph_asset_path=args.graph_asset_path,
        steps=args.steps,
        device=device,
        scheduler=args.scheduler,
        objective=args.objective,
        sigma=args.sigma,
        r_max=args.r_max,
        r_min=args.r_min,
        t_eps=args.t_eps,
        T_diff=args.T_diff,
        scalar_cbar_min=args.scalar_cbar_min,
        scalar_cbar_max=args.scalar_cbar_max,
        fcps_alpha=args.fcps_alpha,
        fcps_calibration=args.fcps_calibration,
        fcps_cbar_T=args.fcps_cbar_T,
        fcps_cmin=args.fcps_cmin,
        reweight=args.mas_reweight,
        dsm_variance_weighted=args.dsm_variance_weighted,
    ).to(device)
    if diffusion.n_item != n_item:
        raise ValueError(f"Graph assets contain {diffusion.n_item} items but dataset contains {n_item}")
    print("Diffusion diagnostics:", json.dumps(diffusion.diagnostics(), indent=2))
    if diffusion.graph_diagnostics:
        print("Graph components after bridging:", diffusion.graph_diagnostics.get("components_after_bridging"))

    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    att_model: Optional[Attention] = None
    if args.r_agg == 'att' or args.f_agg == 'att':
        att_model = Attention(n_item, args.topk).to(device)
        optimizer = optim.AdamW([{"params": model.parameters()}, {"params": att_model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_recall = -float("inf")
    best_epoch = 0
    best_state = None
    best_att_state = None
    best_valid = best_test = None
    topN = eval(args.topN)

    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= args.patience:
            print(f"Early stop: no Recall@20 improvement for {args.patience} epochs")
            break
        model.train()
        if att_model is not None:
            att_model.train()
        start = time.time()
        total_loss = total_r_loss = total_f_loss = 0.0

        for batch in train_loader:
            if args.training_pseudo:
                (r_x, r_idx), (f_x, f_idx) = batch
                f_ctx = diffusion.get_output(model, f_x.to(device), branch="pseudo", t_sampling=args.t_sampling, low_time_prob=args.low_time_prob, low_time_max_step=args.low_time_max_step)
                f_loss = diffusion.get_losses(f_ctx["target_x0"], f_ctx["pred_x0"], f_ctx)["loss"].mean()
            else:
                r_x, r_idx = batch
                f_idx = None
                f_ctx = None
                f_loss = torch.zeros((), device=device)

            r_ctx = diffusion.get_output(model, r_x.to(device), branch="real", t_sampling=args.t_sampling, low_time_prob=args.low_time_prob, low_time_max_step=args.low_time_max_step)
            r_pred = r_ctx["pred_x0"]
            r_info = torch.zeros_like(r_pred)
            f_info = torch.zeros_like(r_pred)

            if args.using_real:
                local_weights = r_topk[r_idx][:, r_idx].to(device)
                if args.r_agg in {'avg', 'sim'}:
                    r_info = local_weights @ r_pred
                else:
                    r_info = att_model(x1=r_pred, label_map=local_weights)
            if args.using_pseudo:
                if args.training_pseudo:
                    local_weights = f_topk[r_idx][:, f_idx].to(device)
                    if args.f_agg in {'avg', 'sim'}:
                        f_info = local_weights @ f_ctx["pred_x0"]
                    else:
                        f_info = att_model(x1=r_pred, x2=f_ctx["pred_x0"], label_map=local_weights)
                else:
                    if args.f_agg == 'att':
                        raise ValueError("f_agg=att requires --training_pseudo")
                    f_info = f_topk[r_idx].to(device) @ f_train_tensor

            collaborative_pred = args.alpha * r_pred + args.beta * r_info + args.gamma * f_info
            r_loss = diffusion.get_losses(r_ctx["target_x0"], collaborative_pred, r_ctx)["loss"].mean()
            loss = r_loss + args.lamda * f_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_r_loss += float(r_loss.detach().cpu())
            total_f_loss += float((args.lamda * f_loss).detach().cpu())

        should_evaluate = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        if should_evaluate:
            valid_result = evaluate(model, diffusion, eval_loader, valid_y_data, r_topk, f_topk, f_train_tensor,
                                    train_data, topN, n_user, n_item, f_n_user, args, device)
            test_result = evaluate(model, diffusion, test_loader, test_y_data, r_topk, f_topk, f_train_tensor,
                                   mask_test, topN, n_user, n_item, f_n_user, args, device)
            evaluate_utils.print_results(total_loss, valid_result, test_result)
            recall20 = valid_result[1][topN.index(20)]
            if recall20 > best_recall:
                best_recall = recall20
                best_epoch = epoch
                best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best_att_state = deepcopy({k: v.detach().cpu() for k, v in att_model.state_dict().items()}) if att_model else None
                best_valid, best_test = valid_result, test_result
        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print(f"Epoch {epoch:03d}: total={total_loss:.5f} real={total_r_loss:.5f} pseudo={total_f_loss:.5f}; time={elapsed}; best_epoch={best_epoch}")

    result_payload = {
        "args": vars(args),
        "scheduler_diagnostics": diffusion.diagnostics(),
        "graph_diagnostics": diffusion.graph_diagnostics,
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_test": best_test,
        "finished": datetime.now().isoformat(),
    }
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result_payload, indent=2))
    if args.save and best_state is not None:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "attention": best_att_state, "args": vars(args),
                    "best_valid": best_valid, "best_test": best_test}, save_dir / "best_by_val_recall20.pth")
        print(f"Saved checkpoint: {save_dir / 'best_by_val_recall20.pth'}")
    print(f"Saved results: {results_path}")
    print(f"Best epoch: {best_epoch}")
    evaluate_utils.print_results(None, best_valid, best_test)


if __name__ == '__main__':
    main(get_config())
