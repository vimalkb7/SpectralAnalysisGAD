import argparse


def get_config():
    parser = argparse.ArgumentParser(description="GAD-MAS-CDiff4Rec")
    parser.add_argument('--dataset', type=str, default='amazon_game')
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--graph_asset_path', type=str, default='./graph_assets/amazon_game_hybrid_k20_gad_mas.npz')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs; exact spectral sampling is costlier than CDiff4Rec')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='Mask validation interactions while testing')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save', '-s', action='store_true')
    parser.add_argument('--save_path', type=str, default='./saved_models_gad_mas/amazon_game')
    parser.add_argument('--results_path', type=str, default='./results/amazon_game_gad_mas.json')
    parser.add_argument('--random_seed', '--rs', type=int, default=1)

    # Existing CDiff4Rec DNN backbone.
    parser.add_argument('--dims', type=str, default='[1000]')
    parser.add_argument('--norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--emb_size', type=int, default=10)

    # GAD/MAS-SNR diffusion.
    parser.add_argument('--scheduler', choices=['mas_snr', 'scalar_gad', 'fcps'], default='mas_snr')
    parser.add_argument('--objective', choices=['x0_mse', 'spectral_dsm'], default='x0_mse')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--sampling_steps', type=int, default=50,
                        help='Number of reverse levels used; <=0 uses all configured steps')
    parser.add_argument('--sampling_noise', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--r_max', type=float, default=1e4)
    parser.add_argument('--r_min', type=float, default=1e-4)
    parser.add_argument('--t_eps', type=float, default=1e-3)
    parser.add_argument('--T_diff', type=float, default=1.0)
    parser.add_argument('--scalar_cbar_min', type=float, default=1e-5)
    parser.add_argument('--scalar_cbar_max', type=float, default=6.0)

    # FCPS scalar schedule (--scheduler fcps): cbar_t = c_min t + (cbar_T - c_min T)(t/T)^(alpha+1).
    # snr_matched derives cbar_T and c_min from --r_max/--r_min so that both ends
    # of the horizon sit at the MASS corruption levels; only alpha is free.
    parser.add_argument('--fcps_alpha', type=float, default=3.0,
                        help='FCPS polynomial exponent alpha (shape of the path between the matched endpoints)')
    parser.add_argument('--fcps_calibration', choices=['snr_matched', 'manual'], default='snr_matched',
                        help='snr_matched: endpoints from --r_max/--r_min; manual: use --fcps_cbar_T and --fcps_cmin')
    parser.add_argument('--fcps_cbar_T', type=float, default=None, help='Terminal accumulated schedule (manual only)')
    parser.add_argument('--fcps_cmin', type=float, default=None, help='Rate floor c_min (manual only)')
    parser.add_argument('--mas_reweight', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dsm_variance_weighted', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--t_sampling', choices=['uniform', 'importance', 'low_time_mix', 'low_time_only'], default='uniform',
                        help='Training timestep sampler. low_time_mix aligns training with recommendation inference.')
    parser.add_argument('--low_time_prob', type=float, default=0.70,
                        help='Probability of sampling a low diffusion step when --t_sampling low_time_mix.')
    parser.add_argument('--low_time_max_step', type=int, default=3,
                        help='Largest low diffusion step used by low_time_mix / low_time_only.')

    # CDiff4Rec collaborative aggregation. Use omega names internally to avoid
    # conflict with Laplacian shift gamma already embedded in graph assets.
    parser.add_argument('--alpha', type=float, default=0.8, help='Own preference aggregation weight')
    parser.add_argument('--beta', type=float, default=0.15, help='Real-neighbour aggregation weight')
    parser.add_argument('--gamma', type=float, default=0.05, help='Pseudo-user aggregation weight')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--topk_pseudo_user', type=int, default=1000)
    parser.add_argument('--using_real', '--ur', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--using_pseudo', '--uf', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--training_pseudo', '--tf', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--r_agg', choices=['avg', 'sim', 'att'], default='avg')
    parser.add_argument('--f_agg', choices=['avg', 'sim', 'att'], default='sim')

    parser.add_argument("--inference_mode", type=str, default="clean_direct", choices=["clean_direct", "paper_compatible", "partial_corruption", "terminal_prior", "public_code_compatible",],)
    parser.add_argument("--partial_start_step", type=int, default=0,)
    
    parser.add_argument('--lamda', type=float, default=0.001, help='Pseudo-user denoising loss weight when --tf')
    return parser.parse_args()
