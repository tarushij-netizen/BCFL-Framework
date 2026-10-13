import logging
import argparse
import random

from task import Task
import config.benchmark
from config.algorithm import RobustFL
from config.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()

#global_args, train_args = config.benchmark.FashionMNIST().get_args()
#global_args, train_args, algorithm = config.benchmark.Sign().get_args()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default="FashionMNIST", help="Running Benchmark(See ./config/benchmark.py)")
    # --- defense experiments (all optional; with none of these the original baseline runs unchanged) ---
    parser.add_argument('--aggregator', choices=['fedavg', 'similarity', 'median', 'krum', 'sketch', 'repsketch'], default=None,
                        help="Use the robust-FL pipeline with this aggregator")
    parser.add_argument('--attack', choices=['none', 'signflip', 'noise', 'freerider', 'labelflip', 'backdoor'],
                        default='none')
    parser.add_argument('--backdoor-target', type=int, default=0, help="backdoor: class the trigger maps to")
    parser.add_argument('--poison-frac', type=float, default=0.5,
                        help="backdoor: share of each malicious batch that gets the trigger")
    parser.add_argument('--num-malicious', type=int, default=0, help="First N clients are attackers")
    parser.add_argument('--attack-scale', type=float, default=5.0)
    parser.add_argument('--sim-threshold', type=float, default=0.1, help="similarity: min cosine")
    parser.add_argument('--mag-ratio', type=float, default=3.0, help="similarity: max norm ratio")
    parser.add_argument('--sketch-dim', type=int, default=64,
                        help="sketch (C1): size of the CountSketch the server screens on")
    parser.add_argument('--class-check', action='store_true',
                        help="sketch/repsketch: also reject clients whose output-layer class profile is an outlier")
    parser.add_argument('--class-mad', type=float, default=3.0, help="class check: robust z-score cut-off")
    parser.add_argument('--rep-init', type=float, default=0.75, help="repsketch (C3): starting reputation")
    parser.add_argument('--rep-beta', type=float, default=0.7, help="repsketch: memory (higher = slower change)")
    parser.add_argument('--rep-tau', type=float, default=0.6, help="repsketch: minimum reputation to take part")
    parser.add_argument('--rep-penalty', type=float, default=0.5, help="repsketch: reputation multiplier on rejection")
    parser.add_argument('--dp-clip', type=float, default=None,
                        help="Local DP: clip each honest client's update to this L2 norm")
    parser.add_argument('--dp-sigma', type=float, default=0.0,
                        help="Local DP: Gaussian noise multiplier (noise std = sigma * clip)")
    parser.add_argument('--onchain', action='store_true',
                        help="C2: record pseudonymous commitments, verdicts, reputation and rewards on the chain")
    parser.add_argument('--epoch-len', type=int, default=5, help="C2: rounds per reputation/reward epoch")
    parser.add_argument('--rounds', type=int, default=None, help="Override communication rounds")
    parser.add_argument('--non-iid', action='store_true', help="Dirichlet split")
    parser.add_argument('--alpha', type=float, default=None, help="Dirichlet alpha (smaller = more skewed)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-client-test', action='store_true',
                        help="Skip per-client test each round (faster; global test acc is still logged)")
    parser.add_argument('--results', type=str, default=None, help="CSV path for per-round results")
    args = parser.parse_args()

    logger.info(f"Get benchmark {args.benchmark}")
    benchmark = config.benchmark.get_benchmark(args.benchmark)
    global_args, train_args, algorithm = benchmark.get_args()

    if args.seed is not None:
        import numpy as np, torch
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.rounds is not None:
        global_args['communication_round'] = args.rounds
    if args.non_iid:
        global_args['non-iid'] = True
    if args.alpha is not None:
        global_args['alpha'] = args.alpha
    global_args['onchain'] = args.onchain
    global_args['epoch_len'] = args.epoch_len
    global_args['backdoor_target'] = args.backdoor_target
    global_args['poison_frac'] = args.poison_frac
    if args.no_client_test:
        global_args['client_test'] = False

    if args.aggregator is not None:
        agg_kwargs = {}
        if args.aggregator == 'similarity':
            agg_kwargs = {'similarity_threshold': args.sim_threshold, 'magnitude_ratio': args.mag_ratio}
        elif args.aggregator in ('sketch', 'repsketch'):
            agg_kwargs = {'sketch_dim': args.sketch_dim, 'similarity_threshold': args.sim_threshold,
                          'magnitude_ratio': args.mag_ratio, 'seed': args.seed or 0,
                          'class_check': args.class_check, 'class_mad': args.class_mad}
            if args.aggregator == 'repsketch':
                agg_kwargs.update({'reputation': True, 'rep_init': args.rep_init,
                                   'rep_beta': args.rep_beta, 'rep_tau': args.rep_tau,
                                   'rep_penalty': args.rep_penalty})
        elif args.aggregator == 'krum':
            agg_kwargs = {'num_byzantine': max(args.num_malicious, 1)}
        algorithm = RobustFL(aggregator=args.aggregator,
                             attack=args.attack,
                             malicious=range(args.num_malicious),
                             attack_scale=args.attack_scale,
                             seed=args.seed or 0,
                             agg_kwargs=agg_kwargs,
                             dp_clip=args.dp_clip,
                             dp_sigma=args.dp_sigma)
        name = f"{args.benchmark}_{args.aggregator}_{args.attack}{args.num_malicious}"
        if args.aggregator in ('sketch', 'repsketch'):
            name += f"_k{args.sketch_dim}"
            if args.class_check:
                name += f"_cls{args.class_mad:g}"
        if args.dp_clip:
            name += f"_dp{args.dp_clip:g}-{args.dp_sigma:g}"
            from server.aggregation_alg.robustAgg import dp_epsilon
            eps = dp_epsilon(args.dp_sigma, global_args['communication_round'])
            print(f"DP: clip {args.dp_clip}, sigma {args.dp_sigma}, "
                  f"epsilon after {global_args['communication_round']} rounds = {eps:.2f} (delta 1e-5)")
        if args.seed is not None:
            name += f"_seed{args.seed}"
        global_args['results_csv'] = args.results or f"results/{name}.csv"
        logger.info(f"Robust FL run: {name}")
        print(f"Run: {name} -> {global_args['results_csv']}")
    elif args.results:
        global_args['results_csv'] = args.results

    logger.info("--training start--")
    logger.info("Get Global args dataset: %s, model: %s",global_args['dataset'], global_args['model'])
    classification_task = Task(global_args=global_args, train_args=train_args, algorithm=algorithm)
    classification_task.run()
