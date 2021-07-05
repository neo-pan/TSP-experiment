import argparse
import os
import time

import torch


def largest_power_of_two(n):
    return 1 << (n.bit_length() - 1)


def get_args(args):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning"
    )

    # Data
    parser.add_argument("--problem", default="tsp", help="The problem to solve, default 'tsp'")
    parser.add_argument("--graph_size", type=int, default=50, help="The size of the problem graph")
    parser.add_argument(
        "--graph_type", type=str, default="knn", choices=["complete", "knn"], help="Graph type to use during training.",
    )
    parser.add_argument(
        "--graph_knn", type=int, default=10, help="K for knn graph"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--epoch_size", type=int, default=5120, help="Number of instances per epoch during training",
    )
    parser.add_argument(
        "--val_size", type=int, default=256, help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset", type=str, default=None, help="Dataset file to use for validation",
    )
    parser.add_argument(
        "--train_dataset", type=str, default=None, help="Dataset file to use for training",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Numbers of CPUs used for generating dataset")

    # Model
    parser.add_argument("--node_dim", type=int, default=2)
    parser.add_argument("--edge_dim", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--tour_gnn_layers", type=int, default=1)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--tanh_clipping", type=float, default=10.0)
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument("--tour_pooling_method", type=str, default="add")
    parser.add_argument("--decode_type", type=str, default="sampling")
    parser.add_argument(
        "--normalization", default="batch", help="Normalization type, 'batch' (default) or 'instance'",
    )

    # Training
    parser.add_argument("--max_num_steps", type=int, default=200, help="Max number of steps of TSP2OPTEnv")
    parser.add_argument(
        "--horizon", type=int, default=10, help="Horizon length of RL agent to compute rewards and update"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1024, help="Number of instances per batch during evaluating",
    )
    parser.add_argument(
        "--lr_model", type=float, default=0.0001, help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic", type=float, default=0.0001, help="Set the learning rate for the critic network",
    )
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch")
    parser.add_argument("--eval_only", action="store_true", help="Set this value to only evaluate model")
    parser.add_argument("--n_epochs", type=int, default=200, help="The number of epochs to train")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_batch_size", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument(
        "--exp_beta", type=float, default=0.8, help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument("--no_norm_return", default=False, action="store_true", help="Disable normalised returns")
    parser.add_argument("--gamma", type=float, default=0.99, help="RL rewards decay")
    parser.add_argument("--value_beta", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--entropy_beta", type=float, default=0.005, help="Entropy loss weight")

    # Misc
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable tqdm")
    parser.add_argument("--log_step", type=int, default=10, help="Log info every log_step steps")
    parser.add_argument(
        "--log_dir", default="logs", help="Directory to write TensorBoard information to",
    )
    parser.add_argument("--run_name", default="run", help="Name to identify the run")
    parser.add_argument("--output_dir", default="outputs", help="Directory to write output models to")
    parser.add_argument(
        "--epoch_start", type=int, default=0, help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )
    parser.add_argument(
        "--load_path", default=None, help="Path to load model parameters and optimizer state from",
    )

    args = parser.parse_args(args)
    args.use_cuda = torch.cuda.is_available() if not args.no_cuda else False
    args.num_workers = min(args.num_workers, largest_power_of_two(os.cpu_count()))

    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    args.save_dir = os.path.join(args.output_dir, "{}_{}".format(args.problem, args.graph_size), args.run_name)

    assert args.epoch_size % args.batch_size == 0, "Epoch size must be integer multiple of batch size!"

    return args
