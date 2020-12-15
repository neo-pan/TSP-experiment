import argparse
import os
import time

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning"
    )

    # Data
    parser.add_argument(
        "--problem", default="tsp", help="The problem to solve, default 'tsp'"
    )
    parser.add_argument(
        "--graph_size", type=int, default=50, help="The size of the problem graph"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=1280000,
        help="Number of instances per epoch during training",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=None,
        help="Dataset file to use for validation",
    )

    # Model
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_embed_layers", type=int, default=2)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--encoder_num_heads", type=int, default=1)
    parser.add_argument("--decoder_num_heads", type=int, default=1)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument("--decode_type", type=str, default="sampling")

    # Training
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Number of instances per batch during evaluating",
    )
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=1e-4,
        help="Set the learning rate for the critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Set this value to only evaluate model"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_batch_size", type=int, default=256)

    # Misc
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable tqdm")
    parser.add_argument(
        "--log_step", type=int, default=50, help="Log info every log_step steps"
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory to write TensorBoard information to",
    )
    parser.add_argument("--run_name", default="run", help="Name to identify the run")
    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )
    parser.add_argument(
        "--load_path",
        default=None,
        help="Path to load model parameters and optimizer state from",
    )

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() if not args.no_cuda else False

    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    args.save_dir = os.path.join(
        args.output_dir, "{}_{}".format(args.problem, args.graph_size), args.run_name
    )

    assert (
        args.epoch_size % args.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"

    return args
