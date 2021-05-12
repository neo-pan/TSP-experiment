import json
import math
import os
import pprint as pp
import time
from datetime import datetime

import numpy as np
import sty
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

import wandb
from args_2opt import get_args
from datasets.tsp import TSPDataset
from environments.tsp_2opt import TSP2OPTEnv
from models.tsp_2opt_agent import TSP2OPTAgent
from rl_algorithms.reinforce_2opt import reinforce_train_batch


def validate(model, dataset, env, args):
    # Validate
    print("Validating...")
    tour_len = rollout(model, dataset, env, args)
    assert tour_len.size(0) == len(dataset)
    avg_len = tour_len.mean()
    print("Validation overall avg_cost: {} +- {}".format(avg_len, torch.std(tour_len) / math.sqrt(len(tour_len))))

    return avg_len


def rollout(model, dataset, env, args):
    # Put in greedy evaluation mode!

    model.set_decode_type("sampling")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            node_pos = to_dense_batch(bat.pos, bat.batch)[0]
            done = False
            state = env.reset(T=args.max_num_steps, node_pos=node_pos)
            embed_data = model.init_embed(bat)
            node_embeddings, _ = model.encoder(embed_data)
            while not done:
                action, _, _ = model(state, node_embeddings, embed_data.batch)
                # adapt costa_decoder outputed action to our environment
                action[:, 0] -= 1
                action %= args.graph_size
                state, _, done, _ = env.step(action.squeeze())

            return state.best_tour_len.cpu()

    return torch.cat(
        [eval_model_bat(bat.to(args.device)) for bat in DataLoader(dataset, batch_size=args.eval_batch_size)], 0,
    )


if __name__ == "__main__":
    args = get_args(None)
    # Pretty print the run args
    pp.pprint(vars(args))

    # Set the random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Optionally configure tensorboard
    tb_logger = SummaryWriter(os.path.join(args.log_dir, "{}_{}".format(args.problem, args.graph_size), args.run_name))

    os.makedirs(args.save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=True)

    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    load_data = None
    # Load data from load_path
    if args.load_path:
        load_path = args.load_path
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch.load(load_path)

    env = TSP2OPTEnv()
    model = TSP2OPTAgent(args).to(args.device)
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr_model}])
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)
    scaler = amp.grad_scaler.GradScaler(enabled=True)

    # init wandb logging
    wandb.init(project="TSP-2opt", entity="neopan", sync_tensorboard=True)
    wandb.config.update(vars(args))
    wandb.watch(model, log="all", log_freq=100)

    # Load saved data
    if load_data:
        model.load_state_dict(load_data["model"])
        optimizer.load_state_dict(load_data["optimizer"])
        torch.set_rng_state(load_data["rng_state"])
        if args.device is torch.device("cuda"):
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
    if args.load_path:
        epoch_resume = int(os.path.splitext(os.path.split(args.load_path)[-1])[0].split("-")[1])
        args.epoch_start = epoch_resume + 1
    val_dataset = TSPDataset(
        size=args.val_size, graph_size=args.graph_size, graph_type=args.graph_type, load_path=args.val_dataset
    )

    if args.eval_only:
        avg_len = validate(model, val_dataset, env, args)
        opt = sum([g.opt for g in val_dataset]) / len(val_dataset)
        opt_gap = (avg_len - opt) / opt
        print(
            sty.fg.green
            + f"Validation, average len: {avg_len:.3f}, opt len:{opt:.3f}, gap: {opt_gap*100:.3f}%"
            + sty.fg.rs
        )
    else:
        learn_count = 0
        for epoch in range(args.epoch_start, args.epoch_start + args.n_epochs):
            # if epoch == 100:
            #     args.horizon = 10
            print(
                "Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]["lr"], args.run_name)
            )
            step = epoch * (args.epoch_size // args.batch_size)
            start_time = time.time()
            tb_logger.add_scalar("learnrate_pg0", optimizer.param_groups[0]["lr"], step)
            training_dataset = TSPDataset(
                size=args.epoch_size,
                graph_size=args.graph_size,
                graph_type=args.graph_type,
                load_path=args.train_dataset,
            )
            training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            model.train()
            model.set_decode_type(args.decode_type)
            for batch_id, batch in enumerate(tqdm(training_dataloader, disable=args.no_progress_bar)):
                learn_count = reinforce_train_batch(
                    model, optimizer, scaler, batch, epoch, batch_id, step, learn_count, env, tb_logger, args,
                )
                step += 1

            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))))

            if (args.checkpoint_epochs != 0 and epoch % args.checkpoint_epochs == 0) or epoch == args.n_epochs - 1:
                print("Saving model and state...")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    },
                    os.path.join(args.save_dir, "epoch-{}.pt".format(epoch)),
                )

            avg_len = validate(model, val_dataset, env, args)
            opt = sum([g.opt for g in val_dataset]) / len(val_dataset)
            opt_gap = (avg_len - opt) / opt
            print(
            sty.fg.green
            + f"Validation, average len: {avg_len:.3f}, opt len:{opt:.3f}, gap: {opt_gap*100:.3f}%"
            + sty.fg.rs
            )
            tb_logger.add_scalar("tour_len_val", avg_len, step)
            tb_logger.add_scalar("gap_val", opt_gap, step)
            wandb.log({"tour_len_val": avg_len, "gap_val": opt_gap})

            # lr_scheduler should be called at end of epoch
            lr_scheduler.step()
